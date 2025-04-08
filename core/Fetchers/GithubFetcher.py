import asyncio
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Annotated, Union

import httpx
import pandas as pd
import QuantLib as ql
import tqdm
import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

from core.Fetchers.BaseFetcher import BaseFetcher
from core.Products.CurveBuilding.Swaptions.ql_cube_building_utils import build_ql_interpolated_vol_cube, build_ql_sabr_vol_cube

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


MONKEY_CUBE_PATH_DIR = "yieldcurvemonkey/MONKEY_CUBE/refs/heads/main"
MONKEY_CUBE_SABR_PARAMS_KEY = "sabr_params"
MONKEY_CUBE_VOL_CUBE_KEY = "vol_cube_dict_df"

MonkeyCubeValue = Dict[
    Annotated[str, "vol_cube_dict_df or MONKEY_CUBE_SABR_PARAMS_KEY"], Union[Dict[int, pd.DataFrame], Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]]]
]


class GithubFetcher(BaseFetcher):
    _monkey_cube_cache: Dict[datetime, MonkeyCubeValue] = None

    def __init__(
        self,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )
        self._monkey_cube_cache = {}

    def _github_headers(self, path: str):
        return {
            "authority": "raw.githubusercontent.com",
            "method": "GET",
            "path": path,
            "scheme": "https",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "dnt": "1",
            "pragma": "no-cache",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

    async def _fetch_github_json_file(
        self,
        client: httpx.AsyncClient,
        path_dir: str,
        file_name: str,
        json_format_func: Callable[[Dict[str, Any]], Any],
        token: Optional[str] = None,
        uid: Optional[Any] = None,
        max_retries: Optional[int] = 1,
        backoff_factor: Optional[int] = 1,
    ) -> Any:
        github_path = f"/{path_dir}/{file_name}"
        url = f"https://raw.githubusercontent.com{github_path}"
        if token:
            url = f"{url}?token={token}"

        headers = self._github_headers(path=github_path)
        retries = 0
        try:
            while retries < max_retries:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    if uid:
                        return github_path, json_format_func(response.json()), uid
                    return github_path, json_format_func(response.json())

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"Github - Bad Status for {github_path}: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return github_path, None, uid
                        return github_path, None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Github - Throttled for {github_path}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Github - Error for {github_path}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Barchart Intraday - Throttled for {github_path}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Github - Max retries exceeded for {github_path}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return github_path, None, uid
            return github_path, None

    async def _fetch_github_json_file_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_github_json_file(*args, **kwargs)

    def fetch_USD_MONKEY_CUBE_data(
        self,
        dates: List[datetime],
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ) -> List[Tuple[str, Dict[str, Dict[int, pd.DataFrame] | Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]]], datetime]]:
        def format_monkey_cube_dict(json_response: Dict) -> Dict[str, Dict[int, pd.DataFrame] | Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]]]:
            vol_cube_dict_df = {}
            for key, vol_grid in json_response.items():
                if key == MONKEY_CUBE_SABR_PARAMS_KEY:
                    continue
                curr_vol_grid_df = pd.DataFrame(vol_grid)
                if "Expiry" in curr_vol_grid_df.columns:
                    curr_vol_grid_df = curr_vol_grid_df.set_index("Expiry")
                vol_cube_dict_df[float(key)] = curr_vol_grid_df

            monkey_cube_sabr_params: Dict[Annotated[str, "Swaption Tenor"], Dict[Annotated[str, "alpha, beta, nu, rho"], float]] = copy.deepcopy(
                json_response[MONKEY_CUBE_SABR_PARAMS_KEY]
            )
            for swaption_tenor, curr_sabr_params in monkey_cube_sabr_params.items():
                curr_sabr_params["alpha"] = curr_sabr_params["alpha"] / 100
                curr_sabr_params["nu"] = curr_sabr_params["nu"] / 100

            return {"vol_cube_dict_df": vol_cube_dict_df, MONKEY_CUBE_SABR_PARAMS_KEY: monkey_cube_sabr_params}

        async def build_fetch_tasks(
            client: httpx.AsyncClient,
            dates: List[datetime],
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_github_json_file_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    path_dir=MONKEY_CUBE_PATH_DIR,
                    file_name=f"{date.strftime("%Y-%m-%d")}.json",
                    json_format_func=format_monkey_cube_dict,
                    uid=date,
                )
                for date in dates
            ]

            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING NVOLS FROM MONKEY CUBE...")
            return await asyncio.gather(*tasks)

        async def run_fetch_all(dates: List[datetime]):
            limits = httpx.Limits(max_connections=max_concurrent_tasks, max_keepalive_connections=max_keepalive_connections)
            async with httpx.AsyncClient(limits=limits, timeout=self._global_timeout, mounts=self._httpx_proxies, verify=False, http2=True) as client:
                all_data = await build_fetch_tasks(client=client, dates=dates)
                return all_data

        fetch_results = asyncio.run(run_fetch_all(dates=dates))
        return fetch_results

    def fetch_USD_MONKEY_CUBE_cache(
        self,
        dates: List[datetime],
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ) -> Dict[datetime, MonkeyCubeValue]:
        missing_dates = [date for date in dates if date not in self._monkey_cube_cache]

        if missing_dates:
            fetch_results = self.fetch_USD_MONKEY_CUBE_data(
                dates=missing_dates,
                show_tqdm=show_tqdm,
                max_concurrent_tasks=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            for _, formatted_monkey_cube_dict, dt in fetch_results:
                if formatted_monkey_cube_dict is not None:
                    self._monkey_cube_cache[dt] = formatted_monkey_cube_dict

        return {date: self._monkey_cube_cache[date] for date in dates if date in self._monkey_cube_cache}

    def fetch_USD_MONKEY_CUBE_dict_df(
        self,
        dates: List[datetime],
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ):
        fetch_results = self.fetch_USD_MONKEY_CUBE_cache(
            dates=dates, show_tqdm=show_tqdm, max_concurrent_tasks=max_concurrent_tasks, max_keepalive_connections=max_keepalive_connections
        )
        vol_cube_ts_dict = {}
        for dt, curr_monkey_cube_value in fetch_results.items():
            vol_cube_ts_dict[dt] = curr_monkey_cube_value[MONKEY_CUBE_VOL_CUBE_KEY]
        return vol_cube_ts_dict

    def fetch_USD_MONKEY_CUBE_ql_vol_cubes(
        self,
        dates: List[datetime],
        ql_discount_curve_dict: Dict[datetime, ql.DiscountCurve],
        use_sabr: Optional[bool] = False,
        precalibrate_cube: Optional[bool] = False,
        n_jobs_vol_cube_build: Optional[int] = -1,
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ) -> Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
        vol_cube_ts_dict = self.fetch_USD_MONKEY_CUBE_dict_df(
            dates=dates, show_tqdm=show_tqdm, max_concurrent_tasks=max_concurrent_tasks, max_keepalive_connections=max_keepalive_connections
        )
        vol_cube_build_results = {}
        max_workers = n_jobs_vol_cube_build if n_jobs_vol_cube_build > 0 else None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    build_usd_monkey_cube_ql_vol_handle,
                    dt,
                    curr_vol_cube_dict["vol_cube_dict_df"],
                    ql_discount_curve_dict[dt],
                    curr_vol_cube_dict[MONKEY_CUBE_SABR_PARAMS_KEY] if use_sabr else None,
                    precalibrate_cube,
                ): dt
                for dt, curr_vol_cube_dict in vol_cube_ts_dict.items()
            }
            if show_tqdm:
                with tqdm.tqdm(total=len(futures), desc="BUILDING QL VOL CUBES...", unit="cube", leave=True) as pbar:
                    for future in as_completed(futures):
                        dt, vol_cube = future.result()
                        if vol_cube is not None:
                            vol_cube_build_results[dt] = vol_cube
                        pbar.update(1)
            else:
                for future in as_completed(futures):
                    dt, vol_cube = future.result()
                    if vol_cube is not None:
                        vol_cube_build_results[dt] = vol_cube

        return vol_cube_build_results

    def fetch_USD_MONKEY_CUBE_sabr_params(
        self,
        dates: List[datetime],
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ) -> Dict[datetime, Dict[Annotated[str, "Swaption Tenor"], Dict[Annotated[str, "alpha, beta, nu, rho"], float]]]:
        fetch_results = self.fetch_USD_MONKEY_CUBE_cache(
            dates=dates, show_tqdm=show_tqdm, max_concurrent_tasks=max_concurrent_tasks, max_keepalive_connections=max_keepalive_connections
        )
        sabr_params = {}
        for dt, curr_monkey_cube_value in fetch_results.items():
            sabr_params[dt] = curr_monkey_cube_value[MONKEY_CUBE_SABR_PARAMS_KEY]
        return sabr_params


def build_usd_monkey_cube_ql_vol_handle(
    as_of_date: datetime,
    vol_cube_dict_df: Dict[int, pd.DataFrame],
    ql_discount_curve: ql.DiscountCurve,
    sabr_params_dict: Optional[Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]]] = None,
    precalibrate_cube: Optional[bool] = False,
) -> Tuple[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
    curr_monkey_cube_ql_on_index = ql.OvernightIndexedSwapIndex(
        "USD-SOFR-OIS Compound", ql.Period("1D"), 2, ql.USDCurrency(), ql.Sofr(ql.YieldTermStructureHandle(ql_discount_curve))
    )
    if sabr_params_dict is not None:
        ql_vol_cube = build_ql_sabr_vol_cube(
            vol_cube=vol_cube_dict_df,
            sabr_params_dict=sabr_params_dict,
            ql_swap_index=curr_monkey_cube_ql_on_index,
            ql_calendar=ql.UnitedStates(ql.UnitedStates.SOFR),
            ql_day_counter=ql.Actual360(),
            ql_bday_convention=ql.ModifiedFollowing,
            pre_calibrate=precalibrate_cube,
            enable_extrapolation=True,
        )
    else:
        ql_vol_cube = build_ql_interpolated_vol_cube(
            vol_cube=vol_cube_dict_df,
            ql_swap_index=curr_monkey_cube_ql_on_index,
            ql_calendar=ql.UnitedStates(ql.UnitedStates.SOFR),
            ql_day_counter=ql.Actual360(),
            ql_bday_convention=ql.ModifiedFollowing,
            pre_calibrate=precalibrate_cube,
            enable_extrapolation=True,
        )

    return as_of_date, ql_vol_cube
