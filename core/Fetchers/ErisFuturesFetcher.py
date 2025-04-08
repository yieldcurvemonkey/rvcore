import asyncio
import calendar
import ssl
import warnings
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
import pandas as pd
import pytz
import QuantLib as ql
import tqdm
import tqdm.asyncio
from dateutil import parser, tz
from pandas.errors import DtypeWarning
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from core.Fetchers.BaseFetcher import BaseFetcher
from core.Products.CurveBuilding.ql_curve_building_utils import build_ql_discount_curve
from core.utils.ql_utils import get_bdates_between

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


def datetime_today_utc():
    return datetime(
        year=datetime.now(timezone.utc).year,
        month=datetime.now(timezone.utc).month,
        day=datetime.now(timezone.utc).day,
    )


class ErisFuturesFetcher(BaseFetcher):
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
        self.eris_ftp_urls = "https://files.erisfutures.com/ftp"
        self.eris_ftp_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Host": "files.erisfutures.com",
            "Referer": "https://files.erisfutures.com/ftp/",
            "Sec-CH-UA": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }

    async def _fetch_eris_ftp_files_helper(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR", "Eris_Intraday_DiscountFactors_SOFR"],
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        def diff_month(d1, d2):
            return (d1.year - d2.year) * 12 + d1.month - d2.month

        if "Intraday" in workbook_type:
            eris_ftp_formatted_url = "https://files.erisfutures.com/ftp/Eris_Intraday_DiscountFactors_SOFR.csv"
            file_name = "Eris_Intraday_DiscountFactors_SOFR.csv"
        else:
            archives_path = f"archives/{date.year}/{date.month:02}-{calendar.month_name[date.month]}"
            file_name = f"Eris_{date.strftime("%Y%m%d")}_{workbook_type}.csv"
            if diff_month(datetime.today(), date) < 3:
                eris_ftp_formatted_url = f"{self.eris_ftp_urls}/{file_name}"
            else:
                eris_ftp_formatted_url = f"{self.eris_ftp_urls}/{archives_path}/{file_name}"

        retries = 0
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            while retries < max_retries:
                try:
                    async with client.stream(
                        method="GET",
                        url=eris_ftp_formatted_url,
                        headers=self.eris_ftp_headers,
                        follow_redirects=True,
                        timeout=self._global_timeout,
                    ) as response:
                        response.raise_for_status()
                        buffer = BytesIO()
                        async for chunk in response.aiter_bytes():
                            buffer.write(chunk)
                        buffer.seek(0)

                    return buffer, file_name

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"ERIS FTP - Bad Status for {workbook_type}-{date}: {response.status_code}")
                    if response.status_code == 404:
                        return None, None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"ERIS FTP - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"ERIS FTP - Error for {workbook_type}-{date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"ERIS FTP - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"ERIS FTP - Max retries exceeded for {workbook_type}-{date}")

        except Exception as e:
            print(e)
            self._logger.error(e)
            return None, None

    def _read_file(self, file_buffer: BytesIO, file_name: str) -> Tuple[Union[str, datetime], pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)

            if file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_buffer)
            elif file_name.lower().endswith(".csv"):
                df = pd.read_csv(file_buffer, low_memory=False)
            else:
                return None

            try:
                datetime.strptime(file_name.split("_")[1], "%Y%m%d")
                key = datetime.strptime(file_name.split("_")[1], "%Y%m%d")
            except:
                key = file_name

            return key, df

    async def _fetch_and_read_eris_ftp_file(
        self,
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR", "Eris_Intraday_DiscountFactors_SOFR", "EOD_DiscountFactors_SOFR"],
        task_id: Optional[Any] = None,
    ):
        async with semaphore:
            buffer, file_name = await self._fetch_eris_ftp_files_helper(client=client, date=date, workbook_type=workbook_type)
            if not buffer or not file_name:
                return None, None

        key, df = await asyncio.to_thread(self._read_file, buffer, file_name)
        if task_id:
            return key, df, task_id
        return key, df

    def fetch_eris_ftp_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        workbook_type: Literal["EOD_ParCouponCurve_SOFR", "EOD_DiscountFactors_SOFR"] = "EOD_ParCouponCurve_SOFR",
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        verbose: Optional[bool] = False,
    ) -> Dict[datetime, pd.DataFrame]:

        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))

        async def build_tasks(
            client: httpx.AsyncClient,
            dates: List[datetime],
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for date in dates:
                task = asyncio.create_task(self._fetch_and_read_eris_ftp_file(semaphore=semaphore, client=client, date=date, workbook_type=workbook_type))
                tasks.append(task)

            return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING ERIS FTP Files...")

        async def run_fetch_all(
            dates: List[datetime],
        ):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits, verify=False, http2=True) as client:
                all_data = await build_tasks(
                    client=client,
                    dates=dates,
                )
                return all_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
                run_fetch_all(
                    dates=bdates,
                )
            )
            if results is None or len(results) == 0:
                print('"fetch_eris_ftp_timeseries" --- empty results') if verbose else None
                return {}

            return dict(results)

    def fetch_historical_eod_discount_curves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        ql_dc=ql.Actual360(),
        ql_cal=ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        show_tqdm: Optional[bool] = True,
        interpolation_algo: Optional[
            List[
                Literal[
                    "log_linear",
                    "mono_log_cubic",
                    "natural_cubic",
                    "kruger_log",
                    "natural_log_cubic",
                    "log_mixed_linear",
                    "log_parabolic_cubic",
                    "mono_log_parabolic_cubic",
                ]
            ]
        ] = "log_linear",
        enable_extrapolation: Optional[bool] = False,
        append_intraday: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
    ) -> Dict[datetime, ql.DiscountCurve]:
        assert (start_date and end_date) or bdates, "Must Pass in 'start_date' and 'end_date' or 'bdates'"

        if end_date:
            if end_date.date() == datetime_today_utc().date():
                append_intraday = True

        if not bdates:
            bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=ql_cal)

        async def build_tasks(
            client: httpx.AsyncClient,
            dates: List[datetime],
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for date in dates:
                task = asyncio.create_task(
                    self._fetch_and_read_eris_ftp_file(semaphore=semaphore, client=client, date=date, workbook_type="EOD_DiscountFactors_SOFR")
                )
                tasks.append(task)

            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING ERIS HISTORICAL DISC CURVES...")
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            dates: List[datetime],
        ):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits, verify=False, http2=True) as client:
                all_data = await build_tasks(
                    client=client,
                    dates=dates,
                )
                return all_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
                run_fetch_all(
                    dates=bdates,
                )
            )
            if results is None or len(results) == 0:
                return {}

            dict_df: Dict[datetime, pd.DataFrame] = dict(results)
            dict_ql_discount_curves: Dict[datetime, ql.DiscountCurve] = {}
            for dt, discount_curve_df in dict_df.items():
                if dt is None or discount_curve_df is None:
                    continue
                discount_curve_df["Date"] = pd.to_datetime(discount_curve_df["Date"], errors="coerce")
                discount_curve_df["DiscountFactor"] = pd.to_numeric(discount_curve_df["DiscountFactor"], errors="coerce")
                ql_curve = build_ql_discount_curve(
                    datetime_series=discount_curve_df["Date"],
                    discount_factor_series=discount_curve_df["DiscountFactor"],
                    ql_dc=ql_dc,
                    ql_cal=ql_cal,
                    interpolation_algo=f"df_{interpolation_algo}",
                )
                if enable_extrapolation:
                    ql_curve.enableExtrapolation()
                dict_ql_discount_curves[dt] = ql_curve

            if append_intraday:
                dict_ql_discount_curves[datetime_today_utc()] = self.fetch_intraday_discount_curve(
                    ql_dc=ql_dc, ql_cal=ql_cal, show_tqdm=show_tqdm, interpolation_algo=interpolation_algo
                )

            return dict_ql_discount_curves

    def fetch_intraday_discount_curve(
        self,
        return_df: Optional[bool] = False,
        ql_dc=ql.Actual360(),
        ql_cal=ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        show_tqdm: Optional[bool] = True,
        interpolation_algo: Optional[
            List[
                Literal[
                    "log_linear",
                    "mono_log_cubic",
                    "natural_cubic",
                    "kruger_log",
                    "natural_log_cubic",
                    "log_mixed_linear",
                    "log_parabolic_cubic",
                    "mono_log_parabolic_cubic",
                ]
            ]
        ] = "log_linear",
        enable_extrapolation: Optional[bool] = False,
        return_intraday_timestamp: Optional[bool] = False,
    ) -> ql.DiscountCurve | pd.DataFrame | Tuple[ql.DiscountCurve, datetime]:
        async def build_tasks(
            client: httpx.AsyncClient,
        ):
            semaphore = asyncio.Semaphore(1)
            tasks = [
                asyncio.create_task(
                    self._fetch_and_read_eris_ftp_file(semaphore=semaphore, client=client, date=None, workbook_type="Eris_Intraday_DiscountFactors_SOFR")
                )
            ]
            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING ERIS INTRADAY DISC CURVE...")
            return await asyncio.gather(*tasks)

        async def run_fetch_all():
            limits = httpx.Limits(
                max_connections=1,
                max_keepalive_connections=1,
            )
            async with httpx.AsyncClient(limits=limits, verify=False, http2=True) as client:
                all_data = await build_tasks(client=client)
                return all_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            results: List[Tuple[str, pd.DataFrame]] = asyncio.run(run_fetch_all())
            if results is None or len(results) == 0:
                return {}

            discount_curve_df = dict(results)["Eris_Intraday_DiscountFactors_SOFR.csv"]
            discount_curve_df["Date"] = pd.to_datetime(discount_curve_df["Date"], errors="coerce")
            discount_curve_df["DiscountFactor"] = pd.to_numeric(discount_curve_df["DiscountFactor"], errors="coerce")
            if return_df:
                return discount_curve_df

            ql_discount_curve = build_ql_discount_curve(
                datetime_series=discount_curve_df["Date"],
                discount_factor_series=discount_curve_df["DiscountFactor"],
                ql_dc=ql_dc,
                ql_cal=ql_cal,
                interpolation_algo=f"df_{interpolation_algo}",
            )
            if enable_extrapolation:
                ql_discount_curve.enableExtrapolation()

            if return_intraday_timestamp:
                intraday_ts = datetime.fromisoformat(
                    str(parser.parse(discount_curve_df["Time"].iloc[0], tzinfos={"EDT": tz.gettz("US/Eastern"), "EST": tz.gettz("US/Eastern")}))
                )
                intraday_ts = intraday_ts.astimezone(pytz.utc)
                return ql_discount_curve, intraday_ts

            return ql_discount_curve
