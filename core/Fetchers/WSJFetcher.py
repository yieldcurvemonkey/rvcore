import asyncio
import warnings
from datetime import datetime, timezone
from functools import reduce
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import tqdm
import ujson as json
from requests.models import PreparedRequest

from core.Fetchers.BaseFetcher import BaseFetcher

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class WSJFetcher(BaseFetcher):
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

    async def _fetch_timeseries(
        self,
        client: httpx.AsyncClient,
        wsj_ticker_key: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_retries: Optional[int] = 5,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
        intraday_timestamp: Optional[bool] = False,
        append_most_recent_last: Optional[bool] = False,
    ):
        payload = {
            "Step": "P1D",
            "TimeFrame": "all",
            "EntitlementToken": "57494d5ed7ad44af85bc59a51dd87c90",
            "IncludeMockTick": True,
            "FilterNullSlots": True,
            "FilterClosedPoints": True,
            "IncludeClosedSlots": True,
            "IncludeOfficialClose": True,
            "InjectOpen": True,
            "ShowPreMarket": True,
            "ShowAfterHours": True,
            "UseExtendedTimeFrame": True,
            "WantPriorClose": True,
            "IncludeCurrentQuotes": True,
            "ResetTodaysAfterHoursPercentChange": False,
            "Series": [
                {
                    "Key": wsj_ticker_key,
                    "Dialect": "Charting",
                    "Kind": "Ticker",
                    "SeriesId": "s1",
                    "DataTypes": ["Last"],
                }
            ],
        }
        params = {
            "json": json.dumps(payload),
            "ckey": "57494d5ed7",
        }
        url = "https://api.wsj.net/api/michelangelo/timeseries/history"
        prep_url = PreparedRequest()
        prep_url.prepare_url(url, params)
        headers = {
            "authority": "api.wsj.net",
            "method": "GET",
            "path": prep_url.path_url,
            "scheme": "https",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Dylan2010.EntitlementToken": "57494d5ed7ad44af85bc59a51dd87c90",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36",
            "sec-ch-ua-platform": '"macOS"',
            "Origin": "https://www.wsj.com",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Referer": "https://www.wsj.com/",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        }

        cols_to_return = ["Date", wsj_ticker_key]
        retries = 0
        try:
            while retries < max_retries:
                try:
                    response = await client.get(prep_url.url, headers=headers)
                    response.raise_for_status()
                    json_data = response.json()
                    if intraday_timestamp:
                        df = pd.DataFrame(
                            {"Date": json_data["Series"][0]["CurrentQuote"]["DateUtc"], wsj_ticker_key: [d[0] for d in json_data["Series"][0]["DataPoints"]]}
                        )
                    else:
                        df = pd.DataFrame({"Date": json_data["TimeInfo"]["Ticks"], wsj_ticker_key: [d[0] for d in json_data["Series"][0]["DataPoints"]]})

                    if append_most_recent_last:
                        intraday_df = pd.DataFrame(
                            {"Date": json_data["Series"][0]["CurrentQuote"]["DateUtc"], wsj_ticker_key: [d[0] for d in json_data["Series"][0]["DataPoints"]]}
                        ).tail(1)

                        df = pd.concat([df.head(-1), intraday_df])

                    df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
                    df = df.drop_duplicates(subset=["Date"]).sort_values(by="Date")
                    if start_date:
                        df = df[df["Date"].dt.date >= start_date.date()]
                    if end_date:
                        df = df[df["Date"].dt.date <= end_date.date()]
                    
                    if uid:
                        return wsj_ticker_key, df, uid
                    return wsj_ticker_key, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"WSJ - Bad Status: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return wsj_ticker_key, pd.DataFrame(columns=cols_to_return), uid
                        return wsj_ticker_key, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"WSJ- Throttled for {wsj_ticker_key}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"WSJ - Error: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"WSJ - Throttled for {wsj_ticker_key}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"WSJ  - Max retries exceeded for {wsj_ticker_key}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return wsj_ticker_key, pd.DataFrame(columns=cols_to_return), uid
            return wsj_ticker_key, pd.DataFrame(columns=cols_to_return)

    async def _fetch_timeseries_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_timeseries(*args, **kwargs)

    def wsj_timeseries_api(
        self,
        wsj_ticker_keys: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        intraday_timestamp: Optional[bool] = False,
        append_most_recent_last: Optional[bool] = False,
        one_df: Optional[bool] = False,
        show_tqdm: Optional[bool] = False,
        max_concurrent_tasks: int = 64,
    ):
        async def build_tasks(
            client: httpx.AsyncClient,
            wsj_ticker_keys: List[str],
            start_date: datetime,
            end_date: datetime,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_timeseries_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    wsj_ticker_key=wsj_ticker_key,
                    start_date=start_date,
                    end_date=end_date,
                    max_retries=3,
                    intraday_timestamp=intraday_timestamp,
                    append_most_recent_last=append_most_recent_last,
                )
                for wsj_ticker_key in wsj_ticker_keys
            ]

            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING DATA FROM WSJ...")
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            wsj_ticker_keys: List[str],
            start_date: datetime,
            end_date: datetime,
        ):
            async with httpx.AsyncClient(timeout=self._global_timeout, mounts=self._httpx_proxies, verify=False, http2=True) as client:
                all_data = await build_tasks(
                    client=client,
                    wsj_ticker_keys=wsj_ticker_keys,
                    start_date=start_date,
                    end_date=end_date,
                )
                return all_data

        dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(
                wsj_ticker_keys=wsj_ticker_keys,
                start_date=start_date,
                end_date=end_date,
            )
        )

        if one_df:
            dict_df = {}
            for symbol, df in dfs:
                if df is not None:
                    dict_df[symbol] = df

            merge_dfs_on_column = lambda dfs_dict, on_column: reduce(lambda left, right: pd.merge(left, right, on=on_column, how="outer"), dfs_dict.values())
            return merge_dfs_on_column(dict_df, "Date").reset_index(drop=True).set_index("Date")

        return dict(dfs)
