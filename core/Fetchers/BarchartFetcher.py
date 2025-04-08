import asyncio
import warnings
from datetime import datetime
from functools import reduce
from io import StringIO
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import quote, unquote

import httpx
import pandas as pd
import requests
import tqdm
import tqdm.asyncio

from backend.Fetchers.BaseFetcher import BaseFetcher

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class BarchartFetcher(BaseFetcher):
    _current_laravel_token: str = None
    _current_xsrf_token: str = None
    _BARCHART_MAX_RECORD = 1_000_000

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

    def _fetch_session_tokens(self, dummy_symbol: Optional[str] = "BTC"):
        interactive_chart_url = f"https://www.barchart.com/futures/quotes/{quote(dummy_symbol)}/interactive-chart"
        interactive_chart_headers = {
            "dnt": "1",
            "referer": f"https://www.barchart.com/futures/quotes/{quote(dummy_symbol)}/interactive-chart",
            "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        }
        interactive_chart_res = requests.get(interactive_chart_url, headers=interactive_chart_headers)
        interactive_chart_res.raise_for_status()

        cookie_pairs = unquote(interactive_chart_res.headers["Set-Cookie"]).split("; ")
        cleaned_cookie_pairs = []
        for pair in cookie_pairs:
            if ", " in pair:
                newpair1, newpair2 = pair.split(", ")
                cleaned_cookie_pairs.append(newpair1)
                cleaned_cookie_pairs.append(newpair2)
            else:
                cleaned_cookie_pairs.append(pair)

        cookie_dict = {}
        for pair in cleaned_cookie_pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                cookie_dict[key] = value

        if "laravel_token" in cookie_dict and "XSRF-TOKEN" in cookie_dict:
            self._current_laravel_token = cookie_dict["laravel_token"]
            self._current_xsrf_token = cookie_dict["XSRF-TOKEN"]
        else:
            raise ValueError(f"Barchart Session Tokne Cookie Parsing Error: could not find laravel_token and XSRF-TOKEN")

    def _parse_aspx_response_to_df(self, response_content: bytes, columns: Optional[List[str]] = None):
        if isinstance(response_content, bytes):
            response_content = response_content.decode("utf-8")

        data_stream = StringIO(response_content)
        df = pd.read_csv(data_stream, header=None)
        if columns and len(columns) == len(df.columns):
            df.columns = columns
        if columns and len(columns) - 1 == len(df.columns):
            df.columns = columns[:-1]

        return df

    async def _fetch_intraday_timeseries(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        interval: Literal[1, 5, 10, 15, 30, 60, 120, 240],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = ["Date", "temp", "Open", "High", "Low", "Close", "Volume"],
        set_dt_index: Optional[bool] = True,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ) -> Tuple[str, pd.DataFrame] | Tuple[str, pd.DataFrame, str]:
        if start_date is not None:
            if start_date.tzinfo is None or start_date.tzinfo.utcoffset(start_date) is None:
                raise ValueError("start_date must include timezone information.")

        if end_date is not None:
            if end_date.tzinfo is None or end_date.tzinfo.utcoffset(end_date) is None:
                raise ValueError("end_date must include timezone information.")

        if start_date is not None and end_date is not None:
            if start_date.tzinfo != end_date.tzinfo:
                raise ValueError("start_date and end_date must have the same timezone.")

        try:
            url = f"https://www.barchart.com/proxies/timeseries/historical/queryminutes.ashx?symbol={quote(symbol)}&interval={interval}&maxrecords={self._BARCHART_MAX_RECORD}&volume=contract&order=asc"
            headers = {
                "dnt": "1",
                "referer": f"https://www.barchart.com/futures/quotes/{quote(symbol)}/interactive-chart",
                "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "cookie": f"laravel_token={self._current_laravel_token}",
                "x-xsrf-token": self._current_xsrf_token,
            }

            retries = 0
            while retries < max_retries:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    df = self._parse_aspx_response_to_df(response.content, columns=columns)

                    if "temp" in df.columns:
                        df = df.drop(columns=["temp"])

                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                        if start_date:
                            if df["Date"].dt.tz is None:
                                df["Date"] = df["Date"].dt.tz_localize(start_date.tzinfo)
                            else:
                                df["Date"] = df["Date"].dt.tz_convert(start_date.tzinfo)
                            df = df[df["Date"] >= start_date]

                        if end_date:
                            if df["Date"].dt.tz is None:
                                df["Date"] = df["Date"].dt.tz_localize(end_date.tzinfo)
                            else:
                                df["Date"] = df["Date"].dt.tz_convert(end_date.tzinfo)
                            df = df[df["Date"] <= start_date]

                        if set_dt_index:
                            df = df.set_index("Date")

                    if uid:
                        return symbol, df, uid
                    return symbol, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"Barchart Intraday - Bad Status for {symbol}: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return symbol, None, uid
                        return symbol, None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Barchart Intraday - Throttled for {symbol}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Barchart Intraday - Error for {symbol}: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Barchart Intraday - Throttled for {symbol}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Barchart Intraday - Max retries exceeded for {symbol}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return symbol, None, uid
            return symbol, None

    async def _fetch_intraday_timeseries_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_intraday_timeseries(*args, **kwargs)

    async def _fetch_eod_timeseries(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Open Interest"],
        set_dt_index: Optional[bool] = True,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ) -> Tuple[str, pd.DataFrame] | Tuple[str, pd.DataFrame, str]:

        try:
            url = f"https://www.barchart.com/proxies/timeseries/historical/queryeod.ashx?symbol={quote(symbol)}&data=daily&maxrecords={self._BARCHART_MAX_RECORD}&volume=contract&order=asc"
            headers = {
                "dnt": "1",
                "referer": f"https://www.barchart.com/futures/quotes/{quote(symbol)}/interactive-chart",
                "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "cookie": f"laravel_token={self._current_laravel_token}",
                "x-xsrf-token": self._current_xsrf_token,
            }

            retries = 0
            while retries < max_retries:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    df = self._parse_aspx_response_to_df(response.content, columns=columns)

                    if "Symbol" in df.columns:
                        df = df.drop(columns=["Symbol"])

                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        if set_dt_index:
                            df = df.set_index("Date")
                        if start_date:
                            df = df[df["Date"] >= start_date] if not set_dt_index else df[df.index >= start_date]
                        if end_date:
                            df = df[df["Date"] <= end_date] if not set_dt_index else df[df.index <= end_date]

                    if uid:
                        return symbol, df, uid
                    return symbol, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"Barchart EOD - Bad Status for {symbol}: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return symbol, None, uid
                        return symbol, None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Barchart EOD - Throttled for {symbol}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Barchart EOD - Error for {symbol}: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"Barchart EOD - Throttled for {symbol}. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Barchart EOD - Max retries exceeded for {symbol}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return symbol, None, uid
            return symbol, None

    async def _fetch_eod_timeseries_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_eod_timeseries(*args, **kwargs)

    def barchart_timeseries_api(
        self,
        barchart_symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: Optional[Literal[1, 5, 10, 15, 30, 60, 120, 240]] = None,
        max_concurrent_tasks: int = 64,
        one_df: Optional[bool] = False,
        show_tqdm: Optional[bool] = False,
        merge_val_col: Optional[Literal["Open", "High", "Low", "Close", "Volume", "Open Interest"]] = "Close",
    ):
        async def build_eod_tasks(
            client: httpx.AsyncClient,
            barchart_symbols: List[str],
            start_date: datetime,
            end_date: datetime,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_eod_timeseries_with_semaphore(
                    semaphore=semaphore, client=client, symbol=symbol, start_date=start_date, end_date=end_date, set_dt_index=not one_df
                )
                for symbol in barchart_symbols
            ]

            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING INTRADAY DATA FROM BARCHART...")
            return await asyncio.gather(*tasks)

        async def build_intraday_tasks(
            client: httpx.AsyncClient,
            barchart_symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            interval: Optional[Literal[1, 5, 10, 15, 30, 60, 120, 240]] = None,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_intraday_timeseries_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    smybol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    set_dt_index=not one_df,
                )
                for symbol in barchart_symbols
            ]

            if show_tqdm:
                return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING EOD DATA FROM BARCHART...")
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            barchart_symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            interval: Optional[Literal[1, 5, 10, 15, 30, 60, 120, 240]] = None,
        ):
            async with httpx.AsyncClient(timeout=self._global_timeout, mounts=self._httpx_proxies, verify=False, http2=True) as client:
                if interval:
                    all_data = await build_intraday_tasks(
                        client=client, barchart_symbols=barchart_symbols, start_date=start_date, end_date=end_date, interval=interval
                    )
                else:
                    all_data = await build_eod_tasks(
                        client=client,
                        barchart_symbols=barchart_symbols,
                        start_date=start_date,
                        end_date=end_date,
                    )
                return all_data

        self._fetch_session_tokens()
        dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(barchart_symbols=barchart_symbols, start_date=start_date, end_date=end_date, interval=interval)
        )

        if one_df:

            def merge_dfs_on_column(dfs_dict: Dict[str, pd.DataFrame], on_column: str, merge_val_col: str):
                dfs_dict = {key: df[[on_column, merge_val_col]].rename(columns={merge_val_col: key}) for key, df in dfs_dict.items() if df is not None}
                merged_df = reduce(lambda left, right: pd.merge(left, right, on=on_column, how="outer"), dfs_dict.values())
                merged_df.sort_values(by=on_column, inplace=True)
                return merged_df

            merged = merge_dfs_on_column(dict(dfs), "Date", merge_val_col)
            return merged.set_index("Date")

        return dict(dfs)


def cme_curve_to_barchart_stirs_futures_symbol(
    curve: Literal[
        "USD-SOFR-1D",
        "USD-FEDFUNDS",
        "JPY-TONAR",
        "CAD-CORRA",
        "EUR-ESTR",
        "EUR-EURIBOR-1M",
        "EUR-EURIBOR-3M",
        "EUR-EURIBOR-6M",
        "GBP-SONIA",
        "CHF-SARON-1D",
        "NOK-NIBOR-6M",
        "HKD-HIBOR-3M",
        "AUD-AONIA",
        "SGD-SORA-1D",
    ],
):
    cme_curve_barchart_symbol_map = {
        "USD-SOFR-1D": "SQ",
        "USD-FEDFUNDS": "ZQ",
        "JPY-TONAR": "T0",
        "CAD-CORRA": "RG",
        "EUR-ESTR": "RA",
        "EUR-EURIBOR-1M": None,
        "EUR-EURIBOR-3M": "IM",
        "EUR-EURIBOR-6M": None,
        "GBP-SONIA": "J8",
        "CHF-SARON-1D": None,
        "NOK-NIBOR-6M": None,
        "HKD-HIBOR-3M": None,
        "AUD-AONIA": None,
        "SGD-SORA-1D": None,
    }

    return cme_curve_barchart_symbol_map[curve]
