import asyncio
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import httpx
import pandas as pd
import pyarrow.csv as pacsv
import pyzipper
import requests
import tqdm
import tqdm.asyncio
import ujson
import numpy as np

from pandas.errors import DtypeWarning
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from core.Fetchers.BaseFetcher import BaseFetcher

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def datetime_today_utc():
    return datetime(
        year=datetime.now(timezone.utc).year,
        month=datetime.now(timezone.utc).month,
        day=datetime.now(timezone.utc).day,
    )


class DTCCFetcher(BaseFetcher):
    pddata_dtcc_base_url = "https://pddata.dtcc.com/ppd"

    def __init__(
        self,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )

    def _get_dtcc_url_and_header(
        self,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        date_string: str,
    ) -> Tuple[str, Dict[str, str]]:
        if agency == "SEC" and asset_class in ["COMMODITIES", "FOREX"]:
            raise ValueError(f"SEC does not store {asset_class} in their SDR data.")

        is_intraday_report = len(date_string.split("_")) == 4
        if is_intraday_report:
            # e.g. .../intraday/cftc/CFTC_SLICE_RATES_2023_01_15_10_420.zip
            dtcc_url = f"{self.pddata_dtcc_base_url}/api/report/intraday/{agency.lower()}/{agency}_SLICE_{asset_class}_{date_string}.zip"
        else:
            # e.g. .../cumulative/cftc/CFTC_CUMULATIVE_RATES_2023_01_15.zip
            dtcc_url = f"{self.pddata_dtcc_base_url}/api/report/cumulative/{agency.lower()}/{agency}_CUMULATIVE_{asset_class}_{date_string}.zip"

        dtcc_headers = {
            "authority": "pddata.dtcc.com",
            "method": "GET",
            "path": dtcc_url.split(".com")[1],
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "dnt": "1",
            "priority": "u=1, i",
            "referer": f"{self.pddata_dtcc_base_url}/{agency.lower()}dashboard",
            "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/130.0.0.0 Safari/537.36"),
        }
        return dtcc_url, dtcc_headers

    async def _fetch_dtcc_sdr_data_helper(
        self,
        client: httpx.AsyncClient,
        date_string: str,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        max_retries: Optional[int] = 5,
        backoff_factor: Optional[int] = 1,
    ) -> Optional[BytesIO]:
        dtcc_sdr_url, dtcc_sdr_header = self._get_dtcc_url_and_header(date_string=date_string, agency=agency, asset_class=asset_class)

        retries = 0
        while retries < max_retries:
            try:
                async with client.stream("GET", dtcc_sdr_url, headers=dtcc_sdr_header, timeout=self._global_timeout) as response:
                    response.raise_for_status()
                    zip_buffer = BytesIO()
                    async for chunk in response.aiter_bytes():
                        zip_buffer.write(chunk)
                    zip_buffer.seek(0)
                return zip_buffer

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    self._logger.debug(f"DTCCFetcher - 404 for {agency}-{asset_class}-{date_string}. Skipping.")
                    return None

                # attempt exponential backoff
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                self._logger.debug(f"DTCCFetcher - HTTP Error {e.response.status_code} for {agency}-{asset_class}-{date_string}. " f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

            except (httpx.RequestError, Exception) as e:
                retries += 1
                wait_time = backoff_factor * (2 ** (retries - 1))
                self._logger.debug(f"DTCCFetcher - Connection/Other Error {e} for {agency}-{asset_class}-{date_string}. " f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        self._logger.error(f"DTCCFetcher - Max retries exceeded for {agency}-{asset_class}-{date_string}.")
        return None

    @staticmethod
    def _parse_filename_to_datetime(filename: str) -> datetime:
        """
        Extract date from the file name by patterning on '_YYYY_MM_DD'.

        :param filename: e.g. "CFTC_IR_2023_01_15"
        :return: datetime(2023, 1, 15)
        """
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})$", filename)
        if not match:
            raise ValueError(f"DTCCFetcher - Filename does not contain a valid date: {filename}")
        year, month, day = map(int, match.groups())
        return datetime(year, month, day)

    @staticmethod
    def _read_single_file(
        file_buffer: bytes,
        file_name: str,
        convert_key_into_dt: bool,
        use_pyarrow: Optional[bool] = False,
    ) -> Tuple[Optional[Union[str, datetime]], Optional[pd.DataFrame]]:
        file_name_lower = file_name.lower()
        extension = None
        if file_name_lower.endswith((".xls", ".xlsx")):
            extension = "excel"
        elif file_name_lower.endswith(".csv"):
            extension = "csv"
        else:
            # Not a recognized extension
            return None, None

        buffer_io = BytesIO(file_buffer)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            if extension == "excel":
                df = pd.read_excel(buffer_io)
            else:  # csv
                if use_pyarrow:
                    try:
                        table = pacsv.read_csv(buffer_io)
                        df = table.to_pandas()
                    except ImportError:
                        df = pd.read_csv(buffer_io, low_memory=False)
                else:
                    df = pd.read_csv(buffer_io, low_memory=False)

        key = file_name
        if convert_key_into_dt:
            try:
                key = DTCCFetcher._parse_filename_to_datetime(file_name.split(".")[0])
            except ValueError:
                # fallback: keep as string if parse fails
                pass

        return key, df

    def _extract_dataframes_from_zip(
        self,
        zip_buffer: BytesIO,
        convert_key_into_dt: Optional[bool] = False,
        parallelize: Optional[bool] = False,
        max_extraction_workers: Optional[int] = 3,
        use_pyarrow: Optional[bool] = False,
    ) -> Dict[Union[str, datetime], pd.DataFrame]:
        if not zip_buffer:
            return {}

        dataframes: Dict[Union[str, datetime], pd.DataFrame] = {}
        with pyzipper.AESZipFile(zip_buffer) as zip_file:
            allowed_extensions = (".xlsx", ".xls", ".csv")
            candidates = [info for info in zip_file.infolist() if not info.is_dir() and info.filename.lower().endswith(allowed_extensions)]
            if not candidates:
                return {}

            def process_single_entry(info):
                file_name = info.filename
                file_content = zip_file.read(file_name)
                key, df = self._read_single_file(
                    file_buffer=file_content,
                    file_name=file_name,
                    convert_key_into_dt=convert_key_into_dt,
                    use_pyarrow=use_pyarrow,
                )
                return key, df

            if parallelize and len(candidates) > 1:
                with ThreadPoolExecutor(max_workers=max_extraction_workers) as executor:
                    results = list(executor.map(process_single_entry, candidates))
            else:
                results = [process_single_entry(info) for info in candidates]

            for key, df in results:
                if key is not None and df is not None:
                    dataframes[key] = df
        return dataframes

    async def _fetch_zip_and_extract(
        self,
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        date_string: str,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        parallelize: bool,
        max_extraction_workers: int,
        convert_key_into_dt: bool,
        use_pyarrow: bool,
        task_id: Optional[Any] = None,
    ) -> Dict[Union[str, datetime], pd.DataFrame]:
        async with semaphore:
            zip_buffer = await self._fetch_dtcc_sdr_data_helper(
                client=client,
                date_string=date_string,
                agency=agency,
                asset_class=asset_class,
            )

        # Use run_in_executor for CPU-bound extraction
        if zip_buffer is None:
            return {}

        loop = asyncio.get_event_loop()
        partial_func = partial(
            self._extract_dataframes_from_zip,
            zip_buffer=zip_buffer,
            convert_key_into_dt=convert_key_into_dt,
            parallelize=parallelize,
            max_extraction_workers=max_extraction_workers,
            use_pyarrow=use_pyarrow,
        )
        dataframes = await loop.run_in_executor(None, partial_func)
        if task_id:
            return dataframes, task_id
        return dataframes

    async def _run_fetch_tasks(
        self,
        date_strings: List[str],
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        parallelize: bool,
        max_extraction_workers: int,
        max_concurrent_tasks: int,
        client: httpx.AsyncClient,
        convert_key_into_dt: bool,
        use_pyarrow: bool,
        tqdm_desc: Optional[str] = "FETCHING DTCC SDR DATASETS...",
    ) -> List[Dict[Union[str, datetime], pd.DataFrame]]:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = []
        for ds in date_strings:
            task = asyncio.create_task(
                self._fetch_zip_and_extract(
                    semaphore=semaphore,
                    client=client,
                    date_string=ds,
                    agency=agency,
                    asset_class=asset_class,
                    parallelize=parallelize,
                    max_extraction_workers=max_extraction_workers,
                    convert_key_into_dt=convert_key_into_dt,
                    use_pyarrow=use_pyarrow,
                )
            )
            tasks.append(task)

        if tqdm_desc:
            results = await tqdm.asyncio.tqdm.gather(*tasks, desc=tqdm_desc)
        else:
            results = await asyncio.gather(*tasks)
        return results

    def fetch_historical_reports(
        self,
        start_date: datetime,
        end_date: datetime,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        parallelize: Optional[bool] = False,
        max_extraction_workers: Optional[int] = 3,
        use_pyarrow: Optional[bool] = False,
        one_df: Optional[bool] = False,
        show_tqdm: Optional[bool] = True,
    ) -> Dict[datetime, pd.DataFrame] | pd.DataFrame:
        bdates = pd.date_range(
            start=start_date.astimezone(timezone.utc),
            end=end_date.astimezone(timezone.utc),
            freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
        )
        date_strings = [d.strftime("%Y_%m_%d") for d in bdates]

        async def run():
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits, timeout=self._global_timeout, mounts=self._httpx_proxies, verify=False, http2=True) as client:
                results = await self._run_fetch_tasks(
                    date_strings=date_strings,
                    agency=agency,
                    asset_class=asset_class,
                    parallelize=parallelize,
                    max_extraction_workers=max_extraction_workers,
                    max_concurrent_tasks=max_concurrent_tasks,
                    client=client,
                    convert_key_into_dt=True,
                    use_pyarrow=use_pyarrow,
                    tqdm_desc="FETCHING HISTORICAL SDR REPORTS..." if show_tqdm else None,
                )
                return results

        all_results = asyncio.run(run())
        merged_data: Dict[datetime, pd.DataFrame] = {}
        for daily_dict in all_results:
            for k, v_df in daily_dict.items():
                if isinstance(k, datetime) and isinstance(v_df, pd.DataFrame):
                    v_df["Event timestamp"] = pd.to_datetime(v_df["Event timestamp"], errors="coerce", utc=True)
                    v_df["Execution Timestamp"] = pd.to_datetime(v_df["Execution Timestamp"], errors="coerce", utc=True)
                    v_df["Effective Date"] = pd.to_datetime(v_df["Effective Date"], errors="coerce")
                    v_df["Expiration Date"] = pd.to_datetime(v_df["Expiration Date"], errors="coerce")
                    v_df = v_df.sort_values(by="Event timestamp")
                    merged_data[k] = v_df

        if len(merged_data.keys()) == 0:
            return pd.DataFrame([])

        if one_df:
            return pd.concat(merged_data.values())
        else:
            return merged_data

    def fetch_intraday_reports(
        self,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        parallelize: Optional[bool] = False,
        max_extraction_workers: Optional[int] = 3,
        use_pyarrow: Optional[bool] = False,
        show_tqdm: Optional[bool] = True,
    ) -> pd.DataFrame:
        slice_ids = self._get_dtcc_intraday_slide_ids(agency=agency, asset_class=asset_class, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

        async def run_intra_slices():
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits, timeout=self._global_timeout, mounts=self._httpx_proxies, verify=False, http2=True) as client:
                results = await self._run_fetch_tasks(
                    date_strings=slice_ids,
                    agency=agency,
                    asset_class=asset_class,
                    parallelize=parallelize,
                    max_extraction_workers=max_extraction_workers,
                    max_concurrent_tasks=max_concurrent_tasks,
                    client=client,
                    convert_key_into_dt=False,
                    use_pyarrow=use_pyarrow,
                    tqdm_desc="FETCHING INTRADAY SDR SLICES..." if show_tqdm else None,
                )
                return results

        all_results = asyncio.run(run_intra_slices())
        combined_results: Dict[str, pd.DataFrame] = {}
        for res_dict in all_results:
            combined_results.update(res_dict)  # merges each slice's data

        if not combined_results:
            return pd.DataFrame()

        list_of_dfs = []
        for slice_id_str, df in combined_results.items():
            df = df.copy()
            df["report_slice"] = slice_id_str
            list_of_dfs.append(df)

        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        combined_df["Event timestamp"] = pd.to_datetime(combined_df["Event timestamp"], errors="coerce", utc=True)
        combined_df["Execution Timestamp"] = pd.to_datetime(combined_df["Execution Timestamp"], errors="coerce", utc=True)
        combined_df["Effective Date"] = pd.to_datetime(combined_df["Effective Date"], errors="coerce")
        combined_df["Expiration Date"] = pd.to_datetime(combined_df["Expiration Date"], errors="coerce")
        combined_df = combined_df.sort_values(by="Event timestamp")

        return combined_df

    def fetch_reports(
        self,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        start_date: datetime,
        end_date: datetime,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        parallelize: Optional[bool] = False,
        max_extraction_workers: Optional[int] = 3,
        use_pyarrow: Optional[bool] = False,
        show_tqdm: Optional[bool] = True,
    ) -> pd.DataFrame:
        append_intraday = False
        if end_date.astimezone(timezone.utc).date() == datetime_today_utc().date() and end_date.weekday() < 5:
            append_intraday = True

        historical_sdr_df = self.fetch_historical_reports(
            agency=agency,
            asset_class=asset_class,
            start_date=start_date.astimezone(timezone.utc),
            end_date=end_date.astimezone(timezone.utc),
            max_concurrent_tasks=max_concurrent_tasks,
            max_keepalive_connections=max_keepalive_connections,
            parallelize=parallelize,
            max_extraction_workers=max_extraction_workers,
            use_pyarrow=use_pyarrow,
            one_df=True,
            show_tqdm=show_tqdm,
        )
        if append_intraday:
            intraday_sdr_df = self.fetch_intraday_reports(
                agency=agency,
                asset_class=asset_class,
                start_timestamp=start_date.astimezone(timezone.utc) if start_date.tzinfo is not None else None,
                end_timestamp=end_date.astimezone(timezone.utc) if end_date.tzinfo is not None else None,
                max_concurrent_tasks=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
                parallelize=parallelize,
                max_extraction_workers=max_extraction_workers,
                use_pyarrow=use_pyarrow,
                show_tqdm=show_tqdm,
            )
            sdr_df = pd.concat([historical_sdr_df, intraday_sdr_df])
        else:
            sdr_df = historical_sdr_df

        if sdr_df.empty:
            return pd.DataFrame([])

        sdr_df.replace(["", " ", None, "None", "NaN"], np.nan, inplace=True)
        return sdr_df.sort_values(by="Event timestamp").reset_index(drop=True)

    def _get_dtcc_intraday_slide_ids(
        self,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
    ):
        hist_intr_asset_class_id_mapper = {
            "COMMODITIES": "CO",
            "CREDITS": "CR",
            "EQUITIES": "EQ",
            "FOREX": "FX",
            "RATES": "IR",
        }
        if asset_class not in hist_intr_asset_class_id_mapper:
            raise ValueError(f"Unsupported asset class: {asset_class}")

        intraday_ids_url = f"{self.pddata_dtcc_base_url}/api/slice/{agency}/{hist_intr_asset_class_id_mapper[asset_class]}"
        intraday_report_ids_headers = {
            "authority": "pddata.dtcc.com",
            "method": "GET",
            "path": f"/ppd/api/slice/{agency}/{hist_intr_asset_class_id_mapper[asset_class]}",
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "dnt": "1",
            "priority": "u=1, i",
            "referer": f"{self.pddata_dtcc_base_url}/{agency.lower()}dashboard",
            "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
        }

        intraday_report_ids_res = requests.get(
            intraday_ids_url,
            headers=intraday_report_ids_headers,
            proxies=self._proxies,
        )
        intraday_report_ids_res.raise_for_status()

        intraday_report_ids = ujson.loads(intraday_report_ids_res.content.decode("utf-8"))
        intraday_report_ids_df = pd.DataFrame(intraday_report_ids)
        intraday_report_ids_df["dissemDTM"] = pd.to_datetime(intraday_report_ids_df["dissemDTM"], errors="coerce", utc=True)
        if start_timestamp:
            start_timestamp = pd.to_datetime(start_timestamp, utc=True)
            intraday_report_ids_df = intraday_report_ids_df[intraday_report_ids_df["dissemDTM"] >= start_timestamp]
        if end_timestamp:
            end_timestamp = pd.to_datetime(end_timestamp, utc=True)
            intraday_report_ids_df = intraday_report_ids_df[intraday_report_ids_df["dissemDTM"] <= end_timestamp]

        return [str(row["fileName"]).split(f"_{asset_class}_")[1].split(".")[0] for _, row in intraday_report_ids_df.iterrows()]
