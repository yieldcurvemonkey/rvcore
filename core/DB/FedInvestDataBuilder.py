import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz
import httpx
import numpy as np
import pandas as pd
import polars as pl
import QuantLib as ql
import tqdm
import tqdm.asyncio
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, "../")
from Fetchers.FedInvest import FedInvestDataFetcher
from Fetchers.NYFRBFetcher import NYFRBDataFetcher
from Fetchers.TreasuryDirectFetcher import TreasuryDirectFetcher
from Fetchers.WSJFetcher import WSJFetcher
from utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime
from utils.ust_utils import fedinvest_ust_back_out_price, fedinvest_ust_pricer, get_isin_from_cusip, historical_auction_cols, is_valid_ust_cusip, ust_sorter

UST_PRICED_COLS = [
    "cusip",
    "type",
    "coupon",
    "offer_price",
    "bid_price",
    "eod_price",
    "mid_price",
    "issue_date",
    "maturity_date",
    "settle_date",
    "eod_ytm",
    "bid_ytm",
    "offer_ytm",
    "mid_ytm",
]

UST_DB_COLS = historical_auction_cols() + [
    "ust_label",
    "cme_ust_label",
    "settle_date",
    "time_to_maturity",
    "coupon",
    "bid_price",
    "offer_price",
    "mid_price",
    "eod_price",
    "bid_ytm",
    "offer_ytm",
    "mid_ytm",
    "eod_ytm",
    "soma_holdings_as_of_date",
    "soma_holdings_par_value",
    "soma_holdings_percent_outstanding",
    "soma_holdings_est_outstanding_amt",
    "outstanding_amt",
    "portion_unstripped_amt",
    "portion_stripped_amt",
    "reconstituted_amt",
    "rank",
    "free_float",
    "timestamp",
    "hash",
]


class FedInvestDataBuilder:
    _treasurydirect_fetcher: TreasuryDirectFetcher = None
    _fedinvest_fetcher: FedInvestDataFetcher = None
    _nyfrb_fetcher: NYFRBDataFetcher = None
    _wsj_fetcher: WSJFetcher = None

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _warning_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _proxies: Dict[str, str] = (None,)
    _no_logs_plz: bool = False

    def __init__(
        self,
        debug_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
        proxies: Optional[Dict[str, str]] = None,
    ):
        self._treasurydirect_fetcher = TreasuryDirectFetcher(debug_verbose=debug_verbose, error_verbose=error_verbose, info_verbose=info_verbose, proxies=proxies)
        self._fedinvest_fetcher = FedInvestDataFetcher(debug_verbose=debug_verbose, error_verbose=error_verbose, info_verbose=info_verbose, proxies=proxies)
        self._nyfrb_fetcher = NYFRBDataFetcher(debug_verbose=debug_verbose, error_verbose=error_verbose, info_verbose=info_verbose, proxies=proxies)
        self._wsj_fetcher = WSJFetcher(debug_verbose=debug_verbose, error_verbose=error_verbose, info_verbose=info_verbose, proxies=proxies)

        self._debug_verbose = debug_verbose
        self._warning_verbose = warning_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose
        self._setup_logger()

    def _setup_logger(self):
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.WARNING)

        if self._debug_verbose or self._info_verbose or self._error_verbose:
            self._logger.setLevel(logging.DEBUG)

        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

    def get_ust_cusip_sets_prices(
        self, start_date: datetime, end_date: datetime, n_jobs: Optional[int] = 1, return_filtered_cols: Optional[bool] = False, show_tqdm: Optional[bool] = False
    ) -> Dict[datetime, pd.DataFrame]:
        if end_date.date() == datetime.today().date() and start_date.date() == datetime.today().date():
            start_date = ql_date_to_datetime(ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(datetime_to_ql_date(datetime.today()), ql.Period("-1D")))

        bdates = [
            bday.to_pydatetime()
            for bday in pd.bdate_range(
                start=start_date,
                end=end_date,
                freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
            )
        ]
        fedinvest_hist_prices_dict_df: Dict[datetime, pd.DataFrame] = self._fedinvest_fetcher.runner(dates=bdates, show_tqdm=show_tqdm)
        cusips_info_df = self._treasurydirect_fetcher._historical_auctions_df.drop_duplicates(subset=["cusip"], keep="first")
        fedinvest_hist_prices_iter = tqdm.tqdm(fedinvest_hist_prices_dict_df.items(), desc="PRICING USTs...") if show_tqdm else fedinvest_hist_prices_dict_df.items()

        def price_for_date(dt: datetime, fedinvest_hist_prices_df: pd.DataFrame):
            curr_cusips_info_df = cusips_info_df[cusips_info_df["cusip"].isin(fedinvest_hist_prices_df["cusip"])]
            merged_df = pd.merge(left=fedinvest_hist_prices_df, right=curr_cusips_info_df, on="cusip")
            priced_df = fedinvest_ust_pricer(merged_df, as_of_date=dt)
            if return_filtered_cols:
                priced_df = priced_df[UST_PRICED_COLS]

            ny = pytz.timezone("US/Eastern")
            naive_close = datetime(dt.year, dt.month, dt.day, hour=17, minute=0, second=0)
            ny_close = ny.localize(naive_close)
            curr_ts = ny_close.astimezone(pytz.UTC)

            priced_df["timestamp"] = curr_ts
            priced_df["hash"] = priced_df["cusip"].apply(lambda c: f"{c}_{curr_ts}")
            return dt, priced_df

        results = Parallel(n_jobs=n_jobs)(delayed(price_for_date)(dt, fedinvest_hist_prices_df) for dt, fedinvest_hist_prices_df in fedinvest_hist_prices_iter)
        cusip_set_dict_df: Dict[datetime, pd.DataFrame] = {dt: priced_df for dt, priced_df in results}

        if end_date.date() == datetime.today().date():
            yday = ql_date_to_datetime(ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(datetime_to_ql_date(datetime.today()), ql.Period("-1D")))
            yday_cusip_set_df = cusip_set_dict_df[yday].copy()

            cusip_icap_ticker_map = dict(
                zip(
                    yday_cusip_set_df["cusip"]
                    .apply(
                        lambda row: f"BOND/US/XTUP/{get_isin_from_cusip(row)[2:]}" if get_isin_from_cusip(row)[2:] != "912810SX72" else "BOND/UK/XTUP/912810SX72"
                    )  # bug from wsj
                    .to_list(),
                    yday_cusip_set_df["cusip"].to_list(),
                )
            )
            intraday_df = self._wsj_fetcher.wsj_timeseries_api(
                wsj_ticker_keys=cusip_icap_ticker_map.keys(), one_df=True, append_most_recent_last=True, show_tqdm=True
            )
            merged_series = intraday_df.apply(lambda col: col.dropna().iloc[-1] if not col.dropna().empty else pd.NA)
            latest_index: pd.Timestamp = max(intraday_df.index)
            merged_df = pd.DataFrame([merged_series], index=[latest_index])
            merged_df = merged_df.rename(columns=cusip_icap_ticker_map)

            ytm_mapping = merged_df.iloc[0].to_dict()

            yday_cusip_set_df["eod_ytm"] = yday_cusip_set_df["cusip"].map(ytm_mapping)
            yday_cusip_set_df["bid_ytm"] = yday_cusip_set_df["cusip"].map(ytm_mapping)
            yday_cusip_set_df["offer_ytm"] = yday_cusip_set_df["cusip"].map(ytm_mapping)
            yday_cusip_set_df["mid_ytm"] = yday_cusip_set_df["cusip"].map(ytm_mapping)
            yday_cusip_set_df = fedinvest_ust_back_out_price(fedinvest_df=yday_cusip_set_df, as_of_date=end_date)

            yday_cusip_set_df["timestamp"] = latest_index
            yday_cusip_set_df["hash"] = yday_cusip_set_df["cusip"].apply(lambda c: f"{c}_{latest_index}")

            cusip_set_dict_df[datetime(datetime.today().year, datetime.today().month, datetime.today().day)] = yday_cusip_set_df

        return cusip_set_dict_df

    def fetch_historical_curve_sets(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        fetch_soma_holdings: Optional[bool] = False,
        fetch_stripping_data: Optional[bool] = False,
        calc_free_float: Optional[bool] = False,
        pricer_n_jobs: Optional[int] = 1,
        max_concurrent_tasks: Optional[int] = 128,
        max_connections: Optional[int] = 64,
        sorted_curve_set: Optional[bool] = False,
    ) -> Dict[datetime, pd.DataFrame]:
        if not end_date:
            end_date = start_date

        keys = set()
        if fetch_soma_holdings:
            keys.add("soma_holdings")
        if fetch_stripping_data:
            keys.add("ust_stripping")
        if calc_free_float:
            fetch_soma_holdings = True
            fetch_stripping_data = True
            keys.add("soma_holdings")
            keys.add("ust_stripping")
        known_keys = list(keys)

        async def gather_tasks(client: httpx.AsyncClient, dates: datetime, max_concurrent_tasks):
            my_semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = []

            if fetch_soma_holdings:
                soma_bwd_date: pd.Timestamp = start_date - CustomBusinessDay(n=5, calendar=USFederalHolidayCalendar())
                tasks += await self._nyfrb_fetcher._build_fetch_tasks_historical_soma_holdings(
                    client=client,
                    dates=[soma_bwd_date.to_pydatetime()] + dates,
                    uid="soma_holdings",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            if fetch_stripping_data:
                strips_bwd_date: pd.Timestamp = start_date - CustomBusinessDay(n=20, calendar=USFederalHolidayCalendar())
                tasks += await self._treasurydirect_fetcher._build_fetch_tasks_historical_stripping_activity(
                    client=client,
                    dates=[strips_bwd_date.to_pydatetime()] + dates,
                    uid="ust_stripping",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING CURVE SETS...")

        async def run_fetch_all(dates: datetime, max_concurrent_tasks: int, max_connections: int):
            limits = httpx.Limits(max_connections=max_connections)
            async with httpx.AsyncClient(
                limits=limits, timeout=self._fedinvest_fetcher._global_timeout, mounts=self._fedinvest_fetcher._httpx_proxies, verify=False, http2=True
            ) as client:
                all_data = await gather_tasks(client=client, dates=dates, max_concurrent_tasks=max_concurrent_tasks)
                return all_data

        bdates = [
            bday.to_pydatetime()
            for bday in pd.bdate_range(
                start=start_date,
                end=end_date,
                freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
            )
        ]

        results: List[Tuple[datetime, pd.DataFrame, str]] = asyncio.run(
            run_fetch_all(dates=bdates, max_concurrent_tasks=max_concurrent_tasks, max_connections=max_connections)
        )
        grouped_results = {key: {} for key in known_keys}
        for dt, df, group_key in results:
            if group_key in grouped_results:
                grouped_results[group_key][dt] = df
            else:
                raise ValueError(f"Unexpected group key encountered: {group_key}")

        curve_sets = self.get_ust_cusip_sets_prices(start_date=start_date, end_date=end_date, n_jobs=pricer_n_jobs, return_filtered_cols=True, show_tqdm=True)

        auctions_df: pl.DataFrame = pl.from_pandas(self._treasurydirect_fetcher._historical_auctions_df.copy())
        auctions_df = auctions_df.filter((pl.col("security_type") == "Bill") | (pl.col("security_type") == "Note") | (pl.col("security_type") == "Bond"))
        auctions_df = auctions_df.with_columns(
            pl.when(pl.col("original_security_term").str.contains("29-Year"))
            .then(pl.lit("30-Year"))
            .when(pl.col("original_security_term").str.contains("30-"))
            .then(pl.lit("30-Year"))
            .otherwise(pl.col("original_security_term"))
            .alias("original_security_term")
        )

        curveset_dict_df: Dict[datetime, List[pd.DataFrame]] = {}

        for curr_dt, curve_set_df in tqdm.tqdm(curve_sets.items(), desc="AGGREGATING CUSIP SET DFs..."):
            last_seen_soma_holdings_df = None
            last_seen_stripping_act_df = None

            if fetch_soma_holdings:
                fetched_soma_holdings_dates = [dt for dt in grouped_results["soma_holdings"].keys() if dt < curr_dt]
                if fetched_soma_holdings_dates:
                    closest = max(fetched_soma_holdings_dates)
                    last_seen_soma_holdings_df = pl.from_pandas(grouped_results["soma_holdings"][closest])
                else:
                    raise ValueError("Couldnt find valid SOMA holding dates fetched")

            if fetch_stripping_data:
                fetched_ust_stripping_dates = [dt for dt in grouped_results["ust_stripping"].keys() if dt < curr_dt]
                if fetched_ust_stripping_dates:
                    closest = max(fetched_ust_stripping_dates)
                    last_seen_stripping_act_df = pl.from_pandas(grouped_results["ust_stripping"][closest])
                else:
                    raise ValueError("Couldnt find valid UST stripping dates fetched")

            price_df = pl.from_pandas(curve_set_df)
            curr_auctions_df = auctions_df.filter((pl.col("issue_date").dt.date() <= curr_dt.date()) & (pl.col("maturity_date") >= curr_dt)).unique(
                subset=["cusip"], keep="first"
            )
            curr_auctions_df = curr_auctions_df.filter(pl.col("cusip").is_in(price_df["cusip"].to_list()))

            merged_df = curr_auctions_df.join(price_df, on="cusip", how="full")

            if fetch_soma_holdings and last_seen_soma_holdings_df is not None:
                merged_df = merged_df.join(last_seen_soma_holdings_df, on="cusip", how="left")
            if fetch_stripping_data and last_seen_stripping_act_df is not None:
                merged_df = merged_df.join(last_seen_stripping_act_df, on="cusip", how="left")

            merged_df = merged_df.filter(pl.col("cusip").map_elements(is_valid_ust_cusip, return_dtype=pl.Boolean))
            merged_df = merged_df.with_columns(pl.col("maturity_date").cast(pl.Datetime).alias("maturity_date"))

            def quantlib_year_fraction(maturity_date, current_date):
                ql_maturity = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
                ql_today = ql.Date(current_date.day, current_date.month, current_date.year)
                day_counter = ql.ActualActual(ql.ActualActual.Actual365)
                return day_counter.yearFraction(ql_today, ql_maturity)

            merged_df = merged_df.with_columns(
                pl.col("maturity_date").map_elements(lambda d: quantlib_year_fraction(d, curr_dt), return_dtype=pl.Float64).alias("time_to_maturity")
            )
            merged_df = merged_df.with_columns(
                pl.col("time_to_maturity").rank(descending=True, method="ordinal").over("original_security_term").sub(1).alias("rank")
            )

            if calc_free_float:
                merged_df = merged_df.with_columns(
                    pl.col("soma_holdings_par_value").cast(pl.Float64).fill_null(0).alias("soma_holdings_par_value"),
                    (pl.col("portion_stripped_amt").cast(pl.Float64).fill_null(0) * 1000).alias("portion_stripped_amt"),
                    (
                        pl.when((pl.col("soma_holdings_est_outstanding_amt").is_not_nan()) & (pl.col("soma_holdings_est_outstanding_amt") != 0))
                        .then(pl.col("soma_holdings_est_outstanding_amt"))
                        .otherwise(pl.col("outstanding_amt"))
                        .cast(pl.Float64)
                        .fill_null(0)
                    ).alias("soma_holdings_est_outstanding_amt"),
                )
                merged_df = merged_df.with_columns(
                    ((pl.col("soma_holdings_est_outstanding_amt") - pl.col("soma_holdings_par_value") - pl.col("portion_stripped_amt")) / 1_000_000).alias(
                        "free_float"
                    )
                )

            curr_curve_set_df = merged_df.to_pandas()
            if sorted_curve_set:
                curr_curve_set_df["sort_key"] = curr_curve_set_df["original_security_term"].apply(ust_sorter)
                curr_curve_set_df = curr_curve_set_df.sort_values(by=["sort_key", "time_to_maturity"]).drop(columns="sort_key").reset_index(drop=True)

            if fetch_soma_holdings and fetch_stripping_data and calc_free_float:
                curveset_dict_df[curr_dt] = curr_curve_set_df[UST_DB_COLS].replace(r"^\s*$|^null$", np.nan, regex=True)

        return curveset_dict_df

    def fetch_otr_timeseries(self):
        ct_ts_df = self._wsj_fetcher.wsj_timeseries_api(
            wsj_ticker_keys=[
                "TMUBMUSD01M",
                "TMUBMUSD03M",
                "TMUBMUSD04M",
                "TMUBMUSD06M",
                "TMUBMUSD01Y",
                "TMUBMUSD02Y",
                "TMUBMUSD03Y",
                "TMUBMUSD05Y",
                "TMUBMUSD07Y",
                "TMUBMUSD10Y",
                "TMUBMUSD20Y",
                "TMUBMUSD30Y",
            ],
            # start_date=start_date,
            # end_date=end_date,
            # intraday_timestamp=True,
            append_most_recent_last=True,
            one_df=True,
            show_tqdm=True,
        )
        ct_ts_df.columns = [
            "CB1",
            "CB3",
            "CB4",
            "CB6",
            "CB12",
            "CT2",
            "CT3",
            "CT5",
            "CT7",
            "CT10",
            "CT20",
            "CT30",
        ]

        def merge_rows_same_date(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("The DataFrame index must be a DatetimeIndex.")

            grouped = df.groupby(df.index.normalize())

            merged_rows = []
            for group_date, group_df in grouped:
                group_df = group_df.sort_index()
                new_index = group_df.index.max()
                merged = group_df.iloc[-1].copy()
                for idx, row in group_df.iloc[-2::-1].iterrows():
                    merged = row.combine_first(merged)
                merged.name = new_index
                merged_rows.append(merged)

            merged_df = pd.DataFrame(merged_rows)
            merged_df.sort_index(inplace=True)
            return merged_df

        ct_ts_df = merge_rows_same_date(ct_ts_df)

        intraday = ct_ts_df.tail(1)
        ct_ts_df = ct_ts_df.head(-1)
        ct_ts_df.index = ct_ts_df.index.tz_localize(None)
        ct_ts_df.index = ct_ts_df.index.normalize() + pd.Timedelta(hours=17)
        ct_ts_df.index = ct_ts_df.index.tz_localize("US/Eastern")
        ct_ts_df.index = ct_ts_df.index.tz_convert(pytz.utc)
        ct_ts_df = pd.concat([ct_ts_df, intraday])

        return ct_ts_df.sort_index()
