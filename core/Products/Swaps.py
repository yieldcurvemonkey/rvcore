import re
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import QuantLib as ql
import tqdm
from joblib import Parallel, delayed
from sqlalchemy import Engine

from core.Fetchers.CMEFetcherV2 import CMEFetcherV2
from core.Fetchers.ErisFuturesFetcher import ErisFuturesFetcher
from core.Products.CurveBuilding.AlchemyQLBootstrapperWrapper import AlchemyQLBootstrapperWrapper
from core.Products.BaseProductPlotter import BaseProductPlotter
from core.Products.CurveBuilding.QLBootstrapper import QLBootstrapper
from core.Products.CurveBuilding.ql_curve_building import build_ql_discount_curve, get_ql_swaps_curve_params
from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, most_recent_business_day_ql, ql_date_to_datetime, ql_period_to_months


class Swaps(BaseProductPlotter):
    _data_source: Literal["CME", "JPM", "SKY", "CSV_{path}"] = None
    _curve: str = None

    _swaps_hist_data_fetcher: CMEFetcherV2 | ErisFuturesFetcher | QLBootstrapper | AlchemyQLBootstrapperWrapper = None
    _swaps_hist_timeseries_csv_df: pd.DataFrame = None
    _swaps_db_engine: Engine = None

    _ql_day_count: ql.DayCounter = None
    _ql_bday_convention = None
    _ql_calendar: ql.Calendar = None
    _ql_compounded: Literal["ql.Compounding"] = None

    _ql_curve_cache: Dict[datetime, ql.YieldTermStructure] = None
    _fwd_swaps_timeseries_cache: Dict[Tuple[datetime, Tuple[str, ...], Tuple[str, ...]], Dict[str, Union[datetime, float]]] = None

    __data_sources_funcs: Dict = None

    _pre_fetch_curves: bool = False
    _PRE_FETCH_CURVE_PERIOD = ql.Period("-5Y")
    _date_col: str = None
    _bootstrap_cols: List[str] = None
    _default_tz = "US/Eastern"

    _show_tqdm: bool = None
    _proxies: Dict[str, str] = None

    _default_fwd_swap_tenors = ["0D", "01M", "03M", "06M", "01Y", "02Y", "05Y", "07Y", "10Y"]
    _default_underlying_tenors = ["01M", "03M", "06M", "09M", "01Y", "18M", "02Y", "03Y", "04Y", "05Y", "06Y", "07Y", "08Y", "09Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"] # fmt: skip
    _DEFAULT_SWAP_PREFIX = "SWAP"
    _DEFAULT_MATURITY_MATCHED_PREFIX = "MMS"  # e.g. f"MMS_{datetime.strftime("%b_%d_%Y")}"
    _SWAP_CURVE_FETCH_FUNC_INPUT_DATES = "_INPUT_DATES"
    _SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE = "_BOOTSTRAP_TDQM_MESSAGE"
    _SWAP_CURVE_BOOTSTRAPPER_NJOBS = "_BOOTSTRAP_NJOBS"
    _HSTORE_QL_CURVE_POSTFIX = "ql_curve_nodes"
    _MAX_NJOBS = None

    _FETCHER_TIMEOUT = 10
    _MAX_CONNECTIONS = 64
    _MAX_KEEPALIVE_CONNECTIONS = 5

    def __init__(
        self,
        data_source: Literal["CME", "ERIS", "JPM", "SKY", "CSV_{path}"] | Engine,
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
        ql_interpolation_algo: Optional[
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
        date_col: Optional[str] = "Date",
        bootstrap_cols: Optional[List[str]] = [
            "1M",
            "2M",
            "3M",
            "4M",
            "5M",
            "6M",
            "7M",
            "8M",
            "9M",
            "10M",
            "11M",
            "12M",
            "18M",
            "2Y",
            "3Y",
            "4Y",
            "5Y",
            "6Y",
            "7Y",
            "8Y",
            "9Y",
            "10Y",
            "15Y",
            "20Y",
            "25Y",
            "30Y",
            "40Y",
            "50Y",
        ],
        pre_fetch_curves: Optional[bool] = False,
        max_njobs: Optional[int] = 1,
        show_tqdm: Optional[bool] = True,
        proxies: Optional[Dict[str, str]] = None,
        info_verbose: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )

        # caches are instance-specific
        self._ql_curve_cache = {}
        self._fwd_swaps_timeseries_cache = {}
        self.__data_sources_funcs = {}
        self._date_col = date_col

        if not isinstance(data_source, str):
            self._swaps_db_engine = data_source
            self._data_source = "ENGINE"
        else:
            if data_source in ["JPM", "SKY"]:
                raise NotImplementedError(f"data source not implemented")

            if "CSV_" in data_source:
                csv_df = pd.read_csv(data_source.split("_", 1)[-1])
                csv_df[self._date_col] = pd.to_datetime(csv_df[self._date_col], errors="coerce", format="mixed")
                self._swaps_hist_timeseries_csv_df = csv_df.set_index(self._date_col)
                self._data_source = "CSV"
            else:
                self._data_source = data_source

        self._curve = curve
        curve_params = get_ql_swaps_curve_params(self._curve)
        self._ql_day_count = curve_params[2]
        self._ql_calendar = curve_params[3]
        self._ql_bday_convention = curve_params[4]
        self._ql_interpolation_algo = ql_interpolation_algo

        self._bootstrap_cols = bootstrap_cols

        self._pre_fetch_curves = pre_fetch_curves
        self._MAX_NJOBS = max_njobs
        self._show_tqdm = show_tqdm
        self._proxies = proxies

        self._info_verbose = info_verbose
        self._debug_verbose = debug_verbose
        self._warning_verbose = warning_verbose
        self._error_verbose = error_verbose

        self.__data_sources_funcs = {
            "HIST_SWAPS": {
                "CME": {
                    "obj": CMEFetcherV2,
                    "init_args": (
                        self._FETCHER_TIMEOUT,
                        self._proxies,
                        self._debug_verbose,
                        self._info_verbose,
                        self._warning_verbose,
                        self._error_verbose,
                    ),
                    "fetch_func": "build_ql_eod_curves",
                    "fetch_func_args": (
                        self._curve,
                        "Df",
                        self._ql_day_count,
                        self._ql_calendar,
                        None,
                        None,
                        self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_interpolation_algo,
                        True,
                        self._show_tqdm,
                        self._MAX_CONNECTIONS,
                        self._MAX_KEEPALIVE_CONNECTIONS,
                    ),
                },
                "ERIS": {
                    "obj": ErisFuturesFetcher,
                    "init_args": (
                        self._FETCHER_TIMEOUT,
                        self._proxies,
                        self._debug_verbose,
                        self._info_verbose,
                        self._warning_verbose,
                        self._error_verbose,
                    ),
                    "fetch_func": "fetch_historical_eod_discount_curves",
                    "fetch_func_args": (
                        None,
                        None,
                        self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_day_count,
                        self._ql_calendar,
                        self._show_tqdm,
                        self._ql_interpolation_algo,
                        True,
                    ),
                },
                "CSV": {
                    "obj": QLBootstrapper,
                    "init_args": (
                        self._swaps_hist_timeseries_csv_df,
                        None,
                    ),
                    "fetch_func": "parallel_swap_curve_bootstrapper",
                    "fetch_func_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_interpolation_algo,
                        True,
                        self._show_tqdm,
                        self._SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                        self._SWAP_CURVE_BOOTSTRAPPER_NJOBS,
                    ),
                },
                "ENGINE": {
                    "obj": AlchemyQLBootstrapperWrapper,
                    "init_args": (self._swaps_db_engine, self._date_col, None, self._bootstrap_cols),
                    "fetch_func": "ql_swap_curve_bootstrap_wrapper",
                    "fetch_func_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_interpolation_algo,
                        True,
                        self._show_tqdm,
                        self._SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                        self._SWAP_CURVE_BOOTSTRAPPER_NJOBS,
                        True,
                    ),
                },
                "JPM": None,
                "SKY": None,
            }
        }

        self._swaps_hist_data_fetcher = self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["obj"](
            *self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["init_args"]
        )

        if self._pre_fetch_curves:
            pre_fetch_end_date = most_recent_business_day_ql(ql_calendar=self._ql_calendar, tz=self._default_tz, to_pydate=True)
            pre_fetch_start_date = ql_date_to_datetime(self._ql_calendar.advance(datetime_to_ql_date(pre_fetch_end_date), self._PRE_FETCH_CURVE_PERIOD))
            self._fetch_ql_swap_curves(start_date=pre_fetch_start_date, end_date=pre_fetch_end_date)

    def _get_latest_cached_curve(self):
        d = self._ql_curve_cache
        k, last_value = _, d[k] = d.popitem()
        return k, last_value

    def _fetch_latest_curve(self):
        hist_swaps_curve_fetch_func: callable = getattr(self._swaps_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func"])
        hist_swaps_curve_fetch_func_args_today = tuple(
            (
                [most_recent_business_day_ql(self._ql_calendar, self._default_tz, to_pydate=True)]
                if arg == self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES
                else (
                    "BOOTSTRAPPING INTRADAY SWAPS CURVE..."
                    if arg == self._SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                    else 1 if arg == self._SWAP_CURVE_BOOTSTRAPPER_NJOBS else arg
                )
            )
            for arg in self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func_args"]
        )
        todays_result: Dict[pd.Timestamp, ql.DiscountCurve] = hist_swaps_curve_fetch_func(*hist_swaps_curve_fetch_func_args_today)

        return todays_result.popitem()

    def _fetch_ql_swap_curves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
        to_pydt: Optional[bool] = False,
    ) -> Dict[datetime, ql.YieldTermStructure]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._ql_calendar) if (start_date and end_date) else bdates
        if not end_date and bdates:
            end_date = max(bdates)

        today = datetime.today().date()
        non_today_bdates = [bday for bday in input_bdates if bday.date() != today]
        todays_bdates = [bday for bday in input_bdates if bday.date() == today]

        if not refresh_cache:
            cached_dates = [dt.date() for dt in self._ql_curve_cache.keys()]
            non_today_bdates_not_cached = [bday for bday in non_today_bdates if bday.date() not in cached_dates]
        else:
            non_today_bdates_not_cached = non_today_bdates

        if non_today_bdates_not_cached:
            hist_swaps_curve_fetch_func: callable = getattr(self._swaps_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func"])
            hist_swaps_curve_fetch_func_args = tuple(
                (
                    non_today_bdates_not_cached
                    if arg == self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING HISTORICAL SWAPS CURVE..."
                        if arg == self._SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else (
                            self._MAX_NJOBS
                            if arg == self._SWAP_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) > 1
                            else 1 if arg == self._SWAP_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) == 1 else arg
                        )
                    )
                )
                for arg in self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func_args"]
            )
            results = hist_swaps_curve_fetch_func(*hist_swaps_curve_fetch_func_args)
            self._ql_curve_cache = self._ql_curve_cache | results

        todays_result: Dict[pd.Timestamp, ql.DiscountCurve] = {}
        if todays_bdates:
            hist_swaps_curve_fetch_func: callable = getattr(self._swaps_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func"])
            hist_swaps_curve_fetch_func_args_today = tuple(
                (
                    todays_bdates
                    if arg == self._SWAP_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING INTRADAY SWAPS CURVE..."
                        if arg == self._SWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else 1 if arg == self._SWAP_CURVE_BOOTSTRAPPER_NJOBS else arg
                    )
                )
                for arg in self.__data_sources_funcs["HIST_SWAPS"][self._data_source]["fetch_func_args"]
            )
            todays_result = hist_swaps_curve_fetch_func(*hist_swaps_curve_fetch_func_args_today)

        final_result = {}
        timestamp_to_datetime_map = {ts.date(): ts for ts in self._ql_curve_cache.keys()}
        for bday in non_today_bdates:
            if bday.date() in timestamp_to_datetime_map:
                if to_pydt:
                    final_result[bday] = self._ql_curve_cache[timestamp_to_datetime_map[bday.date()]]
                else:
                    final_result[timestamp_to_datetime_map[bday.date()]] = self._ql_curve_cache[timestamp_to_datetime_map[bday.date()]]

        if bool(todays_result):
            if to_pydt:
                todays_items = next(iter(todays_result.items()))
                tdate = todays_items[0].date()
                final_result[datetime(tdate.year, tdate.month, tdate.day)] = todays_items[1]
            else:
                final_result.update(todays_result)

        return final_result

    def _get_nodes_dict(self, ql_curve: ql.YieldTermStructure) -> Dict[datetime, float]:
        if hasattr(ql_curve, "nodes"):
            nodes = ql_curve.nodes()
        else:
            nodes = list(zip(ql_curve.dates(), ql_curve.discounts()))
        return {ql_date_to_datetime(node[0]): node[1] for node in nodes}

    def _build_fwd_swaps_timeseries(
        self,
        ql_curves_ts_dict: Dict[datetime, ql.DiscountCurve],
        fwd_tenors: List[str],
        underlying_tenors: List[str],
        matched_maturities: Optional[List[datetime]] = None,
        n_jobs: Optional[int] = 1,
    ) -> pd.DataFrame:
        cleaned_fwd_tenors = []
        for tenor in fwd_tenors:
            tenor_str = tenor.split("_")[1] if self._DEFAULT_SWAP_PREFIX in tenor else tenor
            cleaned_fwd_tenors.append(tenor_str)

        items_to_process = []
        cached_results = []
        for curr_date, ql_curve in ql_curves_ts_dict.items():
            cache_key = (curr_date, tuple(sorted(cleaned_fwd_tenors)), tuple(sorted(underlying_tenors)), tuple(sorted(matched_maturities)))
            if cache_key in self._fwd_swaps_timeseries_cache:
                cached_results.append(self._fwd_swaps_timeseries_cache[cache_key])
            else:
                ql_curve_nodes = self._get_nodes_dict(ql_curve)
                items_to_process.append((curr_date, ql_curve_nodes, cache_key))

        swaps_iter = tqdm.tqdm(items_to_process, desc="PRICING SWAPS...") if self._show_tqdm else items_to_process
        new_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_curve_item_with_cache)(
                curr_date,
                ql_curve_nodes,
                cleaned_fwd_tenors,
                underlying_tenors,
                self._curve,
                self._DEFAULT_SWAP_PREFIX,
                self._ql_interpolation_algo,
                cache_key,
                matched_maturities,
                self._DEFAULT_MATURITY_MATCHED_PREFIX,
                self._date_col,
            )
            for curr_date, ql_curve_nodes, cache_key in swaps_iter
        )
        for ck, result in new_results:
            self._fwd_swaps_timeseries_cache[ck] = result
        all_results = cached_results + [result for _, result in new_results]

        timeseries_df = pd.DataFrame(all_results)
        if timeseries_df.empty:
            return pd.DataFrame([], index=pd.DatetimeIndex([]))
        timeseries_df[self._date_col] = pd.to_datetime(timeseries_df[self._date_col], errors="coerce")
        timeseries_df.set_index(self._date_col, inplace=True)
        return timeseries_df

    def _build_fwd_swaps_term_structure_grid_timeseries(
        self,
        ql_curves_ts_dict: Dict[datetime, ql.YieldTermStructure],
        fwd_tenors: List[str],
        underlying_tenors: List[str],
        n_jobs: Optional[int] = 1,
        tenor_col: Optional[str] = "Tenor",
    ) -> Dict[datetime, pd.DataFrame]:
        swaps_iter = tqdm.tqdm(ql_curves_ts_dict.items(), desc="PRICING SWAPS...") if self._show_tqdm else ql_curves_ts_dict.items()
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_date_grid)(
                curr_date,
                self._get_nodes_dict(ql_curve),
                fwd_tenors,
                underlying_tenors,
                self._curve,
                self._DEFAULT_SWAP_PREFIX,
                self._ql_interpolation_algo,
                tenor_col,
            )
            for curr_date, ql_curve in swaps_iter
        )
        fwd_term_structure_grids = {curr_date: df for curr_date, df in results}
        return fwd_term_structure_grids

    def _extract_swap_tenors(self, s: str):
        pattern = r"(?:[0-9.]+\*)?((?:\d+[YMD]){1,2})x((?:\d+[YMD]){1,2})"
        matches = re.findall(pattern, s)
        fwd_tenors = {match[0] for match in matches}
        underlying_tenors = {match[1] for match in matches}
        return fwd_tenors, underlying_tenors

    def _extract_mss_dates(self, s: str, prefix: Optional[str] = "MMS") -> List[datetime]:
        pattern = re.escape(prefix) + r"_(\w{3})_(\d{1,2})_(\d{4})"
        matches = re.findall(pattern, s)
        dates = []
        for match in matches:
            date_str = "_".join(match)
            try:
                dt = datetime.strptime(date_str, "%b_%d_%Y")
                dates.append(dt)
            except ValueError as e:
                self._logger.error(f"Error parsing MMS date '{date_str}': {e}")
        return dates

    def swaps_timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        cols: List[str | Tuple[str, str]],
        return_components: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ) -> pd.DataFrame:
        extracted_fwd_tenors = set()
        extracted_underlying_tenors = set()
        for col in cols:
            if isinstance(col, tuple):
                col = col[0]
            if self._DEFAULT_SWAP_PREFIX in col:
                curr_ex_fwds, curr_ex_unds = self._extract_swap_tenors(s=col)
                extracted_fwd_tenors.update(curr_ex_fwds)
                extracted_underlying_tenors.update(curr_ex_unds)

        extracted_mms_dates = set()
        for col in cols:
            if isinstance(col, tuple):
                col = col[0]
            if self._DEFAULT_MATURITY_MATCHED_PREFIX in col:
                curr_mms_dates = self._extract_mss_dates(s=col, prefix=self._DEFAULT_MATURITY_MATCHED_PREFIX)
                extracted_mms_dates.update(curr_mms_dates)

        ts_df = self._build_fwd_swaps_timeseries(
            ql_curves_ts_dict=self._fetch_ql_swap_curves(start_date=start_date, end_date=end_date),
            fwd_tenors=extracted_fwd_tenors,
            underlying_tenors=extracted_underlying_tenors,
            n_jobs=n_jobs,
            matched_maturities=extracted_mms_dates,
        )
        if ts_df.empty:
            raise ValueError("'swaps_timeseries_builder': Dataframe is empty")

        cols_to_return = []

        if return_components:
            cols_to_return = list(ts_df.columns)

        for col in cols:
            try:
                if isinstance(col, tuple):
                    ts_df[col[1]] = ts_df.eval(col[0])
                    cols_to_return.append(col[1])
                else:
                    ts_df[col] = ts_df.eval(col)
                    cols_to_return.append(col)

            except Exception as e:
                self._logger.error(f"'timeseries_builder' eval failed for {col}: {e}")

        ts_df = ts_df[cols_to_return].sort_index()
        try:
            ts_df.index = ts_df.index.tz_convert(self._default_tz)
        except Exception as e:
            self._logger.error(f"'timeseries_builder' failed to convert to {self._default_tz}: {e}")
        return ts_df

    def swaps_term_structure_plotter(
        self,
        dates: List[datetime],
        use_plotly: Optional[bool] = False,
        fwd_tenors: Optional[List[str]] = ["0D"],
        swap_tenors: Optional[List[str]] = None,
        n_jobs: Optional[int] = 1,
    ):
        swap_tenors = self._default_underlying_tenors if not swap_tenors else swap_tenors
        swaps_term_structure_dict_df = self._build_fwd_swaps_term_structure_grid_timeseries(
            ql_curves_ts_dict=self._fetch_ql_swap_curves(bdates=dates), fwd_tenors=fwd_tenors, underlying_tenors=swap_tenors, n_jobs=n_jobs
        )
        self._term_structure_plotter(
            term_structure_dict_df=swaps_term_structure_dict_df,
            plot_title=f"{self._curve} Curve",
            x_axis_col_sorter_func=lambda x: ql_period_to_months(ql.Period(x)),
            x_axis_title="Term",
            y_axis_title="Rate",
            use_plotly=use_plotly,
        )


def _process_curve_item_with_cache(
    curr_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    cleaned_fwd_tenors: List[str],
    underlying_tenors: List[str],
    curve: str,
    default_swap_prefix: str,
    ql_interpolation_algop_str: str,
    cache_key: Tuple[datetime, Tuple[str, ...], Tuple[str, ...]],
    matched_maturities: Optional[List[datetime]] = None,
    default_mss_prefix: Optional[str] = None,
    date_col: Optional[str] = "Date",
) -> Tuple[Tuple[datetime, Tuple[str, ...], Tuple[str, ...]], Dict[str, Union[datetime, float]]]:
    result = _process_curve_item(
        curr_date=curr_date,
        ql_curve_nodes=ql_curve_nodes,
        fwd_tenors=cleaned_fwd_tenors,
        underlying_tenors=underlying_tenors,
        curve=curve,
        default_swap_prefix=default_swap_prefix,
        ql_interpolation_algo_str=ql_interpolation_algop_str,
        matched_maturities=matched_maturities,
        default_mss_prefix=default_mss_prefix,
        date_col=date_col,
    )
    return cache_key, result


def _process_curve_item(
    curr_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    fwd_tenors: List[str],
    underlying_tenors: List[str],
    curve: str,
    default_swap_prefix: str,
    ql_interpolation_algo_str: str,
    matched_maturities: Optional[List[datetime]] = None,
    default_mss_prefix: Optional[str] = None,
    date_col: Optional[str] = "Date",
):
    ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_date)

    row = {date_col: curr_date}
    ql_floating_index_obj, is_ois, ql_dc, ql_cal = get_ql_swaps_curve_params(curve)[:4]
    ql_curve = build_ql_discount_curve(
        datetime_series=pd.Series(ql_curve_nodes.keys()),
        discount_factor_series=pd.Series(ql_curve_nodes.values()),
        ql_dc=ql_dc,
        ql_cal=ql_cal,
        interpolation_algo=f"df_{ql_interpolation_algo_str}",
    )
    ql_curve.enableExtrapolation()

    curr_ql_yts = ql.YieldTermStructureHandle(ql_curve)
    curr_ql_swap_engine = ql.DiscountingSwapEngine(curr_ql_yts)
    curr_ql_floating_index = ql_floating_index_obj(curr_ql_yts)

    for fwd_tenor_str in fwd_tenors:
        for underlying_tenor_str in underlying_tenors:
            try:
                if is_ois:
                    swap = ql.MakeOIS(
                        swapTenor=ql.Period(underlying_tenor_str),
                        overnightIndex=curr_ql_floating_index,
                        fixedRate=0,
                        fwdStart=ql.Period(fwd_tenor_str),
                        pricingEngine=curr_ql_swap_engine,
                    )
                else:
                    swap = ql.MakeVanillaSwap(
                        swapTenor=ql.Period(underlying_tenor_str),
                        iborIndex=curr_ql_floating_index,
                        fixedRate=0,
                        forwardStart=ql.Period(fwd_tenor_str),
                        pricingEngine=curr_ql_swap_engine,
                    )
                par_rate = swap.fairRate() * 100
            except Exception:
                par_rate = float("nan")

            col_name = f"{default_swap_prefix}_{fwd_tenor_str}x{underlying_tenor_str}"
            row[col_name] = par_rate

    if matched_maturities and default_mss_prefix:
        for matched_maturity in matched_maturities:
            try:
                if is_ois:
                    swap = ql.MakeOIS(
                        swapTenor=ql.Period("-0D"),
                        fixedRate=-0,
                        overnightIndex=curr_ql_floating_index,
                        effectiveDate=datetime_to_ql_date(curr_date),
                        terminationDate=datetime_to_ql_date(matched_maturity),
                        pricingEngine=curr_ql_swap_engine,
                    )
                else:
                    swap = ql.MakeVanillaSwap(
                        swapTenor=ql.Period("-0D"),
                        fixedRate=-0,
                        iborIndex=curr_ql_floating_index,
                        effectiveDate=datetime_to_ql_date(curr_date),
                        terminationDate=datetime_to_ql_date(matched_maturity),
                        pricingEngine=curr_ql_swap_engine,
                    )
                par_rate = swap.fairRate() * 100
            except Exception:
                par_rate = float("nan")

            col_name = f"{default_mss_prefix}_{matched_maturity.strftime("%b_%d_%Y")}"
            row[col_name] = par_rate

    return row


def _process_date_grid(
    curr_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    fwd_tenors: List[str],
    underlying_tenors: List[str],
    curve: str,
    default_swap_prefix: str,
    ql_interpolation_algo_str: str,
    tenor_col: Optional[str] = "Tenor",
) -> Tuple[datetime, pd.DataFrame]:
    row = _process_curve_item(
        curr_date,
        ql_curve_nodes,
        fwd_tenors,
        underlying_tenors,
        curve,
        default_swap_prefix,
        ql_interpolation_algo_str,
    )
    grid = {}
    for fwd in fwd_tenors:
        col_name = f"{default_swap_prefix}_{fwd}"
        grid[col_name] = []
        for underlying in underlying_tenors:
            key = f"{default_swap_prefix}_{fwd}x{underlying}"
            grid[col_name].append(row.get(key, float("nan")))
    df = pd.DataFrame(grid, index=[str(u) for u in underlying_tenors])
    df[tenor_col] = [str(u) for u in underlying_tenors]
    df = df.set_index(tenor_col)
    return curr_date, df.T
