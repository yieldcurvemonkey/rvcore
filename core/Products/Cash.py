import re
from datetime import datetime
from typing import Annotated, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import pytz
import QuantLib as ql
from scipy.interpolate import interp1d
from sqlalchemy import Engine

from core.Fetchers.WSJFetcher import WSJFetcher
from core.Products.BaseProductPlotter import BaseProductPlotter
from core.Products.CurveBuilding.Cash.AlchemyCashCurveBootstrapperWrapper import AlchemyCashCurveBootstrapperWrapper
from core.Products.CurveBuilding.Cash.CashCurveBootstrapper import CashCurveBootstrapper
from core.Products.CurveBuilding.ql_curve_params import GOVIE_CURVE_QL_PARAMS
from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, most_recent_business_day_ql, ql_date_to_datetime, ql_period_to_months
from core.utils.ust_viz import plot_usts


class Cash(BaseProductPlotter):
    _data_source: Literal["CME", "JPM", "SKY", "CSV_{path}"] = None
    _curve: Literal["USD", "EUR", "GBP"] = None

    _cash_data_fetcher: WSJFetcher | CashCurveBootstrapper | AlchemyCashCurveBootstrapperWrapper = None
    _cash_hist_timeseries_csv_df: pd.DataFrame = None
    _cash_db_engine: Engine = None

    _cash_cache: Dict[datetime, pd.DataFrame] = None
    _ql_curve_cache: Dict[datetime, ql.DiscountCurve] = None
    _scipy_curve_cache: Dict[datetime, ql.DiscountCurve] = None

    __data_sources_funcs: Dict = {}

    _par_curve_model_key: str = None
    _par_curve_model_kwags_ex_cusip_set: Dict = None

    _pre_fetch_par_curves: bool = None
    _PRE_FETCH_CURVE_PERIOD = ql.Period("-1Y")
    _MAX_NJOBS = None
    _show_tqdm: bool = None
    _proxies: Dict[str, str] = None

    _DEFAULT_PAR_PREFIX = "PAR"
    _ql_calendar: ql.Calendar = None
    _ql_day_count: ql.DayCounter = None
    _ql_compounding_freq = None
    _default_tz = "US/Eastern"
    _benchmark_tenors: List[str] = None

    _FETCHER_TIMEOUT = 10
    _MAX_CONNECTIONS = 64
    _MAX_KEEPALIVE_CONNECTIONS = 5

    _CASH_CURVE_FETCH_FUNC_INPUT_DATES = "_INPUT_DATES"
    _CASH_CURVE_TIMESERIES_FUNC_START_DATE = "_START_DATE"
    _CASH_CURVE_TIMESERIES_FUNC_END_DATE = "_END_DATE"
    _CASH_CURVE_TIMESERIES_SEARCH_PARAMS = "_SEARCH_PARAMS"
    _CASH_CURVE_TIMESERIES_SUBSET_COLS = "_SUBSET_COLS"
    _CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE = "_BOOTSTRAP_TDQM_MESSAGE"
    _CASH_CURVE_BOOTSTRAPPER_NJOBS = "_BOOTSTRAP_NJOBS"

    _cubic_knots: List[float] = None
    _poly_degree: int = None
    _timestamp_col: str = None
    _hash_col: str = None

    def __init__(
        self,
        data_source: Literal["WSJ", "JPM", "SKY", "CSV_{path}"] | Engine,
        curve: Literal["USD", "EUR", "GBP"],
        par_curve_model_key: Literal["KEYS OF 'PAR_CURVE_MODELS'"],
        par_curve_model_kwags_ex_cusip_set: Dict,
        timestamp_col: Optional[str] = "timestamp",
        hash_col: Optional[str] = "hash",
        pre_fetch_par_curves: Optional[bool] = False,
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
        self._cash_cache = {}
        self._ql_curve_cache = {}
        self._scipy_curve_cache = {}

        self._timestamp_col = timestamp_col
        self._hash_col = hash_col

        if not isinstance(data_source, str):
            self._cash_db_engine = data_source
            self._data_source = "ENGINE"
        else:
            if data_source in ["SKY", "JPM", "SDR"]:
                raise NotImplementedError(f"data source not implemented")

            if "CSV_" in data_source:
                csv_df = pd.read_csv(data_source.split("_", 1)[-1])
                self._cash_hist_timeseries_csv_df = csv_df.set_index(self._hash_col)
                self._cash_hist_timeseries_csv_df[self._timestamp_col] = pd.to_datetime(
                    self._cash_hist_timeseries_csv_df[self._timestamp_col], errors="coerce", format="mixed", utc=True
                )
                self._cash_hist_timeseries_csv_df[f"{self._timestamp_col}_groupby"] = self._cash_hist_timeseries_csv_df[self._timestamp_col].apply(
                    lambda d: datetime(d.year, d.month, d.day)
                )
                self._data_source = "CSV"
            else:
                self._data_source = data_source
                raise NotImplementedError()

        self._curve = curve
        self._par_curve_model_key = par_curve_model_key
        self._par_curve_model_kwags_ex_cusip_set = par_curve_model_kwags_ex_cusip_set
        self._pre_fetch_par_curves = pre_fetch_par_curves
        self._MAX_NJOBS = max_njobs
        self._show_tqdm = show_tqdm
        self._proxies = proxies

        self._info_verbose = info_verbose
        self._debug_verbose = debug_verbose
        self._warning_verbose = warning_verbose
        self._error_verbose = error_verbose

        self.__data_sources_funcs = {
            "HIST_CASH": {
                "CSV": {
                    "USD": {
                        "obj": CashCurveBootstrapper,
                        "init_args": (
                            {
                                key.tz_localize(None).to_pydatetime() if key.tzinfo is not None else key.to_pydatetime(): group
                                for key, group in self._cash_hist_timeseries_csv_df.groupby(by=f"{self._timestamp_col}_groupby")
                            }
                            if self._cash_hist_timeseries_csv_df is not None
                            else None
                        ),
                        "fetch_ql_curve": "parallel_ql_cash_curve_bootstraper",
                        "fetch_ql_curve_args": (
                            self._curve,
                            self._par_curve_model_key,
                            self._par_curve_model_kwags_ex_cusip_set,
                            None,
                            None,
                            self._CASH_CURVE_FETCH_FUNC_INPUT_DATES,
                            True,  # extrapolation
                            self._show_tqdm,
                            self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                            self._CASH_CURVE_BOOTSTRAPPER_NJOBS,
                        ),
                        "fetch_scipy_spline": "parallel_scipy_cash_curve_bootstraper",
                        "fetch_scipy_spline_args": (
                            self._curve,
                            self._par_curve_model_key,
                            self._par_curve_model_kwags_ex_cusip_set,
                            None,
                            None,
                            self._CASH_CURVE_FETCH_FUNC_INPUT_DATES,
                            self._show_tqdm,
                            self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                            self._CASH_CURVE_BOOTSTRAPPER_NJOBS,
                        ),
                        "fetch_timeseries_df": "hoist_df",
                        "fetch_timeseries_df_args": (
                            self._cash_hist_timeseries_csv_df,
                            self._CASH_CURVE_TIMESERIES_FUNC_START_DATE,
                            self._CASH_CURVE_TIMESERIES_FUNC_END_DATE,
                            self._timestamp_col,
                        ),
                    },
                },
                "ENGINE": {
                    "USD": {
                        "obj": AlchemyCashCurveBootstrapperWrapper,
                        "init_args": (self._cash_db_engine, self._timestamp_col, self._hash_col, "%Y-%m-%d %H:%M:%S%z"),
                        "fetch_ql_curve": "ql_cash_curve_bootstrap_wrapper",
                        "fetch_ql_curve_args": (
                            self._curve,
                            self._par_curve_model_key,
                            self._par_curve_model_kwags_ex_cusip_set,
                            None,
                            None,
                            self._CASH_CURVE_FETCH_FUNC_INPUT_DATES,
                            True,  # extrapolation
                            self._show_tqdm,
                            self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                            self._CASH_CURVE_BOOTSTRAPPER_NJOBS,
                            True,  # utc
                        ),
                        "fetch_scipy_spline": "scipy_cash_curve_bootstrap_wrapper",
                        "fetch_scipy_spline_args": (
                            self._curve,
                            self._par_curve_model_key,
                            self._par_curve_model_kwags_ex_cusip_set,
                            None,
                            None,
                            self._CASH_CURVE_FETCH_FUNC_INPUT_DATES,
                            self._show_tqdm,
                            self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                            self._CASH_CURVE_BOOTSTRAPPER_NJOBS,
                            True,  # utc
                        ),
                        "fetch_timeseries_df": "_fetch_by_col_values",
                        "fetch_timeseries_df_args": (
                            self._curve,
                            self._CASH_CURVE_TIMESERIES_SEARCH_PARAMS,
                            True,
                            "and",
                            self._CASH_CURVE_TIMESERIES_FUNC_START_DATE,
                            self._CASH_CURVE_TIMESERIES_FUNC_END_DATE,
                            None,
                            self._CASH_CURVE_TIMESERIES_SUBSET_COLS,
                            True,
                        ),
                        "fetch_otr_timeseries_df": "_fetch_df_by_dates",
                        "fetch_otr_timeseries_df_args": (
                            f"{self._curve}_otr_timeseries",
                            self._CASH_CURVE_TIMESERIES_FUNC_START_DATE,
                            self._CASH_CURVE_TIMESERIES_FUNC_END_DATE,
                            None,
                            None,
                            False,
                            True,
                        ),
                    }
                },
                "JPM": None,
                "SKY": None,
            },
        }

        self._ql_calendar = GOVIE_CURVE_QL_PARAMS[curve]["calendar"]
        self._ql_day_count = GOVIE_CURVE_QL_PARAMS[curve]["dayCounter"]
        self._ql_compounding_freq = GOVIE_CURVE_QL_PARAMS[curve]["frequency"]
        self._benchmark_tenors = GOVIE_CURVE_QL_PARAMS[curve]["OTR_TENORS"]

        self._cash_data_fetcher = self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["obj"](
            *self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["init_args"]
        )

        if self._pre_fetch_par_curves:
            pre_fetch_end_date = most_recent_business_day_ql(ql_calendar=self._ql_calendar, tz=self._default_tz, to_pydate=True)
            pre_fetch_start_date = ql_date_to_datetime(self._ql_calendar.advance(datetime_to_ql_date(pre_fetch_end_date), self._PRE_FETCH_CURVE_PERIOD))
            self._fetch_ql_cash_curves(start_date=pre_fetch_start_date, end_date=pre_fetch_end_date)

            if self._curve_interpolator_func_str:
                self._fetch_scipy_cash_splines(start_date=pre_fetch_start_date, end_date=pre_fetch_end_date)

    def _fetch_ql_cash_curves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, ql.DiscountCurve | ql.YieldTermStructure]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._ql_calendar) if (start_date and end_date) else bdates
        if not end_date and bdates:
            end_date = max(bdates)

        today = datetime.today().date()
        non_today_bdates = [bday for bday in input_bdates if bday.date() != today]
        todays_bdates = [bday for bday in input_bdates if bday.date() == today]

        if not refresh_cache:
            non_today_bdates_not_cached = [bday for bday in non_today_bdates if bday not in self._ql_curve_cache]
        else:
            non_today_bdates_not_cached = non_today_bdates

        if non_today_bdates_not_cached:
            hist_cash_curve_fetch_func: callable = getattr(
                self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_ql_curve"]
            )
            hist_cash_curve_fetch_func_args = tuple(
                (
                    non_today_bdates_not_cached
                    if arg == self._CASH_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING HISTORICAL CASH CURVE..."
                        if arg == self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else (
                            self._MAX_NJOBS
                            if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) > 1
                            else 1 if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) == 1 else arg
                        )
                    )
                )
                for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_ql_curve_args"]
            )
            results = hist_cash_curve_fetch_func(*hist_cash_curve_fetch_func_args)
            self._ql_curve_cache = self._ql_curve_cache | results

        todays_result = {}
        if todays_bdates:
            hist_cash_curve_fetch_func: callable = getattr(
                self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_ql_curve"]
            )
            hist_cash_curve_fetch_func_args_today = tuple(
                (
                    todays_bdates
                    if arg == self._CASH_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING INTRADAY CASH CURVE..."
                        if arg == self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else 1 if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS else arg
                    )
                )
                for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_ql_curve_args"]
            )
            todays_result = hist_cash_curve_fetch_func(*hist_cash_curve_fetch_func_args_today)

        final_result = {}
        for bday in non_today_bdates:
            if bday in self._ql_curve_cache:
                final_result[bday] = self._ql_curve_cache[bday]
        final_result.update(todays_result)
        return final_result

    def _fetch_scipy_cash_splines(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, interp1d]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._ql_calendar) if (start_date and end_date) else bdates
        if not end_date and bdates:
            end_date = max(bdates)

        today = datetime.today().date()
        non_today_bdates = [bday for bday in input_bdates if bday.date() != today]
        todays_bdates = [bday for bday in input_bdates if bday.date() == today]

        if not refresh_cache:
            non_today_bdates_not_cached = [bday for bday in non_today_bdates if bday not in self._scipy_curve_cache]
        else:
            non_today_bdates_not_cached = non_today_bdates

        if non_today_bdates_not_cached:
            hist_cash_curve_fetch_func: callable = getattr(
                self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_scipy_spline"]
            )
            hist_cash_curve_fetch_func_args = tuple(
                (
                    non_today_bdates_not_cached
                    if arg == self._CASH_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING HISTORICAL CASH CURVE..."
                        if arg == self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else (
                            self._MAX_NJOBS
                            if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) > 1
                            else 1 if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) == 1 else arg
                        )
                    )
                )
                for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_scipy_spline_args"]
            )
            results = hist_cash_curve_fetch_func(*hist_cash_curve_fetch_func_args)
            self._scipy_curve_cache = self._scipy_curve_cache | results

        todays_result = {}
        if todays_bdates:
            hist_cash_curve_fetch_func: callable = getattr(
                self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_scipy_spline"]
            )
            hist_cash_curve_fetch_func_args_today = tuple(
                (
                    todays_bdates
                    if arg == self._CASH_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "FITTING INTRADAY SPLINE..."
                        if arg == self._CASH_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else 1 if arg == self._CASH_CURVE_BOOTSTRAPPER_NJOBS else arg
                    )
                )
                for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_scipy_spline_args"]
            )
            todays_result = hist_cash_curve_fetch_func(*hist_cash_curve_fetch_func_args_today)

        final_result = {}
        for bday in non_today_bdates:
            if bday in self._scipy_curve_cache:
                final_result[bday] = self._scipy_curve_cache[bday]
        final_result.update(todays_result)
        return final_result

    def _extract_swap_tenors(self, s: str):
        pattern = r"(?:[0-9.]+\*)?((?:\d+[YMD]){1,2})x((?:\d+[YMD]){1,2})"
        matches = re.findall(pattern, s)
        fwd_tenors = {match[0] for match in matches}
        underlying_tenors = {match[1] for match in matches}
        return fwd_tenors, underlying_tenors

    def _build_fwd_par_term_structure_grid_timeseries(
        self,
        ql_curves_ts_dict: Dict[datetime, ql.YieldTermStructure],
        fwd_tenors: List[str],
        underlying_tenors: Optional[List[str]] = None,
        tenor_col: Optional[str] = "Tenor",
    ) -> Dict[datetime, pd.DataFrame]:
        if underlying_tenors is None:
            underlying_tenors = ["1M", "3M", "4M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

        result = {}
        for dt, ql_curve in ql_curves_ts_dict.items():
            ql_curve.enableExtrapolation()
            grid_data = {tenor_col: underlying_tenors}
            for fwd in fwd_tenors:
                col_rates = []
                fwd_start_date = self._ql_calendar.advance(datetime_to_ql_date(dt), ql.Period(fwd))
                for underlying in underlying_tenors:
                    try:
                        fwd_end_date = self._ql_calendar.advance(fwd_start_date, ql.Period(underlying))
                        par_rate = compute_par_yield(
                            ql_curve,
                            maturity_date=ql_date_to_datetime(fwd_end_date),
                            frequency=self._ql_compounding_freq,
                            day_counter=self._ql_day_count,
                            calendar=self._ql_calendar,
                            implied_forward_date=ql_date_to_datetime(fwd_start_date) if fwd != "0D" else None,
                        )
                        col_rates.append(par_rate * 100)
                    except Exception as e:
                        # TODO handle errors
                        col_rates.append(None)

                grid_data[fwd] = col_rates

            df = pd.DataFrame(grid_data)
            df.set_index(tenor_col, inplace=True)
            result[dt] = df.T

        return result

    def par_timeseries_builder(self, start_date: datetime, end_date: datetime, cols: List[str | Tuple[str, str]]) -> pd.DataFrame:
        if not self._par_curve_set_filter_func:
            raise ValueError("No Par Curve Filter Func Defined")

        extracted_fwd_tenors = set()
        extracted_underlying_tenors = set()
        for col in cols:
            col_str = col[0] if isinstance(col, tuple) else col
            curr_ex_fwds, curr_ex_unds = self._extract_swap_tenors(s=col_str)
            extracted_fwd_tenors.update(curr_ex_fwds)
            extracted_underlying_tenors.update(curr_ex_unds)

        ql_discount_curves_dict = self._fetch_ql_cash_curves(start_date=start_date, end_date=end_date)
        grid_dict = self._build_fwd_par_term_structure_grid_timeseries(
            ql_curves_ts_dict=ql_discount_curves_dict, fwd_tenors=list(extracted_fwd_tenors), underlying_tenors=list(extracted_underlying_tenors)
        )

        data_flat = {}
        for dt, df in grid_dict.items():
            row = {}
            for fwd in df.index:
                for und in df.columns:
                    col_name = f"{self._DEFAULT_PAR_PREFIX}_{fwd}x{und}"
                    row[col_name] = df.loc[fwd, und]
            data_flat[dt] = row

        ts_df = pd.DataFrame.from_dict(data_flat, orient="index")
        ts_df.index.name = "Date"

        cols_to_eval_and_return = []
        for col in cols:
            try:
                if isinstance(col, tuple):
                    ts_df[col[1]] = ts_df.eval(col[0])
                    cols_to_eval_and_return.append(col[1])
                else:
                    ts_df[col] = ts_df.eval(col)
                    cols_to_eval_and_return.append(col)

            except Exception as e:
                self._logger.error(f"'par_timeseries_builder' eval failed: {e}")

        return ts_df[cols_to_eval_and_return].sort_index()

    def par_term_structure_plotter(
        self,
        dates: List[datetime],
        use_plotly: Optional[bool] = False,
        fwd_tenors: Optional[List[str]] = ["0D"],
        underlying_tenors: Optional[List[str]] = None,
    ):
        if not self._par_curve_set_filter_func:
            raise ValueError("No Par Curve Filter Func Defined")

        cash_term_structure_dict_df = self._build_fwd_par_term_structure_grid_timeseries(
            ql_curves_ts_dict=self._fetch_ql_cash_curves(bdates=dates), fwd_tenors=fwd_tenors, underlying_tenors=underlying_tenors
        )
        self._term_structure_plotter(
            term_structure_dict_df=cash_term_structure_dict_df,
            plot_title=f"{self._curve} Par Curve",
            x_axis_col_sorter_func=lambda x: ql_period_to_months(ql.Period(x)),
            x_axis_title="Term",
            y_axis_title="Par Rate (%)",
            use_plotly=use_plotly,
        )

    def _timeseries_func_wrapper(
        self, start_date: datetime, end_date: datetime, cusips: Optional[str] = None, cols_to_return: Optional[List[str]] = None
    ) -> pd.DataFrame:
        hist_cash_curve_timeseries_func: callable = getattr(
            self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_timeseries_df"]
        )

        hist_cash_curve_timeseries_func_args = []
        for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_timeseries_df_args"]:
            if isinstance(arg, pd.DataFrame):
                hist_cash_curve_timeseries_func_args.append(arg)
            elif arg == self._CASH_CURVE_TIMESERIES_FUNC_START_DATE:
                hist_cash_curve_timeseries_func_args.append(start_date)
            elif arg == self._CASH_CURVE_TIMESERIES_FUNC_END_DATE:
                hist_cash_curve_timeseries_func_args.append(end_date)
            elif arg == self._CASH_CURVE_TIMESERIES_SEARCH_PARAMS:
                if cusips is not None:
                    hist_cash_curve_timeseries_func_args.append({"cusip": cusips})
                else:
                    hist_cash_curve_timeseries_func_args.append({})
            elif arg == self._CASH_CURVE_TIMESERIES_SUBSET_COLS:
                hist_cash_curve_timeseries_func_args.append(cols_to_return)
            else:
                hist_cash_curve_timeseries_func_args.append(arg)

        return hist_cash_curve_timeseries_func(*hist_cash_curve_timeseries_func_args)

    def otr_timeseries_builder(
        self, start_date: datetime, end_date: datetime, cols: Optional[List[str] | List[Tuple[str, str]]] = None, tz: Optional[pytz.timezone] = None
    ) -> pd.DataFrame:
        if self._data_source != "ENGINE":
            raise NotImplementedError()

        hist_otr_timeseries_func: callable = getattr(
            self._cash_data_fetcher, self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_otr_timeseries_df"]
        )

        hist_otr_timeseries_func_args = []
        for arg in self.__data_sources_funcs["HIST_CASH"][self._data_source][self._curve]["fetch_otr_timeseries_df_args"]:
            if arg == self._CASH_CURVE_TIMESERIES_FUNC_START_DATE:
                hist_otr_timeseries_func_args.append(start_date)
            elif arg == self._CASH_CURVE_TIMESERIES_FUNC_END_DATE:
                hist_otr_timeseries_func_args.append(end_date)
            else:
                hist_otr_timeseries_func_args.append(arg)

        otr_ts_df: pd.DataFrame = hist_otr_timeseries_func(*hist_otr_timeseries_func_args)
        otr_ts_df.set_index(self._timestamp_col, inplace=True)

        if not cols:
            return otr_ts_df

        cols_to_return = []
        for col in cols:
            try:
                if isinstance(col, tuple):
                    otr_ts_df[col[1]] = otr_ts_df.eval(col[0])
                    cols_to_return.append(col[1])
                else:
                    otr_ts_df[col] = otr_ts_df.eval(col)
                    cols_to_return.append(col)

            except Exception as e:
                self._logger.error(f"'otr_timeseries_builder' eval failed for {col}: {e}")

        otr_ts_df = otr_ts_df[cols_to_return]
        try:
            if tz:
                otr_ts_df.index = otr_ts_df.index.tz_convert(tz)
            else:
                otr_ts_df.index = otr_ts_df.index.tz_convert(self._default_tz)
        except Exception as e:
            self._logger.error(f"'otr_timeseries_builder' failed to convert to timezone: {tz or self._default_tz}: {e}")

        return otr_ts_df[cols_to_return].sort_index()

    def _extract_cusips(self, text: str):
        pattern = r"(?<=`)[0-9A-Z]{9}(?=_)"
        cusips = re.findall(pattern, text, flags=re.IGNORECASE)
        return cusips

    def _extract_cusip_set_cols(self, text: str):
        # pattern = r"`(?:\d{6}[A-Z]{2}\d)_([\w]+)`"
        pattern = r"`[0-9A-Z]{9}_([\w]+)`"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return matches

    def cusip_timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        cols: Optional[List[str] | List[Tuple[str, str]]] = None,
        cols_to_return: Optional[List[str]] = ["eod_ytm"],
        cusip_col: Optional[str] = "cusip",
        tz: Optional[pytz.timezone] = None,
    ) -> pd.DataFrame:
        assert cusips or cols, "MUST PASS IN A LIST OF CUSIPS OR COLS TO EVAL"

        if cols and not cusips:
            cusips = []
            for col in cols:
                if isinstance(col, tuple):
                    col = col[0]
                cusips += self._extract_cusips(col)

                extracted_cols = self._extract_cusip_set_cols(col)
                if extracted_cols:
                    cols_to_return += extracted_cols

            if not cusips:
                raise ValueError("cusip extraction failed")

        ts_df = self._timeseries_func_wrapper(
            start_date=start_date, end_date=end_date, cusips=cusips, cols_to_return=[cusip_col, self._hash_col] + cols_to_return
        ).reset_index()

        ts_df[[cusip_col, "Date"]] = ts_df["hash"].str.split("_", expand=True)
        ts_df["Date"] = pd.to_datetime(ts_df["Date"], format="mixed", errors="coerce", utc=True)
        ts_df = ts_df.drop(columns="hash")
        ts_df = ts_df.set_index(["Date", cusip_col])
        df_pivot = ts_df.unstack(cusip_col)
        df_pivot.columns = [f"{cusip}_{col}" for col, cusip in df_pivot.columns]

        ts_df = df_pivot
        if not cols:
            ts_df = ts_df[(ts_df.index.date >= start_date.date()) & (ts_df.index.date <= end_date.date())]
            return ts_df

        cols_to_eval_and_return = []
        for col in cols:
            try:
                if isinstance(col, tuple):
                    ts_df[col[1]] = ts_df.eval(col[0])
                    cols_to_eval_and_return.append(col[1])
                else:
                    ts_df[col] = ts_df.eval(col)
                    cols_to_eval_and_return.append(col)

            except Exception as e:
                self._logger.error(f"'cusip_timeseries_builder' eval failed: {e}")

        ts_df = ts_df[cols_to_eval_and_return].sort_index()
        try:
            if tz:
                ts_df.index = ts_df.index.tz_convert(tz)
            else:
                ts_df.index = ts_df.index.tz_convert(self._default_tz)
        except Exception as e:
            self._logger.error(f"'cusip_timeseries_builder' failed to convert to timezone: {tz or self._default_tz}: {e}")

        return ts_df

    def cusip_curve_set_builder(
        self, start_date: datetime, end_date: datetime, filter_func: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df
    ) -> Dict[datetime, pd.DataFrame]:
        ts_df = self._timeseries_func_wrapper(start_date=start_date, end_date=end_date)

        def format_cusip_set_df(df: pd.DataFrame):
            df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce", format="mixed")
            df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce", format="mixed")
            return filter_func(df.sort_values("maturity_date"))

        def to_vanilla_pydt(dt: datetime):
            return datetime(dt.year, dt.month, dt.day)

        return {(to_vanilla_pydt(group)): format_cusip_set_df(group_df) for group, group_df in ts_df.groupby(self._timestamp_col)}

    def cusip_spline_spreads_timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: List[str],
        ytm_col: Optional[str] = "eod_ytm",
        ttm_col: Optional[str] = "time_to_maturity",
        use_cusip_cols: bool = False,
    ):
        def filter_to_cusips(df: pd.DataFrame):
            return df[df["cusip"].isin(cusips)]

        cusip_curve_sets: Dict[datetime, pd.DataFrame] = self.cusip_curve_set_builder(start_date=start_date, end_date=end_date, filter_func=filter_to_cusips)
        scipy_splines: Dict[datetime, interp1d] = self._fetch_scipy_cash_splines(start_date=start_date, end_date=end_date)

        ts = []
        for dt, cusip_curve_set_df in cusip_curve_sets.items():
            if dt not in scipy_splines:
                continue

            for _, cusip_row in cusip_curve_set_df.iterrows():
                par_curve_ytm = scipy_splines[dt](cusip_row[ttm_col])
                ts.append(
                    {
                        self._timestamp_col: dt,
                        "cusip": cusip_row["cusip"],
                        ttm_col: cusip_row[ttm_col],
                        ytm_col: cusip_row[ytm_col],
                        "Par Curve YTM": par_curve_ytm,
                        "Spline Spread bps": (par_curve_ytm - cusip_row[ytm_col]) * 100,
                    }
                )

        df = pd.DataFrame(ts).set_index(self._timestamp_col).sort_index()
        if use_cusip_cols:
            df = df.reset_index().pivot_table(
                index=self._timestamp_col, columns="cusip", values="Spline Spread bps", aggfunc="first", dropna=False  # or np.mean if you prefer
            )

        return df

    def plot_all_cusips(
        self,
        date: datetime,
        ttm_col: Optional[str] = "time_to_maturity",
        ytm_col: Optional[str] = "eod_ytm",
        hover_data: Optional[List[str]] = [
            "issue_date",
            "maturity_date",
            "cusip",
            "original_security_term",
            "ust_label",
            "eod_price",
            "free_float",
        ],
        cusips_hightlighter: Optional[List[Tuple[Annotated[str, "cusip"], Annotated[str, "color"]]]] = None,
        ust_labels_highlighter: Optional[List[Tuple[Annotated[str, "ust_label"], Annotated[str, "color"]]]] = None,
        return_spline: Optional[bool] = False,
    ):
        cusip_set_df = self.cusip_curve_set_builder(start_date=date, end_date=date)[date]
        par_curve_spline = self._fetch_scipy_cash_splines(bdates=[date])[date]

        plot_usts(
            curve_set_df=cusip_set_df,
            ttm_col=ttm_col,
            ytm_col=ytm_col,
            hover_data=hover_data,
            cusips_hightlighter=cusips_hightlighter,
            ust_labels_highlighter=ust_labels_highlighter,
            splines=[(par_curve_spline, "Par Curve")],
            title=f"US Treasury Market: All Active CUSIPS: {max(cusip_set_df["timestamp"])}",
        )

        if return_spline:
            return par_curve_spline


def compute_par_yield(
    discount_curve: ql.DiscountCurve,
    maturity_date: datetime,
    frequency=ql.Semiannual,
    day_counter=ql.ActualActual(ql.ActualActual.Bond),
    calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    implied_forward_date: Optional[datetime] = None,
):
    if implied_forward_date is not None:
        new_ref_date = datetime_to_ql_date(implied_forward_date)
        discount_curve = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(discount_curve), new_ref_date)
        discount_curve.enableExtrapolation()

    settlement_date = discount_curve.referenceDate()
    ql_maturity_date = datetime_to_ql_date(maturity_date)

    schedule = ql.Schedule(
        settlement_date,
        ql_maturity_date,
        ql.Period(frequency),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )

    pv_factor_sum = 0.0
    dates = list(schedule)
    for i in range(1, len(dates)):
        start_date = dates[i - 1]
        end_date = dates[i]
        accrual = day_counter.yearFraction(start_date, end_date)
        pv = discount_curve.discount(end_date)
        pv_factor_sum += accrual * pv

    discount_at_maturity = discount_curve.discount(ql_maturity_date)
    par_yield = (1.0 - discount_at_maturity) / pv_factor_sum
    return par_yield
