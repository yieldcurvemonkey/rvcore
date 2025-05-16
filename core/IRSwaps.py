from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from core.utils.ql_loader import ql
import tqdm
from joblib import Parallel, delayed

from core.Plotting.BaseProductPlotter import BaseProductPlotter

from core.DataFetching.CMEFetcherV2 import CMEFetcherV2

from core.CurveBuilding.IRSwaps.ql_curve_building_utils import build_ql_discount_curve, get_nodes_dict
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.CurveBuilding.IRSwaps.IRSwapCurveBootstrapper import IRSwapCurveBootstrapper

from core.TimeseriesBuilding.IRSwaps.IRSwapQuery import IRSwapQuery, IRSwapValue, IRSwapStructure, IRSwapQueryWrapper
from core.TimeseriesBuilding.IRSwaps.IRSwapStructure import IRSwapStructureFunctionMap
from core.TimeseriesBuilding.IRSwaps.IRSwapValue import IRSwapValueFunctionMap

from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, most_recent_business_day_ql, ql_date_to_datetime, ql_period_to_days


class IRSwaps(BaseProductPlotter):
    _data_source: Literal["CME", "CSV_{path}"]
    _fixings: Dict[datetime, float] = None
    _curve: str = None

    _irswaps_hist_data_fetcher: IRSwapCurveBootstrapper = None
    _irswaps_hist_timeseries_csv_df: pd.DataFrame = None

    _ql_day_count: ql.DayCounter = None
    _ql_bday_convention = None
    _ql_calendar: ql.Calendar = None
    _ql_compounded: Literal["ql.Compounding"] = None

    _ql_curve_cache: Dict[datetime, ql.YieldTermStructure] = None
    _ql_irswap_index_cache: Dict[datetime, ql.SwapIndex] = None
    _fwd_irswaps_timeseries_cache: Dict[Tuple[datetime, Tuple[str, ...], Tuple[str, ...]], Dict[str, Union[datetime, float]]] = None

    __data_sources_funcs: Dict = None

    _pre_fetch_curves: bool = False
    _PRE_FETCH_CURVE_PERIOD = ql.Period("-1Y")
    _date_col: str = None
    _bootstrap_cols: List[str] = None
    _default_tz = "US/Eastern"

    _show_tqdm: bool = None
    _proxies: Dict[str, str] = None

    _default_fwd_irswap_tenors = ["0D", "01M", "03M", "06M", "09M", "01Y", "02Y", "03Y", "05Y", "10Y"]
    _default_underlying_tenors = None # fmt: skip
    _IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES = "_INPUT_DATES"
    _IRSWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE = "_BOOTSTRAP_TDQM_MESSAGE"
    _IRSWAP_CURVE_BOOTSTRAPPER_NJOBS = "_BOOTSTRAP_NJOBS"
    _MAX_NJOBS = None

    _FETCHER_TIMEOUT = 10
    _MAX_CONNECTIONS = 64
    _MAX_KEEPALIVE_CONNECTIONS = 5

    def __init__(
        self,
        data_source: Literal["CME", "CSV_{path}"],
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
        fixings: Optional[Dict[datetime, float]] = None,
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
        self._ql_irswap_index_cache = {}
        self._fwd_irswaps_timeseries_cache = {}
        self.__data_sources_funcs = {}
        self._date_col = date_col

        if "CSV_" in data_source:
            csv_df = pd.read_csv(data_source.split("_", 1)[-1])
            csv_df[self._date_col] = pd.to_datetime(csv_df[self._date_col], errors="coerce", format="mixed")
            self._irswaps_hist_timeseries_csv_df = csv_df.set_index(self._date_col)
            self._data_source = "CSV"
        else:
            self._data_source = data_source

        self._curve = curve
        self._fixings = fixings
        self._ql_day_count = CME_IRSWAP_CURVE_QL_PARAMS[self._curve]["dayCounter"]
        self._ql_calendar = CME_IRSWAP_CURVE_QL_PARAMS[self._curve]["calendar"]
        self._ql_bday_convention = CME_IRSWAP_CURVE_QL_PARAMS[self._curve]["businessConvention"]
        self._default_underlying_tenors = CME_IRSWAP_CURVE_QL_PARAMS[self._curve]["default_tenors"]
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
            "HIST_IRSWAPS": {
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
                        self._IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_interpolation_algo,
                        True,
                        self._show_tqdm,
                        self._MAX_CONNECTIONS,
                        self._MAX_KEEPALIVE_CONNECTIONS,
                    ),
                },
                "CSV": {
                    "obj": IRSwapCurveBootstrapper,
                    "init_args": (self._irswaps_hist_timeseries_csv_df,),
                    "fetch_func": "parallel_ql_irswap_curve_bootstrapper",
                    "fetch_func_args": (
                        self._curve,
                        None,
                        None,
                        self._IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES,
                        self._ql_interpolation_algo,
                        True,
                        self._show_tqdm,
                        self._IRSWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE,
                        self._IRSWAP_CURVE_BOOTSTRAPPER_NJOBS,
                    ),
                },
            }
        }

        self._irswaps_hist_data_fetcher = self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["obj"](
            *self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["init_args"]
        )

        if self._pre_fetch_curves:
            pre_fetch_end_date = most_recent_business_day_ql(ql_calendar=self._ql_calendar, tz=self._default_tz, to_pydate=True)
            pre_fetch_start_date = ql_date_to_datetime(self._ql_calendar.advance(datetime_to_ql_date(pre_fetch_end_date), self._PRE_FETCH_CURVE_PERIOD))
            self.fetch_ql_irswap_curves(start_date=pre_fetch_start_date, end_date=pre_fetch_end_date)

    def fetch_irswap_index(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        to_pydt: Optional[bool] = False,
    ) -> Dict[datetime, ql.SwapIndex]:
        curves = self.fetch_ql_irswap_curves(
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            to_pydt=to_pydt,
        )

        idx_cls = CME_IRSWAP_CURVE_QL_PARAMS[self._curve]["swapIndex"]
        out: Dict[datetime, ql.SwapIndex] = {}

        for dt, curve in curves.items():
            if dt not in self._ql_irswap_index_cache:
                handle = ql.RelinkableYieldTermStructureHandle()
                idx = idx_cls(handle)

                for d_fix, rate in self._fixings.items():
                    if dt > d_fix or rate == None or np.isnan(rate):
                        continue
                    idx.addFixing(datetime_to_ql_date(d_fix), rate)
                self._ql_irswap_index_cache[dt] = (idx, handle)

            idx, handle = self._ql_irswap_index_cache[dt]
            handle.linkTo(curve)

            out[dt] = idx

        return out

    def get_latest_cached_curve(self):
        d = self._ql_curve_cache
        k, last_value = _, d[k] = d.popitem()
        return k, last_value

    def fetch_latest_curve(self):
        hist_irswaps_curve_fetch_func: callable = getattr(
            self._irswaps_hist_data_fetcher, self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func"]
        )
        hist_irswaps_curve_fetch_func_args_today = tuple(
            (
                [most_recent_business_day_ql(self._ql_calendar, self._default_tz, to_pydate=True)]
                if arg == self._IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES
                else (
                    "BOOTSTRAPPING INTRADAY IRSWAPS CURVE..."
                    if arg == self._IRSWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                    else 1 if arg == self._IRSWAP_CURVE_BOOTSTRAPPER_NJOBS else arg
                )
            )
            for arg in self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func_args"]
        )
        todays_result: Dict[pd.Timestamp, ql.DiscountCurve] = hist_irswaps_curve_fetch_func(*hist_irswaps_curve_fetch_func_args_today)

        return todays_result.popitem()

    def fetch_ql_irswap_curves(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
        to_pydt: Optional[bool] = False,
    ) -> Dict[datetime, ql.DiscountCurve]:
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
            hist_irswaps_curve_fetch_func: callable = getattr(
                self._irswaps_hist_data_fetcher, self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func"]
            )
            hist_irswaps_curve_fetch_func_args = tuple(
                (
                    non_today_bdates_not_cached
                    if arg == self._IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING HISTORICAL IRSWAPS CURVE..."
                        if arg == self._IRSWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else (
                            self._MAX_NJOBS
                            if arg == self._IRSWAP_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) > 1
                            else 1 if arg == self._IRSWAP_CURVE_BOOTSTRAPPER_NJOBS and len(non_today_bdates_not_cached) == 1 else arg
                        )
                    )
                )
                for arg in self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func_args"]
            )
            results = hist_irswaps_curve_fetch_func(*hist_irswaps_curve_fetch_func_args)
            self._ql_curve_cache = self._ql_curve_cache | results

        todays_result: Dict[pd.Timestamp, ql.DiscountCurve] = {}
        if todays_bdates:
            hist_irswaps_curve_fetch_func: callable = getattr(
                self._irswaps_hist_data_fetcher, self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func"]
            )
            hist_irswaps_curve_fetch_func_args_today = tuple(
                (
                    todays_bdates
                    if arg == self._IRSWAP_CURVE_FETCH_FUNC_INPUT_DATES
                    else (
                        "BOOTSTRAPPING INTRADAY IRSWAPS CURVE..."
                        if arg == self._IRSWAP_CURVE_BOOTSTRAPPING_TQDM_MESSAGE
                        else 1 if arg == self._IRSWAP_CURVE_BOOTSTRAPPER_NJOBS else arg
                    )
                )
                for arg in self.__data_sources_funcs["HIST_IRSWAPS"][self._data_source]["fetch_func_args"]
            )
            todays_result = hist_irswaps_curve_fetch_func(*hist_irswaps_curve_fetch_func_args_today)

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

    def _build_fwd_irswaps_timeseries(
        self,
        ql_curves_ts_dict: Dict[datetime, ql.DiscountCurve],
        queries: List[IRSwapQuery | List[IRSwapQuery] | IRSwapQueryWrapper],
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        flat_queries: List[IRSwapQuery] = []
        for q in queries:
            if isinstance(q, Tuple):
                q = q[0]
            if isinstance(q, list):
                flat_queries.extend(q)
            elif isinstance(q, IRSwapQueryWrapper):
                flat_queries.extend(q.return_query())
            else:
                flat_queries.append(q)

        tasks, cached_rows = [], []
        for ref_date, ql_curve in ql_curves_ts_dict.items():
            ql_curve_nodes = get_nodes_dict(ql_curve)
            for query in flat_queries:
                q_key = _query_to_key(query)
                cache_key = (ref_date, q_key)
                if cache_key in self._fwd_irswaps_timeseries_cache:
                    cached_rows.append(self._fwd_irswaps_timeseries_cache[cache_key])
                else:
                    tasks.append((cache_key, ref_date, ql_curve_nodes, query))

        tasks_iter = tqdm.tqdm(tasks, desc="PRICING IRSWAPS...") if self._show_tqdm else tasks
        new_results = Parallel(n_jobs=n_jobs, timeout=999999 if n_jobs != 1 else None)(
            delayed(_process_irswap_single_query_with_cache)(
                cache_key,
                ref_date,
                ql_curve_nodes,
                self._curve,
                self._ql_interpolation_algo,
                query,
                self._fixings,
            )
            for cache_key, ref_date, ql_curve_nodes, query in tasks_iter
        )

        for ck, row in new_results:
            self._fwd_irswaps_timeseries_cache[ck] = row

        all_rows = cached_rows + [row for _, row in new_results]
        if not all_rows:
            return pd.DataFrame([], index=pd.DatetimeIndex([]))

        rows_df = pd.DataFrame(all_rows, columns=[self._date_col, "col", "val"])
        rows_df = rows_df.drop_duplicates(subset=[self._date_col, "col"], keep="last")
        df = rows_df.pivot(index=self._date_col, columns="col", values="val").sort_index()
        df.columns.name = None
        try:
            df.index = df.index.tz_convert(self._default_tz)
        except Exception as e:
            self._logger.error(f"'timeseries_builder' failed to convert to {self._default_tz}: {e}")

        return df

    def irswaps_timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        queries: List[IRSwapQuery | List[IRSwapQuery] | IRSwapQueryWrapper | Tuple[IRSwapQuery | List[IRSwapQuery] | IRSwapQueryWrapper, str]],
        return_all: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ) -> pd.DataFrame:

        ts_df = self._build_fwd_irswaps_timeseries(
            ql_curves_ts_dict=self.fetch_ql_irswap_curves(start_date=start_date, end_date=end_date),
            queries=queries,
            n_jobs=n_jobs,
        )
        if ts_df.empty:
            raise ValueError("'irswaps_timeseries_builder': Dataframe is empty")

        cols_to_return = []
        if return_all:
            cols_to_return = list(ts_df.columns)

        for q in queries:
            priority_col_name = None
            if isinstance(q, Tuple):
                priority_col_name = q[1]
                q = q[0]

            if isinstance(q, List):
                col_name, eval_str = _pretty_package_query_col_name_and_eval_str(self._curve, q)
            else:
                col_name = q.col_name(self._curve)
                eval_str = q.eval_expression(self._curve)

            try:
                ts_df[priority_col_name or col_name] = ts_df.eval(eval_str)
                cols_to_return.append(priority_col_name or col_name)
            except Exception as e:
                self._logger.error(f"'timeseries_builder' eval failed for {col_name}: {e}")

        cols_to_return = list(set(cols_to_return))
        ts_df = ts_df[cols_to_return].sort_index()
        try:
            ts_df.index = ts_df.index.tz_convert(self._default_tz)
        except Exception as e:
            self._logger.error(f"'timeseries_builder' failed to convert to {self._default_tz}: {e}")
        return ts_df

    def _build_fwd_irswaps_term_structure_grid_timeseries(
        self,
        ql_curves_ts_dict: Dict[datetime, ql.YieldTermStructure],
        fwd_tenors: List[str],
        underlying_tenors: List[str],
        n_jobs: Optional[int] = 1,
        irswap_value: Optional[IRSwapValue] = IRSwapValue.RATE,
    ) -> Dict[datetime, pd.DataFrame]:
        queries: List[IRSwapQuery] = []
        for fwd in fwd_tenors:
            for u in underlying_tenors:
                queries.append(IRSwapQuery(tenor=f"{fwd}x{u}", value=irswap_value))

        irswaps_iter = tqdm.tqdm(ql_curves_ts_dict.items(), desc="PRICING IRSWAPS...") if self._show_tqdm else ql_curves_ts_dict.items()
        results: Dict[pd.Timestamp | datetime, pd.DataFrame] = Parallel(n_jobs=n_jobs)(
            delayed(_process_date_grid)(
                curr_date,
                get_nodes_dict(ql_curve),
                self._curve,
                self._ql_interpolation_algo,
                queries,
            )
            for curr_date, ql_curve in irswaps_iter
        )
        fwd_term_structure_grids = {curr_date: df for curr_date, df in results}
        return fwd_term_structure_grids

    def irswaps_data_grid_builder(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        fwd_tenors: Optional[List[str]] = None,
        irswap_tenors: Optional[List[str]] = None,
        irswap_value: Optional[IRSwapValue] = IRSwapValue.RATE,
        to_pydt: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
    ):
        fwd_tenors = self._default_fwd_irswap_tenors if not fwd_tenors else fwd_tenors
        irswap_tenors = self._default_underlying_tenors if not irswap_tenors else irswap_tenors
        return self._build_fwd_irswaps_term_structure_grid_timeseries(
            ql_curves_ts_dict=self.fetch_ql_irswap_curves(start_date=start_date, end_date=end_date, bdates=bdates, to_pydt=to_pydt),
            fwd_tenors=fwd_tenors,
            underlying_tenors=irswap_tenors,
            irswap_value=irswap_value,
            n_jobs=n_jobs,
        )

    def irswaps_term_structure_plotter(
        self,
        bdates: List[datetime],
        use_plotly: Optional[bool] = False,
        fwd_tenors: Optional[List[str]] = ["0D"],
        irswap_tenors: Optional[List[str]] = None,
        n_jobs: Optional[int] = 1,
        return_data: Optional[bool] = False,
    ):
        irswap_tenors = self._default_underlying_tenors if not irswap_tenors else irswap_tenors
        irswaps_term_structure_dict_df = self._build_fwd_irswaps_term_structure_grid_timeseries(
            ql_curves_ts_dict=self.fetch_ql_irswap_curves(bdates=bdates), fwd_tenors=fwd_tenors, underlying_tenors=irswap_tenors, n_jobs=n_jobs
        )
        self._term_structure_plotter(
            term_structure_dict_df=irswaps_term_structure_dict_df,
            plot_title=f"{self._curve} Curve",
            x_axis_col_sorter_func=lambda x: ql_period_to_days(ql.Period(x)),
            x_axis_title="Term",
            y_axis_title="Rate",
            use_plotly=use_plotly,
        )
        if return_data:
            return irswaps_term_structure_dict_df


def _process_irswap_single_query_with_cache(
    cache_key: Tuple[datetime, Tuple],
    *args,
    **kwargs,
) -> Tuple[Tuple[datetime, Tuple], Tuple[datetime, str, float]]:
    return cache_key, _process_irswap_single_query(*args, **kwargs)


def _process_irswap_single_query(
    ref_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    curve: str,
    ql_interpolation_algo_str: str,
    query: IRSwapQuery,
    fixings: Optional[Dict[datetime, float]] = None,
) -> Tuple[datetime, str, float]:
    ql.Settings.instance().evaluationDate = datetime_to_ql_date(ref_date)

    ql_curve = build_ql_discount_curve(
        datetime_series=pd.Series(ql_curve_nodes.keys()),
        discount_factor_series=pd.Series(ql_curve_nodes.values()),
        ql_dc=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
        ql_cal=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
        interpolation_algo=f"df_{ql_interpolation_algo_str}",
    )
    ql_curve.enableExtrapolation()
    curve_handle = ql.YieldTermStructureHandle(ql_curve)

    irswap_index = None
    if fixings:
        irswap_index = CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle)
        for d, f in fixings.items():
            irswap_index.addFixing(datetime_to_ql_date(d), f)

    pkg, rw = IRSwapStructureFunctionMap(curve=curve, curve_handle=curve_handle, swap_index=irswap_index).apply(
        tenor=query.tenor,
        effective_date=query.effective_date,
        maturity_date=query.maturity_date,
        value=query.value,
        structure=query.structure,
        **query.structure_kwargs,
    )
    val = IRSwapValueFunctionMap(package=pkg, risk_weights=rw, curve=curve, curve_handle=curve_handle).apply(query.value)
    return ref_date, query.col_name(curve), val


def _process_irswap_queries(
    ref_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    curve: str,
    ql_interpolation_algo_str: str,
    irswap_queries: List[IRSwapQuery],
    fixings: Optional[Dict[datetime, float]] = None,
    date_col: Optional[str] = "Date",
):
    ql.Settings.instance().evaluationDate = datetime_to_ql_date(ref_date)

    row = {date_col: ref_date}
    ql_curve = build_ql_discount_curve(
        datetime_series=pd.Series(ql_curve_nodes.keys()),
        discount_factor_series=pd.Series(ql_curve_nodes.values()),
        ql_dc=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
        ql_cal=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
        interpolation_algo=f"df_{ql_interpolation_algo_str}",
    )
    ql_curve.enableExtrapolation()
    curve_handle = ql.YieldTermStructureHandle(ql_curve)

    irswap_index = None
    if fixings:
        irswap_index: ql.SwapIndex = CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle)
        for d, f in fixings.items():
            irswap_index.addFixing(fixingDate=datetime_to_ql_date(d), fixing=f)

    ss_func_map = IRSwapStructureFunctionMap(curve=curve, curve_handle=curve_handle, swap_index=irswap_index)
    for q in irswap_queries:
        package, risk_weights = ss_func_map.apply(
            tenor=q.tenor, effective_date=q.effective_date, maturity_date=q.maturity_date, value=q.value, structure=q.structure, **q.structure_kwargs
        )
        row[q.col_name(curve)] = IRSwapValueFunctionMap(package=package, risk_weights=risk_weights, curve=curve, curve_handle=curve_handle).apply(q.value)
    return row


def _process_date_grid(
    ref_date: datetime,
    ql_curve_nodes: Dict[datetime, float],
    curve: str,
    ql_interpolation_algo_str: str,
    irswap_queries: List[IRSwapQuery],
) -> Tuple[datetime, pd.DataFrame]:
    row = _process_irswap_queries(
        ref_date=ref_date,
        ql_curve_nodes=ql_curve_nodes,
        curve=curve,
        ql_interpolation_algo_str=ql_interpolation_algo_str,
        irswap_queries=irswap_queries,
    )

    records = []
    for q in irswap_queries:
        fwd, tenor = q.tenor.split("x")
        col = q.col_name(curve)
        val = row.get(col, float("nan"))
        records.append({"forward": fwd, "tenor": tenor, "value": val})

    df = pd.DataFrame(records).pivot(index="tenor", columns="forward", values="value")
    df = df.loc[sorted(df.index, key=ql.Period)].T
    df = df.loc[sorted(df.index, key=ql.Period)]
    return ref_date, df


def _query_to_key(q: IRSwapQuery) -> tuple:
    tenor = str(q.tenor)
    eff = q.effective_date.isoformat() if q.effective_date else None
    mat = q.maturity_date.isoformat() if q.maturity_date else None
    val = tuple(q.value) if isinstance(q.value, list) else q.value
    struct = q.structure.name
    kwargs = tuple(sorted(q.structure_kwargs.items()))
    return (tenor, eff, mat, val, struct, kwargs, q.name, q.risk_weight)


def _pretty_package_query_col_name_and_eval_str(curve: str, queries: List[IRSwapQuery]) -> Tuple[str, str]:
    if not queries:
        return ""

    structs = {q.structure for q in queries}
    vals = {q.value for q in queries}
    if len(structs) > 1 or len(vals) > 1:
        raise ValueError("All queries must have the same structure & value")
    struct = structs.pop()
    val = vals.pop()

    terms = []
    eval = ""
    for q in queries:
        name = str(q.tenor) if q.tenor else f"{q.effective_date.date()}-{q.maturity_date.date()}"
        weight = q.risk_weight if q.risk_weight is not None else 1.0
        terms.append((weight, name))
        eval += f"{q.eval_expression(curve)} + "

    parts = []
    for i, (w, name) in enumerate(terms):
        if i == 0:
            parts.append(f"{w:.1f} * {name}")
        else:
            sign = "-" if w < 0 else "+"
            parts.append(f" {sign} {abs(w):.1f} * {name}")

    body = "".join(parts)
    return f"{body} {struct.name} {val.name}", eval[:-3]
