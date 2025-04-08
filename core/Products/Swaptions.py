import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import QuantLib as ql
import tqdm
import ujson as json
from joblib import Parallel, delayed
from sqlalchemy import Engine

from core.Fetchers.GithubFetcher import GithubFetcher
from core.Products.BaseProductPlotter import BaseProductPlotter
from core.Products.CurveBuilding.ql_curve_building_utils import build_ql_discount_curve
from core.Products.CurveBuilding.ql_curve_params import CME_SWAP_CURVE_QL_PARAMS
from core.Products.CurveBuilding.Swaptions.SwaptionCubeBuilder import SwaptionCubeBuilder
from core.Products.CurveBuilding.Swaptions.AlchemySwaptionCubeBuilderWrapper import AlchemySwaptionCubeBuilderWrapper
from core.Products.CurveBuilding.Swaptions.types import SABRParams, SCube
from core.Products.Swaps import Swaps
from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, ql_date_to_datetime, ql_period_to_months


@dataclass(frozen=True)
class SwaptionNvolQuery:
    expiry: str
    tail: str
    atmf_offset_bps: int
    _DEFAULT_SWAPTION_PREFIX: str = field(default="NVOL", init=False, repr=False, compare=False)

    def col_name(self) -> str:
        return f"{self._DEFAULT_SWAPTION_PREFIX}_{self.expiry}x{self.tail}_ATMF{self.atmf_offset_bps:+d}"


class Swaptions(BaseProductPlotter):
    _data_source: Literal["GITHUB", "ENGINE", "JSON_{path}"] = None
    _swaptions_db_engine: Engine = None
    _swaps_product: Swaps = None
    _curve: str = None

    _use_sabr: bool = None
    _sabr_params_key: str = None
    _precalibrate_cubes: bool = None
    _n_jobs: int = None
    _show_tqdm: bool = None
    _proxies: Dict[str, str] = None

    _EXPIRY_COL: str = "Expiry"
    _TAIL_COL: str = "Tail"

    _qlcube_cache: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube] = None
    _scube_cache: Dict[datetime, SCube] = None
    _sabr_params_cache: Dict[datetime, SABRParams] = None
    _nvol_timeseries_cache: Dict[Tuple[datetime, Tuple[str, ...], Tuple[str, ...]], Dict[str, Union[datetime, float]]] = None
    __data_sources_funcs: Dict = {}

    _swaption_hist_data_fetcher: GithubFetcher = None

    _DEFAULT_SWAPTION_PREFIX = "NVOL"
    _SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES = "_INPUT_DATES"
    _SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES = "_INPUT_QL_DISC_CURVES"

    def __init__(
        self,
        data_source: Literal["GITHUB", "JSON_{path}"] | Engine,
        swaps_product: Swaps,
        use_sabr: Optional[bool] = False,
        sabr_params_key: Optional[str] = "sabr_params",
        precalibrate_cubes: Optional[bool] = False,
        n_jobs: Optional[int] = 1,
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

        self._qlcube_cache = {}
        self._scube_cache = {}
        self._sabr_params_cache = {}
        self._nvol_timeseries_cache = {}
        self._sabr_params_key = sabr_params_key

        if not isinstance(data_source, str):
            self._swaptions_db_engine = data_source
            self._data_source = "ENGINE"
        else:
            if data_source in ["JPM", "SKY"]:
                raise NotImplementedError(f"data source not implemented")

            if "JSON_" in data_source:
                with open(data_source.split("_", 1)[-1], "r") as file:
                    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = json.load(file)
                self._scube_cache = {
                    datetime.strptime(date_str, "%Y-%m-%d"): {
                        int(inner_key): pd.DataFrame.from_records(records).set_index(self._EXPIRY_COL)
                        for inner_key, records in inner_dict.items()
                        if inner_key != self._sabr_params_key
                    }
                    for date_str, inner_dict in data.items()
                }
                self._sabr_params_cache = {
                    datetime.strptime(date_str, "%Y-%m-%d"): inner_dict[self._sabr_params_key]
                    for date_str, inner_dict in data.items()
                    if self._sabr_params_key in inner_dict
                }
                self._data_source = "JSON"

            else:
                self._data_source = data_source

        self._swaps_product = swaps_product
        self._curve = self._swaps_product._curve
        self._use_sabr = use_sabr
        self._precalibrate_cubes = precalibrate_cubes
        self._n_jobs = n_jobs

        self._show_tqdm = show_tqdm
        self._proxies = proxies

        self.__data_sources_funcs = {
            "HIST_SWAPTIONS": {
                "GITHUB": {
                    "obj": GithubFetcher,
                    "init_args": (
                        self._swaps_product._FETCHER_TIMEOUT,
                        self._proxies,
                        self._debug_verbose,
                        self._info_verbose,
                        self._warning_verbose,
                        self._error_verbose,
                    ),
                    "fetch_qlcubes": "fetch_USD_MONKEY_CUBE_ql_vol_cubes",
                    "fetch_qlcubes_args": (
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES,
                        self._use_sabr,
                        self._precalibrate_cubes,
                        self._n_jobs,
                        self._show_tqdm,
                        self._swaps_product._MAX_CONNECTIONS,
                        self._swaps_product._MAX_KEEPALIVE_CONNECTIONS,
                    ),
                    "fetch_scubes": "fetch_USD_MONKEY_CUBE_dict_df",
                    "fetch_scubes_args": (
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._show_tqdm,
                        self._swaps_product._MAX_CONNECTIONS,
                        self._swaps_product._MAX_KEEPALIVE_CONNECTIONS,
                    ),
                    "fetch_sabr_params": "fetch_USD_MONKEY_CUBE_sabr_params",
                    "fetch_sabr_params_args": (
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._show_tqdm,
                        self._swaps_product._MAX_CONNECTIONS,
                        self._swaps_product._MAX_KEEPALIVE_CONNECTIONS,
                    ),
                },
                "JSON": {
                    "obj": SwaptionCubeBuilder,
                    "init_args": (self._scube_cache, self._sabr_params_cache),
                    "fetch_qlcubes": "parallel_ql_vol_cube_builder",
                    "fetch_qlcubes_args": (
                        self._curve,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._use_sabr,
                        self._precalibrate_cubes,
                        True,  # extrapolation
                        self._show_tqdm,
                        None,
                        self._n_jobs,
                    ),
                    "fetch_scubes": "return_scube",
                    "fetch_scubes_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                    ),
                    "fetch_sabr_params": "return_sabr_params",
                    "fetch_sabr_params_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                    ),
                },
                "ENGINE": {
                    "obj":  AlchemySwaptionCubeBuilderWrapper,
                    "init_args": (self._swaptions_db_engine, None, None, None, self._EXPIRY_COL),
                    "fetch_qlcubes": "ql_vol_cube_builder_wrapper",
                    "fetch_qlcubes_args": (
                        self._curve,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._use_sabr,
                        self._precalibrate_cubes,
                        True,  # extrapolation
                        self._show_tqdm,
                        None,
                        self._n_jobs,
                    ),
                    "fetch_scubes": "return_scube",
                    "fetch_scubes_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                    ),
                    "fetch_sabr_params": "return_sabr_params",
                    "fetch_sabr_params_args": (
                        self._curve,
                        None,
                        None,
                        self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                    ),
                }
            }
        }
        self._swaption_hist_data_fetcher = self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["obj"](
            *self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["init_args"]
        )

    def _fetch_qlcubes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
        assert (start_date and end_date) or bdates, "MUST PASS IN start and end dates or a list of bdates"

        ql_discount_curves_dict = self._swaps_product._fetch_ql_swap_curves(start_date=start_date, end_date=end_date, bdates=bdates, to_pydt=True)
        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._swaps_product._ql_calendar) if start_date and end_date else bdates

        if not refresh_cache:
            bdates_not_cached = [bday for bday in input_bdates if bday not in self._qlcube_cache]
        else:
            bdates_not_cached = input_bdates

        if len(bdates_not_cached) > 0:
            hist_vol_cube_fetch_func: callable = getattr(
                self._swaption_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_qlcubes"]
            )
            hist_vol_cube_fetch_func_args = tuple(
                (
                    bdates_not_cached
                    if arg == self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES
                    else ql_discount_curves_dict if arg == self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES else arg
                )
                for arg in self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_qlcubes_args"]
            )
            ql_vol_cube_fetch_results = hist_vol_cube_fetch_func(*hist_vol_cube_fetch_func_args)
            self._qlcube_cache = self._qlcube_cache | ql_vol_cube_fetch_results

        return {k: self._qlcube_cache[k] for k in input_bdates if k in self._qlcube_cache}

    def _fetch_scubes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, SCube]:
        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._swaps_product._ql_calendar) if start_date and end_date else bdates
        if not refresh_cache:
            bdates_not_cached = [bday for bday in input_bdates if bday not in self._scube_cache]
        else:
            bdates_not_cached = input_bdates

        if len(bdates_not_cached) > 0:
            hist_vol_cube_fetch_func: callable = getattr(
                self._swaption_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_scubes"]
            )
            hist_vol_cube_fetch_func_args = tuple(
                (bdates_not_cached if arg == self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES else arg)
                for arg in self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_scubes_args"]
            )
            dict_df_vol_cube = hist_vol_cube_fetch_func(*hist_vol_cube_fetch_func_args)
            self._scube_cache = self._scube_cache | dict_df_vol_cube

        return {k: self._scube_cache[k] for k in input_bdates if k in self._scube_cache}

    def _fetch_sabr_params(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, SABRParams]:
        input_bdates = get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._swaps_product._ql_calendar) if start_date and end_date else bdates
        if not refresh_cache:
            bdates_not_cached = [bday for bday in input_bdates if bday not in self._sabr_params_cache]
        else:
            bdates_not_cached = input_bdates

        if len(bdates_not_cached) > 0:
            hist_vol_cube_fetch_func: callable = getattr(
                self._swaption_hist_data_fetcher, self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_sabr_params"]
            )
            hist_vol_cube_fetch_func_args = tuple(
                (bdates_not_cached if arg == self._SWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES else arg)
                for arg in self.__data_sources_funcs["HIST_SWAPTIONS"][self._data_source]["fetch_sabr_params_args"]
            )
            dict_df_vol_cube = hist_vol_cube_fetch_func(*hist_vol_cube_fetch_func_args)
            self._sabr_params_cache = self._sabr_params_cache | dict_df_vol_cube

        return {k: self._sabr_params_cache[k] for k in input_bdates if k in self._sabr_params_cache}

    def _build_nvol_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        nvol_queries: List[SwaptionNvolQuery],
        n_jobs: Optional[int] = 1,
    ):
        vol_cube_ts_dict_df = self._fetch_scubes(start_date=start_date, end_date=end_date)
        ql_discount_curves_dict = self._swaps_product._fetch_ql_swap_curves(start_date=start_date, end_date=end_date)
        discount_curve_nodes_dict = {
            dt: dict(
                zip(
                    [ql_date_to_datetime(ql_date) for ql_date in ql_curve.dates()],
                    [d for d in ql_curve.discounts()],
                )
            )
            for dt, ql_curve in ql_discount_curves_dict.items()
        }

        items_to_process = []
        cached_results = []
        for curr_date, curr_vol_cube_dict_df in vol_cube_ts_dict_df.items():
            cache_key = (curr_date, tuple(sorted(query.col_name() for query in nvol_queries)))
            if hasattr(self, "_nvol_timeseries_cache") and cache_key in self._nvol_timeseries_cache:
                cached_results.append(self._nvol_timeseries_cache[cache_key])
            else:
                items_to_process.append((curr_date, curr_vol_cube_dict_df, discount_curve_nodes_dict[curr_date], cache_key))

        new_results = Parallel(n_jobs=n_jobs)(
            delayed(_process_nvol_timeseries_with_cache)(
                curr_date, curr_vol_cube_dict_df, discount_curve_nodes, nvol_queries, self._curve, self._swaps_product._ql_interpolation_algo
            )
            for curr_date, curr_vol_cube_dict_df, discount_curve_nodes, _ in tqdm.tqdm(items_to_process, desc="PRICING SWAPTIONS...")
        )

        for ck, result in new_results:
            self._nvol_timeseries_cache[ck] = result

        all_results = cached_results + [result for ck, result in new_results]
        timeseries_df = pd.DataFrame(all_results)
        timeseries_df.set_index("Date", inplace=True)
        return timeseries_df

    def _parse_swaption_nvol_queries_from_col(self, col: str):
        pattern = r"NVOL_([0-9]+[MY])x([0-9]+[MY])_ATMF([+-]\d+)"
        matches = re.findall(pattern, col)
        seen = set()
        queries = []
        for expiry, tail, offset_str in matches:
            key = (expiry, tail, offset_str)
            if key not in seen:
                seen.add(key)
                queries.append(SwaptionNvolQuery(expiry=expiry, tail=tail, atmf_offset_bps=int(offset_str)))
        return queries

    def _wrap_nvol_column_names(self, expr: str) -> str:
        # The regex matches patterns like NVOL_3Mx2Y_ATMF+100 or NVOL_3Mx2Y_ATMF-25.
        # The negative lookbehind/lookahead ensures that we do not double-wrap if already wrapped.
        pattern = r"(?<!`)(NVOL_[0-9]+[MY]x[0-9]+[MY]_ATMF[+-]\d+)(?!`)"
        return re.sub(pattern, r"`\1`", expr)

    def timeseries_builder(self, start_date: datetime, end_date: datetime, cols: List[str | Tuple[str, str]], n_jobs: Optional[int] = None) -> pd.DataFrame:
        should_fetch_swaps = True in [
            self._swaps_product._DEFAULT_SWAP_PREFIX in col[0] if isinstance(col, tuple) else self._swaps_product._DEFAULT_SWAP_PREFIX in col for col in cols
        ]
        should_fetch_nvols = True in [self._DEFAULT_SWAPTION_PREFIX in col[0] if isinstance(col, tuple) else self._DEFAULT_SWAPTION_PREFIX in col for col in cols]

        ts_df: pd.DataFrame = None

        if should_fetch_swaps:
            extracted_fwd_tenors = set()
            extracted_underlying_tenors = set()
            for col in cols:
                if isinstance(col, tuple):
                    col = col[0]
                curr_ex_fwds, curr_ex_unds = self._swaps_product._extract_swap_tenors(s=col)
                extracted_fwd_tenors.update(curr_ex_fwds)
                extracted_underlying_tenors.update(curr_ex_unds)

            ts_df = self._swaps_product._build_fwd_swaps_timeseries(
                ql_curves_ts_dict=self._swaps_product._fetch_ql_swap_curves(start_date=start_date, end_date=end_date),
                fwd_tenors=extracted_fwd_tenors,
                underlying_tenors=extracted_underlying_tenors,
                n_jobs=n_jobs,
            )

        if should_fetch_nvols:
            parsed_nvol_queries: List[SwaptionNvolQuery] = []
            for col in cols:
                if isinstance(col, tuple):
                    col = col[0]
                parsed_nvol_queries += self._parse_swaption_nvol_queries_from_col(col)

            nvol_ts_df = self._build_nvol_timeseries(start_date=start_date, end_date=end_date, nvol_queries=parsed_nvol_queries, n_jobs=n_jobs)
            if ts_df is not None:
                ts_df = ts_df.join(nvol_ts_df)
            else:
                ts_df = nvol_ts_df

        cols_to_return = []
        for col in cols:
            try:
                if isinstance(col, tuple):
                    expr = self._wrap_nvol_column_names(col[0])
                    ts_df[col[1]] = ts_df.eval(expr)
                    cols_to_return.append(col[1])
                else:
                    expr = self._wrap_nvol_column_names(col)
                    ts_df[col] = ts_df.eval(expr)
                    cols_to_return.append(col)
            except Exception as e:
                self._logger.error(f"'timeseries_builder' eval failed for {col}: {e}")

        return ts_df[cols_to_return].sort_index()

    def vol_surface_plotter(
        self,
        date: datetime,
        strike_offset: Optional[Literal[-200, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 200]] = None,
        expiry: Optional[str] = None,
        tail: Optional[str] = None,
        use_plotly: Optional[bool] = False,
        return_df: Optional[bool] = False,
    ):
        assert strike_offset is not None or expiry or tail, "MUST PASS IN A 'strike_offset', 'expiry', or 'tail'"
        vol_cube_dict_df = self._fetch_scubes(start_date=date, end_date=date)[date]

        if strike_offset is not None:
            scube: SCube = self._fetch_scubes(start_date=date, end_date=date)[date]
            if not strike_offset in scube:
                raise ValueError(f"Strike offset: {strike_offset} not in {date.date()}'s cube")

            vol_grid_df = scube[strike_offset]
            if use_plotly:
                self._plotly_vol_surface_plotter(
                    vol_grid_df.iloc[::-1].T,
                    f"{date.date()} ATMF{strike_offset:+d} Vol Grid",
                    "Expiry",
                    "Tail",
                    self._DEFAULT_SWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    vol_grid_df.iloc[::-1].T,
                    f"{date.date()} ATMF{strike_offset:+d} Vol Grid",
                    "Expiry",
                    "Tail",
                    self._DEFAULT_SWAPTION_PREFIX,
                )

            if return_df:
                return vol_grid_df

        elif expiry:
            tail_strike_surface = []
            for curr_strike_offset, expiry_tail_grid_df in vol_cube_dict_df.items():
                tail_strike_surface.append({"Strike": curr_strike_offset} | expiry_tail_grid_df.loc[expiry].to_dict())

            tail_strike_surface_df = pd.DataFrame(tail_strike_surface).set_index("Strike")
            if use_plotly:
                self._plotly_vol_surface_plotter(
                    tail_strike_surface_df.T,
                    f"{date.date()} {expiry} Expiry, Tail-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Tail",
                    self._DEFAULT_SWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    tail_strike_surface_df.T,
                    f"{date.date()} {expiry} Expiry, Tail-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Tail",
                    self._DEFAULT_SWAPTION_PREFIX,
                )

            if return_df:
                return tail_strike_surface_df

        else:
            expiry_strike_surface = []
            for curr_strike_offset, expiry_tail_grid_df in vol_cube_dict_df.items():
                for expiry, row in expiry_tail_grid_df.iterrows():
                    expiry_strike_surface.append(
                        {
                            "Strike": curr_strike_offset,
                            "Expiry": expiry,
                            self._DEFAULT_SWAPTION_PREFIX: row[tail],
                        }
                    )

            expiry_strike_surface_df = pd.DataFrame(expiry_strike_surface)
            expiry_strike_surface_df["Strike"] = pd.to_numeric(expiry_strike_surface_df["Strike"])
            expiry_strike_surface_df = (
                expiry_strike_surface_df.sort_values(by=["Strike"])
                .pivot(index="Strike", columns="Expiry", values=self._DEFAULT_SWAPTION_PREFIX)
                .dropna(axis="columns")
            )
            expiry_strike_surface_df = expiry_strike_surface_df[sorted(expiry_strike_surface_df.columns, key=lambda x: ql.Period(x))]

            if use_plotly:
                self._plotly_vol_surface_plotter(
                    expiry_strike_surface_df.T,
                    f"{date.date()} {tail} Tail, Expiry-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Expiry",
                    self._DEFAULT_SWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    expiry_strike_surface_df.T,
                    f"{date.date()} {tail} Tail, Expiry-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Expiry",
                    self._DEFAULT_SWAPTION_PREFIX,
                )

            if return_df:
                return expiry_strike_surface_df

    def vol_smile_plotter(
        self,
        dates: List[datetime],
        tenor: str,
        return_dict: Optional[bool] = False,
        strike_offsets_bps: Optional[List[int]] = None,
        use_offsets: Optional[bool] = True,
        use_plotly: Optional[bool] = False,
    ):
        expiry, tail = tenor.split("x")
        DEFAULT_STRIKE_OFFSETS_BPS = [-200, -150, -100, -75, -50, -25, 0, 25, 50, 75, 100, 150, 200]
        strike_offsets_bps = DEFAULT_STRIKE_OFFSETS_BPS if not strike_offsets_bps else strike_offsets_bps

        ql_vol_cubes = self._fetch_qlcubes(bdates=dates)

        result_dict = {}
        all_data = []

        for d in dates:
            ql_vol_cube = ql_vol_cubes[d]
            ql_smile_section: ql.SmileSection = ql_vol_cube.smileSection(ql.Period(expiry), ql.Period(tail), True)
            atm_strike = ql_smile_section.atmLevel()
            strikes = [atm_strike + (offset / 10_000) for offset in strike_offsets_bps]
            nvols = [ql_smile_section.volatility(strike, ql.Normal) * 10_000 for strike in strikes]
            try:
                atm_index = strike_offsets_bps.index(0)
            except ValueError:
                atm_index = None

            result_dict[d] = {
                "atm_strike": atm_strike,
                "strikes": strikes,
                "nvols": nvols,
                "strike_offsets_bps": strike_offsets_bps,
            }
            all_data.append((d, strikes, nvols, atm_index))

        if use_offsets:
            x_axis_label = "Strike Offset (bps)"
        else:
            x_axis_label = "Strike"

        if use_plotly:
            fig = go.Figure()
            for d, strikes, nvols, atm_index in all_data:
                x_vals = strike_offsets_bps if use_offsets else strikes
                custom_data = list(zip(np.round(np.array(strikes) * 100, 5), strike_offsets_bps))
                hovertemplate = "<b>Strike:</b> %{customdata[0]}<br>" "<b>Strike Offset:</b> %{customdata[1]}<br>" "<b>NVOL:</b> %{y}<extra></extra>"
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=nvols,
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(width=2),
                        name=f"{d.date()}",
                        customdata=custom_data,
                        hovertemplate=hovertemplate,
                    )
                )
            fig.update_layout(
                title=f"{expiry}x{tail} Swaption Smile",
                xaxis_title=x_axis_label,
                yaxis_title=self._DEFAULT_SWAPTION_PREFIX,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Smiles:"),
                template="plotly_dark",
                height=750,
                width=1250,
            )
            fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
            fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
            fig.show()
        else:
            plt.figure()
            for d, strikes, nvols, atm_index in all_data:
                x_vals = strike_offsets_bps if use_offsets else strikes
                plt.plot(x_vals, nvols, marker="o", linestyle="-", label=f"{d.date()} Smile")
                if atm_index is not None:
                    plt.plot(x_vals[atm_index], nvols[atm_index], "ro", markersize=10, label=f"{d.date()} ATMF")
            plt.title(f"{expiry}x{tail} Swaption Smile")
            plt.xlabel(x_axis_label)
            plt.ylabel("NVOL")
            plt.grid(True)
            plt.legend()
            plt.show()

        if return_dict:
            return result_dict

    def term_structure_plotter(
        self,
        dates: List[datetime],
        expiry: Optional[str] = None,
        tail: Optional[str] = None,
        strike_offset: Optional[int] = 0,
        strike: Optional[int] = None,
    ):
        assert expiry or tail, "MUST PASS IN A 'expiry' or 'tail'"

        ql_vol_cubes: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube] = self._fetch_qlcubes(bdates=dates)
        term_structure_dict_df: Dict[datetime, pd.DataFrame] = {}
        ql_cube_iter = tqdm.tqdm(ql_vol_cubes.items(), desc="PLOTTING TERM STRUCTURE...") if self._show_tqdm else ql_vol_cubes.items()

        if expiry is not None:
            sample_cube = next(iter(ql_vol_cubes.values()))
            tails = sample_cube.swapTenors()
            columns = [str(t) for t in tails]

            for d, cube in ql_cube_iter:
                vols = []
                for t in tails:
                    smile_section: ql.SmileSection = cube.smileSection(ql.Period(expiry), t, True)
                    strike = smile_section.atmLevel() + (strike_offset / 10_000) if strike is None else strike
                    vols.append(smile_section.volatility(strike, True) * 10_000)

                df = pd.DataFrame([vols], columns=columns, index=[expiry])
                term_structure_dict_df[d] = df

            plot_title = f"ATMF{strike_offset:+d} Term Structure for {expiry} Expiry"
            x_axis_title = "Tail"
            y_axis_title = self._DEFAULT_SWAPTION_PREFIX
        else:
            sample_cube = next(iter(ql_vol_cubes.values()))
            expiries = sample_cube.optionTenors()
            columns = [str(e) for e in expiries]

            for d, cube in ql_cube_iter:
                vols = []
                for e in expiries:
                    smile_section: ql.SmileSection = cube.smileSection(e, ql.Period(tail), True)
                    strike = smile_section.atmLevel() + (strike_offset / 10_000) if strike is None else strike
                    vols.append(smile_section.volatility(strike, True) * 10_000)

                df = pd.DataFrame([vols], columns=columns, index=[tail])
                term_structure_dict_df[d] = df

            plot_title = f"ATMF{strike_offset:+d} Term Structure for {tail} Tail"
            x_axis_title = "Expiry"
            y_axis_title = self._DEFAULT_SWAPTION_PREFIX

        self._term_structure_plotter(
            term_structure_dict_df=term_structure_dict_df,
            plot_title=plot_title,
            x_axis_col_sorter_func=lambda x: ql_period_to_months(ql.Period(x)),
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
            show_idx_tenor_in_legend=True,
            use_plotly=True,
        )


def _process_nvol_timeseries_with_cache(
    as_of_date: datetime,
    vol_cube_dict: Dict[int, pd.DataFrame],
    discount_curve_nodes: Dict[datetime, float],
    nvol_queries: List[SwaptionNvolQuery],
    curve: str,
    ql_interpolation_algo: str,
) -> Tuple[Tuple[datetime, Tuple[str, ...]], Dict]:
    cache_key = (as_of_date, tuple(sorted(query.col_name() for query in nvol_queries)))
    result = _process_nvol_timeseries(as_of_date, curve, vol_cube_dict, discount_curve_nodes, nvol_queries, ql_interpolation_algo)
    return cache_key, result


def _process_nvol_timeseries(
    as_of_date: datetime,
    curve_str: str,
    vol_cube_dict: Dict[int, pd.DataFrame],
    ql_curve_nodes: Dict[datetime, float],
    nvol_queries: List[SwaptionNvolQuery],
    ql_interpolation_algo_str: str,
) -> Dict:
    ql_curve = build_ql_discount_curve(
        datetime_series=pd.Series(ql_curve_nodes.keys()),
        discount_factor_series=pd.Series(ql_curve_nodes.values()),
        ql_dc=CME_SWAP_CURVE_QL_PARAMS[curve_str]["dayCounter"],
        ql_cal=CME_SWAP_CURVE_QL_PARAMS[curve_str]["calendar"],
        interpolation_algo=ql_interpolation_algo_str,
    )

    if CME_SWAP_CURVE_QL_PARAMS[curve_str]["is_ois"]:
        ql_swap_index_wrapper = ql.OvernightIndexedSwapIndex
    else:
        ql_swap_index_wrapper = ql.SwapIndex

    ql_swap_index = ql_swap_index_wrapper(
        curve_str,
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["period"],
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["settlementDays"],
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["currency"],
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["swapIndex"](ql.YieldTermStructureHandle(ql_curve)),
    )

    atm_vol_grid = vol_cube_dict[0]
    expiries = [ql.Period(e) for e in atm_vol_grid.index]
    tails = [ql.Period(t) for t in atm_vol_grid.columns]

    atm_swaption_vol_matrix = ql.SwaptionVolatilityMatrix(
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["calendar"],
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["businessConvention"],
        expiries,
        tails,
        ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_grid.values]),
        CME_SWAP_CURVE_QL_PARAMS[curve_str]["dayCounter"],
        False,
        ql.Normal,
    )

    vol_spreads = []
    strike_spreads = [float(k) / 10_000 for k in vol_cube_dict.keys()]
    strike_offsets = sorted(vol_cube_dict.keys(), key=lambda x: int(x))
    for option_tenor in atm_vol_grid.index:
        for swap_tenor in atm_vol_grid.columns:
            vol_spread_row = [
                ql.QuoteHandle(ql.SimpleQuote((vol_cube_dict[strike].loc[option_tenor, swap_tenor] - atm_vol_grid.loc[option_tenor, swap_tenor]) / 10_000))
                for strike in strike_offsets
            ]
            vol_spreads.append(vol_spread_row)

    ql_vol_cube = ql.InterpolatedSwaptionVolatilityCube(
        ql.SwaptionVolatilityStructureHandle(atm_swaption_vol_matrix),
        expiries,
        tails,
        strike_spreads,
        vol_spreads,
        ql_swap_index,
        ql_swap_index,
        True,
    )
    ql_vol_cube.enableExtrapolation()

    ql.Settings.instance().evaluationDate = datetime_to_ql_date(as_of_date)
    precomputed_queries = []
    for query in nvol_queries:
        precomputed_queries.append(
            {
                "query": query,
                "ql_period_expiry": ql.Period(query.expiry),
                "ql_period_tail": ql.Period(query.tail),
                "col_name": query.col_name(),
                "atmf_offset": query.atmf_offset_bps / 10_000,  # Convert basis points to a decimal offset
            }
        )

    curr_ts = {"Date": as_of_date}
    for prec in precomputed_queries:
        try:
            expiry_ql_date = CME_SWAP_CURVE_QL_PARAMS[curve_str]["calendar"].advance(
                datetime_to_ql_date(as_of_date),
                prec["ql_period_expiry"],
                CME_SWAP_CURVE_QL_PARAMS[curve_str]["businessConvention"],
            )
            curr_atmf = ql_vol_cube.atmStrike(expiry_ql_date, prec["ql_period_tail"])
            vol = ql_vol_cube.volatility(
                expiry_ql_date,
                prec["ql_period_tail"],
                curr_atmf + prec["atmf_offset"],
                True,
            )
            curr_ts[prec["col_name"]] = vol * 10_000

        # TODO handle error
        except Exception:
            pass

    return curr_ts
