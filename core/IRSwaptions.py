import functools
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from joblib import Parallel, delayed

from core.Caching.ZODBCacheMixin import ZODBCacheMixin
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.CurveBuilding.IRSwaps.ql_curve_building_utils import build_ql_discount_curve, get_nodes_dict
from core.CurveBuilding.IRSwaptions.IRSwaptionCubeBuilder import IRSwaptionCubeBuilder, SCube
from core.CurveBuilding.IRSwaptions.ql_cube_building_utils import build_ql_interpolated_vol_cube, df_to_cube_ts
from core.DataFetching.GithubFetcher import GithubFetcher
from core.IRSwaps import IRSwaps
from core.Plotting.BaseProductPlotter import BaseProductPlotter
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionQuery import IRSwaptionQuery, IRSwaptionQueryWrapper
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionStructure import IRSwaptionStructure, IRSwaptionStructureFunctionMap
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionValue import IRSwaptionValue, IRSwaptionValueFunctionMap
from core.utils.ql_loader import ql
from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, ql_date_to_datetime, ql_period_to_months


class IRSwaptions(BaseProductPlotter, ZODBCacheMixin):
    _EXPIRY_COL: str = "Expiry"
    _TAIL_COL: str = "Tail"

    _DEFAULT_TZ = "US/Eastern"
    _DEFAULT_DSF_KEY = "_DEFAULT_DSF_KEY"
    _DEFAULT_DSF_OBJ = "_DEFAULT_DSF_OBJ"
    _DEFAULT_DSF_OBJ_ARGS = "_DEFAULT_DSF_OBJ_ARGS"
    _DEFAULT_DFS_FETCH_QLCUBES_FUNC = "_DEFAULT_DFS_FETCH_QLCUBES_FUNC"
    _DEFAULT_DFS_FETCH_QLCUBES_FUNC_ARGS = "_DEFAULT_DFS_FETCH_QLCUBES_FUNC_ARGS"
    _DEFAULT_DFS_FETCH_SCUBES_FUNC = "_DEFAULT_DFS_FETCH_SCUBES_FUNC"
    _DEFAULT_DFS_FETCH_SCUBES_FUNC_ARGS = "_DEFAULT_DFS_FETCH_SCUBES_FUNC_ARGS"
    _DEFAULT_IRSWAPTION_PREFIX = "IRSWAPTION"
    _DEFAULT_CSV_PREFIX = "CSV_"

    _IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES = "_INPUT_DATES"
    _IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES = "_INPUT_QL_DISC_CURVES"
    _DEFAULT_STRIKE_OFFSETS_BPS = [-200, -150, -125, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 125, 150, 200]

    _IRSWAPTIONS_SCUBE_CACHE = "_irswaptions_scube_cache"
    _IRSWAPTIONS_TIMESERIES_CACHE = "_irswaptions_timeseries_cache"

    def __init__(
        self,
        data_source: Literal["CSV{path}", "GITHUB"],
        irswaps_product: IRSwaps,
        precalibrate_cubes: Optional[bool] = False,
        max_n_jobs: Optional[int] = 1,
        date_col: Optional[str] = "Date",
        show_tqdm: Optional[bool] = True,
        proxies: Optional[Dict[str, str]] = None,
        info_verbose: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        BaseProductPlotter.__init__(
            self,
            info_verbose=info_verbose,
            debug_verbose=debug_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )
        ZODBCacheMixin.__init__(self)

        # in memory ql cube cache
        self._qlcube_cache = {}

        self._irswaps_product = irswaps_product
        self._curve = self._irswaps_product._curve

        if self._DEFAULT_CSV_PREFIX in data_source:
            self._data_source = "CSV"  # for the cache

            self._ensure_cache(self._IRSWAPTIONS_SCUBE_CACHE)
            csv_path = data_source.split("_", 1)[-1]
            df_wide = pd.read_csv(csv_path).set_index(date_col)
            df_wide.index = pd.to_datetime(df_wide.index)

            cached_dates = set(getattr(self, self._IRSWAPTIONS_SCUBE_CACHE).keys())
            missing_mask = ~df_wide.index.isin(cached_dates)
            if missing_mask.any():
                getattr(self, self._IRSWAPTIONS_SCUBE_CACHE).update(df_to_cube_ts(df_wide.loc[missing_mask]))
                self.zodb_commit()

            self.close_zodb()
            self._data_source = "JSON"
        else:
            self._data_source = data_source

        self._precalibrate_cubes = precalibrate_cubes
        self._n_jobs = max_n_jobs

        self._show_tqdm = show_tqdm
        self._proxies = proxies

        self.__data_sources_funcs = {
            self._DEFAULT_DSF_KEY: {
                "JSON": {
                    self._DEFAULT_DSF_OBJ: IRSwaptionCubeBuilder,
                    self._DEFAULT_DSF_OBJ_ARGS: (
                        getattr(self, self._IRSWAPTIONS_SCUBE_CACHE, None),
                        None,
                        info_verbose,
                        debug_verbose,
                        warning_verbose,
                        error_verbose,
                    ),
                    self._DEFAULT_DFS_FETCH_QLCUBES_FUNC: "parallel_ql_vol_cube_builder",
                    self._DEFAULT_DFS_FETCH_QLCUBES_FUNC_ARGS: (
                        self._curve,
                        self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES,
                        False,
                        self._precalibrate_cubes,
                        self._show_tqdm,
                        None,
                        self._n_jobs,
                    ),
                    self._DEFAULT_DFS_FETCH_SCUBES_FUNC: "return_scube",
                    self._DEFAULT_DFS_FETCH_SCUBES_FUNC_ARGS: (self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,),
                },
                "GITHUB_USD_MONKEY_CUBE": {
                    self._DEFAULT_DSF_OBJ: GithubFetcher,
                    self._DEFAULT_DSF_OBJ_ARGS: (
                        self._irswaps_product._FETCHER_TIMEOUT,
                        self._proxies,
                        self._debug_verbose,
                        self._info_verbose,
                        self._warning_verbose,
                        self._error_verbose,
                    ),
                    self._DEFAULT_DFS_FETCH_QLCUBES_FUNC: "fetch_USD_MONKEY_CUBE_ql_vol_cubes",
                    self._DEFAULT_DFS_FETCH_QLCUBES_FUNC_ARGS: (
                        self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES,
                        False,
                        self._precalibrate_cubes,
                        self._n_jobs,
                        self._show_tqdm,
                        self._irswaps_product._MAX_CONNECTIONS,
                        self._irswaps_product._MAX_KEEPALIVE_CONNECTIONS,
                        "yieldcurvemonkey/MONKEY_CUBE/refs/heads/main",
                    ),
                    self._DEFAULT_DFS_FETCH_SCUBES_FUNC: "fetch_USD_MONKEY_CUBE_dict_df",
                    self._DEFAULT_DFS_FETCH_SCUBES_FUNC_ARGS: (
                        self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES,
                        self._show_tqdm,
                        self._irswaps_product._MAX_CONNECTIONS,
                        self._irswaps_product._MAX_KEEPALIVE_CONNECTIONS,
                        "yieldcurvemonkey/MONKEY_CUBE/refs/heads/main",
                    ),
                },
            }
        }
        self._irswaption_hist_data_fetcher = self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DSF_OBJ](
            *self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DSF_OBJ_ARGS]
        )

    def _cache_config(self) -> dict[str, dict]:
        cache_ds_id = self._data_source if not self._data_source.startswith("CSV_") else f"CSV_{Path(self._data_source.split('_', 1)[-1]).stem}"
        return {
            self._IRSWAPTIONS_SCUBE_CACHE: dict(
                path=ZODBCacheMixin.default_cache_path(
                    stem=f"{self._IRSWAPTIONS_SCUBE_CACHE}_{cache_ds_id}_{self._irswaps_product._curve}_{self._irswaps_product._ql_interpolation_algo}"
                ),
            ),
            self._IRSWAPTIONS_TIMESERIES_CACHE: dict(
                path=ZODBCacheMixin.default_cache_path(
                    stem=f"{self._IRSWAPTIONS_TIMESERIES_CACHE}_{cache_ds_id}_{self._irswaps_product._curve}_{self._irswaps_product._ql_interpolation_algo}"
                ),
            ),
        }

    def _ensure_cache(self, attr: str, force_refresh: Optional[bool] = False):
        if hasattr(self, attr):
            delattr(self, attr)

        cfg = self._cache_config()[attr]
        self.zodb_open_cache(cache_attr=attr, force=force_refresh, **cfg)

    def fetch_qlcubes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        to_pydt: Optional[bool] = False,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, ql.InterpolatedSwaptionVolatilityCube]:
        assert (start_date and end_date) or bdates, "MUST PASS IN start and end dates or a list of bdates"

        input_bdates = (
            # TODO more robust timezone handling e.g. parse from data source
            get_bdates_between(start_date=start_date, end_date=end_date, calendar=self._irswaps_product._ql_calendar)
            if start_date and end_date
            else bdates
        )

        cached_dates = {ts.date() for ts in self._qlcube_cache.keys()}
        if refresh_cache:
            bdates_not_cached = input_bdates
        else:
            bdates_not_cached = [d for d in input_bdates if d.date() not in cached_dates]

        if bdates_not_cached:
            fetch = getattr(
                self._irswaption_hist_data_fetcher, self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DFS_FETCH_QLCUBES_FUNC]
            )
            args = tuple(
                (
                    bdates_not_cached
                    if a == self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES
                    else (
                        self._irswaps_product.fetch_ql_irswap_curves(start_date=start_date, end_date=end_date, bdates=bdates)
                        if a == self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_QL_DISC_CURVES
                        else a
                    )
                )
                for a in self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DFS_FETCH_QLCUBES_FUNC_ARGS]
            )
            self._qlcube_cache |= fetch(*args)

        ts_map = {ts.date(): ts for ts in self._qlcube_cache.keys()}
        result: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube] = {}

        for bday in input_bdates:
            ts = ts_map.get(bday.date())
            if ts is None:
                continue
            cube = self._qlcube_cache[ts]

            if to_pydt:
                result[bday] = cube
            else:
                result[ts] = cube

        return result

    def fetch_scubes(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        to_pydt: Optional[bool] = False,
        refresh_cache: Optional[bool] = False,
    ) -> Dict[datetime, SCube]:
        try:
            self._ensure_cache(self._IRSWAPTIONS_SCUBE_CACHE, refresh_cache)
            assert (start_date and end_date) or bdates, "MUST PASS IN start/end dates OR explicit bdates"
            input_bdates = get_bdates_between(start_date, end_date, self._irswaps_product._ql_calendar) if (start_date and end_date) else bdates

            cached_dates = {ts.date() for ts in getattr(self, self._IRSWAPTIONS_SCUBE_CACHE).keys()}
            if refresh_cache:
                bdates_not_cached = input_bdates
            else:
                bdates_not_cached = [d for d in input_bdates if d.date() not in cached_dates]

            if bdates_not_cached:
                fetch = getattr(
                    self._irswaption_hist_data_fetcher, self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DFS_FETCH_SCUBES_FUNC]
                )
                args = tuple(
                    (self._irswaps_product.fetch_ql_irswap_curves(bdates=bdates_not_cached).keys() if a == self._IRSWAPTION_VOL_CUBE_FETCH_FUNC_INPUT_DATES else a)
                    for a in self.__data_sources_funcs[self._DEFAULT_DSF_KEY][self._data_source][self._DEFAULT_DFS_FETCH_SCUBES_FUNC_ARGS]
                )
                getattr(self, self._IRSWAPTIONS_SCUBE_CACHE).update(fetch(*args))
                self.zodb_commit()

            ts_map = {ts.date(): ts for ts in getattr(self, self._IRSWAPTIONS_SCUBE_CACHE).keys()}
            result: Dict[datetime, SCube] = {}

            for bday in input_bdates:
                ts = ts_map.get(bday.date())
                if ts is None:
                    continue
                scube = getattr(self, self._IRSWAPTIONS_SCUBE_CACHE)[ts]

                if to_pydt:
                    result[bday] = scube
                else:
                    result[ts] = scube

            self.close_zodb()
            return result

        except Exception as e:
            self.close_zodb()
            raise e

    def _build_irswaption_queries_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        irswaption_queries: List[IRSwaptionQuery],
        n_jobs: Optional[int] = 1,
        ignore_cache: Optional[False] = False,
    ) -> pd.DataFrame:
        self._ensure_cache(self._IRSWAPTIONS_TIMESERIES_CACHE)

        vol_cube_ts = self.fetch_scubes(start_date=start_date, end_date=end_date)
        ql_disc_curves = self._irswaps_product.fetch_ql_irswap_curves(start_date=start_date, end_date=end_date)
        dc_nodes = {d: get_nodes_dict(ql_curve) for d, ql_curve in ql_disc_curves.items()}

        tasks, cached_rows = [], []
        for ref_date, vol_cube in vol_cube_ts.items():
            for q in irswaption_queries:
                q_key = _irswaption_query_key(q, self._curve)
                ck = (ref_date, q_key)  # cache key
                if ck in getattr(self, self._IRSWAPTIONS_TIMESERIES_CACHE) and not ignore_cache:
                    cached_rows.append(getattr(self, self._IRSWAPTIONS_TIMESERIES_CACHE)[ck])
                else:
                    tasks.append((ck, ref_date, vol_cube, dc_nodes[ref_date], q))

        if self._show_tqdm:
            tasks_iter = tqdm.tqdm(tasks, desc="PRICING IRSWAPTIONS…")
        else:
            tasks_iter = tasks

        new_results = Parallel(n_jobs=n_jobs, timeout=99999)(
            delayed(_process_irswaption_single_query)(
                ref_date,
                vol_cube,
                ql_curve_nodes,
                query,
                self._curve,
                self._irswaps_product._ql_interpolation_algo,
            )
            for _, ref_date, vol_cube, ql_curve_nodes, query in tasks_iter
        )

        for (ck, _, _, _, _), (d, col, val) in zip(tasks, new_results):
            getattr(self, self._IRSWAPTIONS_TIMESERIES_CACHE)[ck] = (d, col, val)
        if new_results:
            self.zodb_commit()

        all_rows = cached_rows + [(d, col, val) for d, col, val in new_results]
        if not all_rows:
            return pd.DataFrame([], index=pd.DatetimeIndex([]))

        df = (
            pd.DataFrame(all_rows, columns=["Date", "col", "val"])
            .drop_duplicates(subset=["Date", "col"], keep="last")
            .pivot(index="Date", columns="col", values="val")
            .sort_index()
        )
        df.columns.name = None

        self.close_zodb()
        return df

    def timeseries_builder(
        self,
        start_date: datetime,
        end_date: datetime,
        queries: List[IRSwaptionQuery | IRSwaptionQueryWrapper],
        n_jobs: Optional[int] = None,
        ignore_cache: Optional[False] = False,
    ) -> pd.DataFrame:
        flatten_queries: List[IRSwaptionQuery] = []
        expr_cols: List[Tuple[str, str]] = []
        ignore_risk_weight = False
        for q in queries:
            if isinstance(q, IRSwaptionQueryWrapper):
                expr_cols.append((q.eval_expression(self._curve), q.col_name(self._curve)))
                ignore_risk_weight = q.ignore_risk_weights
            flatten_queries += q.return_query()

        nvol_ts_df = self._build_irswaption_queries_timeseries(
            start_date=start_date, end_date=end_date, irswaption_queries=flatten_queries, n_jobs=n_jobs, ignore_cache=ignore_cache
        )

        cols_to_return = []
        for q in flatten_queries:
            curr_col = q.col_name(self._curve)
            nvol_ts_df[curr_col] = nvol_ts_df.eval(q.eval_expression(self._curve, ignore_risk_weight=ignore_risk_weight))
            cols_to_return.append(curr_col)

        if expr_cols:
            for eval_str, col_name in expr_cols:
                nvol_ts_df[col_name] = nvol_ts_df.eval(eval_str)
                cols_to_return.append(col_name)

        return nvol_ts_df[cols_to_return].sort_index()

    def vol_surface_plotter(
        self,
        date: datetime,
        strike_offset: Optional[int] = None,
        expiry: Optional[str] = None,
        tail: Optional[str] = None,
        use_plotly: Optional[bool] = False,
        return_df: Optional[bool] = False,
    ):
        assert strike_offset is not None or expiry or tail, "MUST PASS IN A 'strike_offset', 'expiry', or 'tail'"

        if strike_offset is not None:
            scube: SCube = self.fetch_scubes(bdates=[date], to_pydt=True)[date]
            if not strike_offset in scube:
                raise ValueError(f"Strike offset: {strike_offset} not in {date.date()}'s cube")

            vol_grid_df = scube[strike_offset]
            if use_plotly:
                self._plotly_vol_surface_plotter(
                    vol_grid_df.iloc[::-1].T,
                    f"{date.date()} ATMF{strike_offset:+d} Vol Grid",
                    "Expiry",
                    "Tail",
                    self._DEFAULT_IRSWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    vol_grid_df.iloc[::-1].T,
                    f"{date.date()} ATMF{strike_offset:+d} Vol Grid",
                    "Expiry",
                    "Tail",
                    self._DEFAULT_IRSWAPTION_PREFIX,
                )

            if return_df:
                return vol_grid_df

        elif expiry:
            vol_cube_dict_df = self.fetch_scubes(bdates=[date], to_pydt=True)[date]
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
                    self._DEFAULT_IRSWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    tail_strike_surface_df.T,
                    f"{date.date()} {expiry} Expiry, Tail-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Tail",
                    self._DEFAULT_IRSWAPTION_PREFIX,
                )

            if return_df:
                return tail_strike_surface_df

        else:
            vol_cube_dict_df = self.fetch_scubes(start_date=date, end_date=date, to_pydt=True)[date]
            expiry_strike_surface = []
            for curr_strike_offset, expiry_tail_grid_df in vol_cube_dict_df.items():
                for expiry, row in expiry_tail_grid_df.iterrows():
                    expiry_strike_surface.append(
                        {
                            "Strike": curr_strike_offset,
                            "Expiry": expiry,
                            self._DEFAULT_IRSWAPTION_PREFIX: row[tail],
                        }
                    )

            expiry_strike_surface_df = pd.DataFrame(expiry_strike_surface)
            expiry_strike_surface_df["Strike"] = pd.to_numeric(expiry_strike_surface_df["Strike"])
            expiry_strike_surface_df = (
                expiry_strike_surface_df.sort_values(by=["Strike"])
                .pivot(index="Strike", columns="Expiry", values=self._DEFAULT_IRSWAPTION_PREFIX)
                .dropna(axis="columns")
            )
            expiry_strike_surface_df = expiry_strike_surface_df[sorted(expiry_strike_surface_df.columns, key=lambda x: ql.Period(x))]

            if use_plotly:
                self._plotly_vol_surface_plotter(
                    expiry_strike_surface_df.T,
                    f"{date.date()} {tail} Tail, Expiry-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Expiry",
                    self._DEFAULT_IRSWAPTION_PREFIX,
                )
            else:
                self._matplotlib_vol_surface_plotter(
                    expiry_strike_surface_df.T,
                    f"{date.date()} {tail} Tail, Expiry-Strike Vol Surface",
                    "ATMF Strike Offsets",
                    "Expiry",
                    self._DEFAULT_IRSWAPTION_PREFIX,
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
        strike_offsets_bps = self._DEFAULT_STRIKE_OFFSETS_BPS if not strike_offsets_bps else strike_offsets_bps

        ql_vol_cubes = self.fetch_qlcubes(bdates=dates, to_pydt=True)

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
                title=f"{expiry}x{tail} {self._curve} IRSwaption Smile",
                xaxis_title=x_axis_label,
                yaxis_title=self._DEFAULT_IRSWAPTION_PREFIX,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Smiles:"),
                template="plotly_dark",
                height=750,
                width=1250,
                showlegend=True,
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
            plt.title(f"{expiry}x{tail} {self._curve} IRSwaption Smile")
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

        ql_vol_cubes: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube] = self.fetch_qlcubes(bdates=dates)
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
            y_axis_title = self._DEFAULT_IRSWAPTION_PREFIX
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
            y_axis_title = self._DEFAULT_IRSWAPTION_PREFIX

        self._term_structure_plotter(
            term_structure_dict_df=term_structure_dict_df,
            plot_title=plot_title,
            x_axis_col_sorter_func=lambda x: ql_period_to_months(ql.Period(x)),
            x_axis_title=x_axis_title,
            y_axis_title=y_axis_title,
            show_idx_tenor_in_legend=True,
            use_plotly=True,
        )


def _irswaption_query_key(q: IRSwaptionQuery, curve: str) -> Tuple:
    return (
        q.col_name(curve),
        q.expiry,
        q.tail,
        q.exercise_date,
        q.underlying_effective_date,
        q.underlying_maturity_date,
        tuple(sorted(q.structure_kwargs.items())) if q.structure_kwargs else None,
        q.value,
        q.structure,
    )


def _process_irswaption_single_query(
    ref_date: datetime,
    vol_cube: SCube | Dict[int, pd.DataFrame],
    ql_curve_nodes: Dict[datetime, float],
    query: IRSwaptionQuery,
    curve: str,
    ql_interpolation_algo: str,
) -> Tuple[datetime, str, float]:
    row = _process_irswaption_queries(
        ref_date=ref_date,
        curve=curve,
        vol_cube=vol_cube,
        ql_curve_nodes=ql_curve_nodes,
        swaption_queries=[query],
        ql_interpolation_algo_str=ql_interpolation_algo,
    )
    col = query.col_name(curve)
    return ref_date, col, row[col]


def _process_irswaption_queries(
    ref_date: datetime,
    curve: str,
    vol_cube: SCube | Dict[int, pd.DataFrame],
    ql_curve_nodes: Dict[datetime, float],
    swaption_queries: List[IRSwaptionQuery],
    ql_interpolation_algo_str: str,
) -> Dict:
    ql.Settings.instance().evaluationDate = datetime_to_ql_date(ref_date)

    ql_curve = build_ql_discount_curve(
        datetime_series=pd.Series(ql_curve_nodes.keys()),
        discount_factor_series=pd.Series(ql_curve_nodes.values()),
        ql_dc=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
        ql_cal=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
        interpolation_algo=f"df_{ql_interpolation_algo_str}",
    )
    curve_handle = ql.YieldTermStructureHandle(ql_curve)

    ql_swap_index = _ql_swap_index_cached(curve, curve_handle)
    vol_handle = ql.SwaptionVolatilityStructureHandle(build_ql_interpolated_vol_cube(vol_cube, ql_swap_index))
    pricing_engine = _ql_swaption_pricing_engine_cached(curve_handle, vol_handle)

    results: Dict[str, float] = {"Date": ref_date}

    for query in swaption_queries:
        col = query.col_name(curve)
        package, rws = IRSwaptionStructureFunctionMap(
            curve=curve,
            curve_handle=curve_handle,
            pricing_engine=pricing_engine,
        ).apply(
            structure=query.structure,
            expiry=query.expiry,
            tail=query.tail,
            exercise_date=query.exercise_date,
            underlying_effective_date=query.underlying_effective_date,
            underlying_maturity_date=query.underlying_maturity_date,
            **(query.structure_kwargs or {}),
        )
        results[col] = IRSwaptionValueFunctionMap(
            package=package, risk_weights=rws, curve=curve, curve_handle=curve_handle, pricing_engine=pricing_engine, swaption_structure=query.structure
        ).apply(query.value)

    return results


@functools.lru_cache(maxsize=252)
def _ql_swap_index_cached(curve: str, curve_handle: ql.YieldTermStructureHandle) -> ql.SwapIndex:
    params = CME_IRSWAP_CURVE_QL_PARAMS[curve]
    idx_cls = ql.OvernightIndexedSwapIndex if params["is_ois"] else ql.SwapIndex
    return idx_cls(
        curve,
        params["period"],
        params["settlementDays"],
        params["currency"],
        params["swapIndex"](curve_handle),
    )


@functools.lru_cache(maxsize=252)
def _ql_swaption_pricing_engine_cached(curve_handle: ql.YieldTermStructureHandle, vol_handle: ql.SwaptionVolatilityStructureHandle) -> ql.BachelierSwaptionEngine:
    idx_cls = ql.BachelierSwaptionEngine
    return idx_cls(curve_handle, vol_handle, ql.BachelierSwaptionEngine.DiscountCurve)
