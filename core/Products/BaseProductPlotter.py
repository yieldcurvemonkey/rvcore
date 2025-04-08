import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm
from scipy.stats import tstd, zscore


class BaseProductPlotter:
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        info_verbose: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        warning_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self._info_verbose = info_verbose
        self._debug_verbose = debug_verbose
        self._warning_verbose = warning_verbose
        self._error_verbose = error_verbose
        self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        elif self._warning_verbose:
            self._logger.setLevel(logging.WARNING)
        else:
            self._logger.disabled = True

    @staticmethod
    def _timeseries_builder_from_grid(
        grid_dict_df: Dict[datetime, pd.DataFrame],
        cols: List[str | Tuple[str, str]],
        bdates: Optional[List[pd.Timestamp] | List[datetime]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tqdm_desc: Optional[str] = "BUILDING TIMESERIES DF...",
        flip_index: Optional[bool] = False,
        drop_na_dates: Optional[bool] = True,
    ) -> pd.DataFrame:
        filtered_dates = [d for d in grid_dict_df.keys() if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)]
        filtered_dates.sort()

        timeseries_data = {expr[0] if isinstance(expr, tuple) else expr: [] for expr in cols}
        dates_list = []
        token_pattern = re.compile(r"(\d+[DMY])x(\d+[DMY])")

        for dt in tqdm.tqdm(filtered_dates, desc=tqdm_desc):
            grid = grid_dict_df[dt]
            if flip_index:
                grid = grid.T
            dates_list.append(dt)

            for expr in cols:
                if isinstance(expr, tuple):
                    expr = expr[0]

                def _get_val(expiry: str, tail: str):
                    try:
                        return grid.loc[expiry, tail]
                    except Exception as e:
                        BaseProductPlotter._logger.error(f"'timeseries_builder' - Error for {expr} at {dt}: {e}")
                        return np.nan

                transformed_expr = token_pattern.sub(lambda m: f'_get_val("{m.group(1)}", "{m.group(2)}")', expr)
                try:
                    val = eval(transformed_expr, {"__builtins__": None}, {"_get_val": _get_val})
                except Exception:
                    val = np.nan
                timeseries_data[expr].append(val)

        ts_df = pd.DataFrame(timeseries_data, index=dates_list)
        ts_df = ts_df.rename(columns={col[0]: col[1] for col in cols if isinstance(col, tuple)})
        if bdates is not None:
            ts_df = ts_df.reindex(bdates)
        if drop_na_dates:
            ts_df = ts_df.dropna()

        return ts_df

    @staticmethod
    def _term_structure_plotter(
        term_structure_dict_df: Dict[datetime, pd.DataFrame],  # fwd/expiry tenors index, tails/underlying columns
        plot_title: str,
        x_axis_col_sorter_func: Callable,
        x_axis_col_sorter_func_args: Optional[Tuple[Any]] = (),
        x_axis_title: Optional[str] = "Tenor",
        y_axis_title: Optional[str] = "Yield",
        show_idx_tenor_in_legend: Optional[bool] = True,
        use_plotly: Optional[bool] = False,
    ):
        def format_dt(dt: datetime) -> str:
            if dt.tzinfo is not None:
                tzname = dt.tzinfo.tzname(dt)
                if tzname is None or tzname.startswith("UTC"):
                    dt = dt.astimezone(ZoneInfo("America/New_York"))
            if dt.time() == datetime.min.time():
                return dt.strftime("%Y-%m-%d %Z")
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        if use_plotly:
            fig = go.Figure()
            ref_df = next(iter(term_structure_dict_df.values()))
            x_computed = [x_axis_col_sorter_func(t, *x_axis_col_sorter_func_args) for t in ref_df.columns]
            x_ticktext = list(ref_df.columns)

            for dt, term_struct_df in term_structure_dict_df.items():
                formatted_dt = format_dt(dt)
                for tenor_idx in term_struct_df.index:
                    x_vals = [x_axis_col_sorter_func(t, *x_axis_col_sorter_func_args) for t in term_struct_df.columns]
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=term_struct_df.loc[tenor_idx],
                            mode="lines",
                            name=f"{formatted_dt} - {tenor_idx}" if show_idx_tenor_in_legend else f"{formatted_dt}",
                        )
                    )
            fig.update_layout(
                title=plot_title,
                xaxis_title=x_axis_title,
                yaxis_title=y_axis_title,
                font=dict(size=11),
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Curves:"),
                height=750,
            )
            fig.update_xaxes(
                tickvals=x_computed,
                ticktext=x_ticktext,
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikemode="across",
                showgrid=True,
            )
            fig.update_yaxes(
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikethickness=0.5,
                showgrid=True,
            )
            fig.show()

        else:
            plt.figure()
            ref_df = next(iter(term_structure_dict_df.values()))
            x_computed = [x_axis_col_sorter_func(t, *x_axis_col_sorter_func_args) for t in ref_df.columns]
            x_ticktext = list(ref_df.columns)

            for dt, term_struct_df in term_structure_dict_df.items():
                formatted_dt = format_dt(dt)
                for tenor_idx in term_struct_df.index:
                    x_vals = [x_axis_col_sorter_func(t, *x_axis_col_sorter_func_args) for t in term_struct_df.columns]
                    y_vals = term_struct_df.loc[tenor_idx].values
                    label = f"{formatted_dt} - {tenor_idx}" if show_idx_tenor_in_legend else f"{formatted_dt}"
                    plt.plot(x_vals, y_vals, marker="o", label=label)

            plt.title(plot_title, fontsize=14)
            plt.xlabel(x_axis_title, fontsize=12)
            plt.ylabel(y_axis_title, fontsize=12)
            plt.xticks(ticks=x_computed, labels=x_ticktext, rotation=45)
            plt.grid(True)
            plt.legend(title="Curves:", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
            plt.tight_layout()
            plt.show()

    @staticmethod
    def timeseries_df_plotter(
        df: pd.DataFrame,
        cols_to_plot: List[str],
        cols_to_plot_raxis: Optional[List[str]] = None,
        use_plotly: Optional[bool] = False,
        custom_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        yaxis_title_r: Optional[str] = None,
        stds: Optional[Dict[str, Tuple[List[int], datetime]]] = {},
        plot_zscores: Optional[bool] = False,
        entry_date: Optional[datetime] = None,
        entry_level: Optional[float] = None,
    ):
        assert isinstance(df.index, pd.DatetimeIndex), "The DataFrame must have a DatetimeIndex"

        df = df.copy()
        hover_template = "%{x|%Y-%m-%d %H:%M:%S}<br>%{y}<extra></extra>"

        if use_plotly:
            fig = go.Figure()
            for tenor in cols_to_plot:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[tenor] if not plot_zscores else zscore(df[tenor]),
                        mode="lines",
                        name=f"{tenor} (lhs)" if not plot_zscores else f"{tenor} Z-Score (lhs)",
                        yaxis="y1",
                        hovertemplate=hover_template,
                    )
                )
            if cols_to_plot_raxis:
                for tenor in cols_to_plot_raxis:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[tenor] if not plot_zscores else zscore(df[tenor]),
                            mode="lines",
                            name=f"{tenor} (rhs)" if not plot_zscores else f"{tenor} Z-Score (rhs)",
                            yaxis="y2",
                        )
                    )

            fig.update_layout(
                title=(
                    custom_title
                    or (
                        f"{", ".join(cols_to_plot)} (lhs) & {", ".join(cols_to_plot_raxis)} (rhs) Timeseries {"Z-Scores" if plot_zscores else ""}"
                        if cols_to_plot_raxis
                        else f"{", ".join(cols_to_plot)} Timeseries {"Z-Scores" if plot_zscores else ""}"
                    )
                ),
                xaxis_title="Date",
                yaxis=dict(title=yaxis_title if yaxis_title else ", ".join(cols_to_plot), side="left"),
                yaxis2=dict(
                    title=yaxis_title_r if yaxis_title_r else ", ".join(cols_to_plot_raxis) if cols_to_plot_raxis else None,
                    overlaying="y",
                    side="right",
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="Tenors"),
                template="plotly_dark",
                font=dict(size=11),
                height=750,
                newshape=dict(line=dict(color="red")),  # set the default drawing color to red
            )

            fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across", showgrid=True)
            fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5, showgrid=True)
            fig.show(
                config={
                    "modeBarButtonsToAdd": [
                        "drawline",
                        "drawopenpath",
                        "drawclosedpath",
                        "drawcircle",
                        "drawrect",
                        "eraseshape",
                    ]
                }
            )
        else:
            df = df.copy()
            date_col = df.index
            df = df.reset_index(drop=True)
            df["Date"] = date_col

            unique_colors = plt.cm.tab10.colors
            i = 0
            fig, ax_left = plt.subplots()

            lns = []
            for tenor in cols_to_plot:
                lns += ax_left.plot(
                    df["Date"],
                    df[tenor],
                    label=f"{tenor} (lhs)\nMost Recent: {df["Date"].iloc[-1].date()}, {np.round(df[tenor].iloc[-1], 3)}",
                    color=unique_colors[i],
                )
                i += 1

                if tenor in stds.keys():
                    ex_post_date = max(df["Date"])
                    if isinstance(stds[tenor], tuple):
                        ex_post_date = stds[tenor][1]

                    level_std = tstd(df[df["Date"] <= ex_post_date][tenor])
                    level_mean = np.mean(df[df["Date"] <= ex_post_date][tenor])
                    lns += ax_left.plot(
                        df[df["Date"] <= ex_post_date]["Date"],
                        [level_mean] * len(df[df["Date"] <= ex_post_date]["Date"]),
                        linestyle="--",
                        color="red",
                        label=f"Mean: {level_mean}",
                    )

                    std_vals = stds[tenor][0] if isinstance(stds[tenor], tuple) else stds[tenor]
                    for std in std_vals:
                        curr_std_level = level_mean + (level_std * std)
                        curr_std_level_opp = level_mean + (level_std * std * -1)
                        curr = ax_left.plot(
                            df[df["Date"] <= ex_post_date]["Date"],
                            [curr_std_level] * len(df[df["Date"] <= ex_post_date]["Date"]),
                            linestyle="--",
                            label=f"± {std} STD: {np.round(curr_std_level, 3)}, {np.round(curr_std_level_opp, 3)}",
                            color="red",
                            alpha=0.75,
                        )
                        lns += curr
                        ax_left.plot(
                            df[df["Date"] <= ex_post_date]["Date"],
                            [curr_std_level_opp] * len(df[df["Date"] <= ex_post_date]["Date"]),
                            linestyle="--",
                            color=curr[0].get_color(),
                            alpha=0.75,
                        )

            ax_left.set_ylabel(yaxis_title if yaxis_title else ", ".join(cols_to_plot))
            ax_left.tick_params(axis="y")

            if cols_to_plot_raxis:
                ax_right = ax_left.twinx()
                for tenor in cols_to_plot_raxis:
                    lns += ax_right.plot(
                        df["Date"],
                        df[tenor],
                        label=f"{tenor} (lhs)\nMost Recent: {df["Date"].iloc[-1].date()}, {np.round(df[tenor].iloc[-1], 3)}",
                        color=unique_colors[i],
                    )
                    i += 1

                    if tenor in stds.keys():
                        level_std = tstd(df[tenor])
                        level_mean = np.mean(df[tenor])
                        lns += ax_right.plot(df["Date"], [level_mean] * len(df["Date"]), linestyle="--", color="red", label=f"Mean: {level_mean}")

                        for std in stds[tenor]:
                            curr_std_level = level_mean + (level_std * std)
                            curr_std_level_opp = level_mean + (level_std * std * -1)
                            curr = ax_right.plot(
                                df["Date"],
                                [curr_std_level] * len(df["Date"]),
                                linestyle="--",
                                label=f"±{std} std: {np.round(curr_std_level, 3)}, {np.round(curr_std_level_opp, 3)}",
                                color="red",
                                alpha=0.75,
                            )
                            lns += curr
                            ax_right.plot(
                                df["Date"],
                                [curr_std_level_opp] * len(df["Date"]),
                                linestyle="--",
                                color=curr[0].get_color(),
                                alpha=0.75,
                            )

                ax_right.set_ylabel(yaxis_title_r if yaxis_title_r else ", ".join(cols_to_plot_raxis))
                ax_right.tick_params(axis="y")

            locator = mdates.AutoDateLocator(minticks=3, maxticks=15)
            formatter = mdates.DateFormatter("%Y-%m-%d")
            ax_left.xaxis.set_major_locator(locator)
            ax_left.xaxis.set_major_formatter(formatter)
            ax_left.set_xlabel("Date")
            ax_left.set_xticks(ax_left.get_xticks(), ax_left.get_xticklabels(), rotation=25, ha="right")

            labs = [l.get_label() for l in lns]
            ax_left.legend(lns, labs, loc=(0, 0))
            if custom_title:
                plt.title(custom_title)
            else:
                plt.title(
                    f"{' '.join(cols_to_plot)} (lhs) & {' '.join(cols_to_plot_raxis)} (rhs) Timeseries"
                    if cols_to_plot_raxis
                    else f"{' '.join(cols_to_plot)} Timeseries"
                )
            ax_left.grid(True)
            plt.xticks(rotation=25)
            fig.tight_layout()

            if entry_date and entry_level:
                plt.axvline(entry_date, label=f"Entry: {entry_date}, {entry_level}", c="r", linestyle="--", alpha=0.7)
                plt.axhline(entry_level, c="r", linestyle="--", alpha=0.7)
            elif entry_level:
                plt.axhline(entry_level, c="r", linestyle="--", alpha=0.7)

            plt.show()

    def _matplotlib_vol_surface_plotter(self, vol_surface_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, zlabel: str):
        X, Y = np.meshgrid(range(len(vol_surface_df.columns)), range(len(vol_surface_df.index)))
        Z = vol_surface_df.values
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_zlabel(zlabel, fontsize=12)
        ax.set_xticks(range(len(vol_surface_df.columns)))
        ax.set_xticklabels(vol_surface_df.columns, rotation=45, fontsize=10)
        ax.set_yticks(range(len(vol_surface_df.index)))
        ax.set_yticklabels(vol_surface_df.index, fontsize=10)
        fig.colorbar(surf, shrink=0.5, aspect=10, label="Nornmal Vol")
        plt.show()

    def _plotly_vol_surface_plotter(self, vol_surface_df: pd.DataFrame, title: str, xlabel: str, ylabel: str, zlabel: str):
        X, Y = np.meshgrid(range(len(vol_surface_df.columns)), range(len(vol_surface_df.index)))
        Z = vol_surface_df.values

        hover_template = f"{xlabel}: %{{x}}<br>" f"{ylabel}: %{{y}}<br>" f"{zlabel}: %{{z}}<extra></extra>"
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="RdYlGn_r", showscale=True, hovertemplate=hover_template)])
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title=xlabel, tickvals=list(range(len(vol_surface_df.columns))), ticktext=vol_surface_df.columns),
                yaxis=dict(title=ylabel, tickvals=list(range(len(vol_surface_df.index))), ticktext=vol_surface_df.index),
                zaxis=dict(title=zlabel),
                aspectratio={"x": 1, "y": 1, "z": 0.6},
            ),
            template="plotly_dark",
            height=750,
            width=1250,
        )
        fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
        fig.update_yaxes(
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikethickness=0.5,
        )
        fig.show()
