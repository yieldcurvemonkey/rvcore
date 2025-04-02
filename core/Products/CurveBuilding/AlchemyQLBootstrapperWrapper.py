from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d
from sqlalchemy import Column, Engine, String, cast, inspect, text, Date
from sqlalchemy.dialects.postgresql import HSTORE
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.Products.CurveBuilding.QLBootstrapper import QLBootstrapper
from core.Products.CurveBuilding.ql_curve_building import build_ql_discount_curve, get_ql_swaps_curve_params
from core.utils.ql_utils import get_bdates_between


class AlchemyQLBootstrapperWrapper:
    _engine: Engine = None
    _ql_bootstrapper: QLBootstrapper = None
    _date_col: str = None
    _index_col: str = None
    _bootstrap_cols: List[str] = None
    _timestamp_col_format: str = None

    _ql_cash_curves_cache: Dict[datetime, ql.DiscountCurve] = None
    _scipy_cash_splines_cache: Dict[datetime, interp1d] = None

    def __init__(
        self,
        engine: Engine,
        date_col: Optional[str] = "Date",
        index_col: Optional[str] = None,
        bootstrap_cols: Optional[str] = None,
        timestamp_col_format: Optional[str] = "%Y-%m-%d %H:%M:%S%z",
    ):
        self._ql_cash_curves_cache = {}
        self._scipy_cash_splines_cache = {}

        self._engine = engine
        self._date_col = date_col
        self._index_col = index_col
        if not self._index_col:
            self._index_col = self._date_col
        self._ql_bootstrapper = QLBootstrapper(pd.DataFrame([], index=pd.DatetimeIndex([])), None)
        self._bootstrap_cols = bootstrap_cols
        self._timestamp_col_format = timestamp_col_format

    def fetch_latest_row(self, table_name: str, cols: Optional[List[str]] = None, set_index: Optional[bool] = True, utc: Optional[bool] = False) -> pd.DataFrame:
        selected_columns = ", ".join(cols) if cols else "*"
        table_identifier = f'"{table_name}"'
        date_column = f'"{self._date_col}"'

        query = text(f"SELECT {selected_columns} FROM {table_identifier} ORDER BY {date_column} DESC LIMIT 1")
        df = pd.read_sql_query(query, self._engine)

        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", format=self._timestamp_col_format, utc=utc)
        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df

    def _fetch_df_by_dates(
        self,
        table_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        cols: Optional[List[str]] = None,
        set_index: Optional[bool] = True,
        utc: Optional[bool] = False,
    ) -> pd.DataFrame:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        selected_columns = ", ".join(cols) if cols else "*"
        table_identifier = f'"{table_name}"'
        date_column = f'"{self._date_col}"'

        if start_date and end_date:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            query = text(f"SELECT {selected_columns} FROM {table_identifier} WHERE CAST({date_column} AS DATE) BETWEEN :start_date AND :end_date")
            params = {"start_date": start_str, "end_date": end_str}
        else:
            placeholders = []
            params = {}
            for i, d in enumerate(bdates):
                key = f"date_{i}"
                placeholders.append(f":{key}")
                params[key] = d.strftime("%Y-%m-%d")
            in_clause = ", ".join(placeholders)
            query = text(f"SELECT {selected_columns} FROM {table_identifier} " f"WHERE CAST({date_column} AS DATE) IN ({in_clause})")

        df = pd.read_sql_query(query, self._engine, params=params)
        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", format=self._timestamp_col_format, utc=utc)
        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df

    def _fetch_by_col_values(
        self,
        table_name: str,
        search_params_dict: Dict[str, List[str]],
        set_index: Optional[bool] = True,
        and_or: Literal["and", "or"] = "and",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        cols_to_look: Optional[List[str]] = None,
        utc: Optional[bool] = False,
    ) -> pd.DataFrame:
        if len(search_params_dict) == 0 and start_date and end_date and cols_to_look is None:
            return self._fetch_df_by_dates(table_name=table_name, start_date=start_date, end_date=end_date, set_index=set_index, utc=utc)

        if cols_to_look is not None:
            cols_to_look = list(set(cols_to_look))
            selected_columns = ", ".join([f'"{col}"' for col in cols_to_look])
        else:
            selected_columns = "*"

        table_identifier = f'"{table_name}"'

        conditions = []
        params = {}
        for col, values in search_params_dict.items():
            if values:
                placeholders = []
                for i, val in enumerate(values):
                    key = f"{col}_val_{i}"
                    placeholders.append(f":{key}")
                    params[key] = val
                col_identifier = f'"{col}"'
                conditions.append(f"{col_identifier} IN ({', '.join(placeholders)})")

        joiner = " AND " if and_or.lower() == "and" else " OR "
        if conditions:
            final_where_clause = " WHERE " + joiner.join(f"({cond})" for cond in conditions)
        else:
            final_where_clause = ""

        query = text(f"SELECT {selected_columns} FROM {table_identifier}{final_where_clause}")
        df = pd.read_sql_query(query, self._engine, params=params)

        if self._date_col in df.columns:
            df[self._date_col] = pd.to_datetime(df[self._date_col], errors="coerce", format=self._timestamp_col_format, utc=utc)
            if start_date:
                df = df[df[self._date_col].dt.date >= start_date.date()]
            if end_date:
                df = df[df[self._date_col].dt.date <= end_date.date()]
            if bdates:
                df = df[df[self._date_col].dt.date.isin([dt.date() for dt in bdates])]

        if set_index:
            df.set_index(self._index_col, inplace=True)

        return df

    def ql_swap_curve_bootstrap_wrapper(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
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
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        utc: Optional[bool] = False,
    ) -> Dict[datetime, ql.DiscountCurve]:
        ql_curves: Dict[datetime, ql.DiscountCurve] = {}
        params = get_ql_swaps_curve_params(curve_str=curve)
        _, _, ql_day_counter, ql_calendar, _, _, _, _ = params

        hstore_table_name = f"{curve}_{ql_interpolation_algo}_ql_cache_nodes"
        inspector = inspect(self._engine)
        if inspector.has_table(hstore_table_name):
            Base = declarative_base()

            class QLCurveCache(Base):
                __tablename__ = hstore_table_name
                timestamp = Column(String, primary_key=True)
                nodes: Dict = Column(HSTORE)

            Session = sessionmaker(bind=self._engine)
            session = Session()

            try:
                query = session.query(QLCurveCache)
                if start_date and end_date:
                    query = query.filter(
                        cast(QLCurveCache.timestamp, Date) >= start_date.date(),
                        cast(QLCurveCache.timestamp, Date) <= end_date.date(),
                    )
                elif bdates:
                    bdates_date = [d.date() for d in bdates]
                    query = query.filter(cast(QLCurveCache.timestamp, Date).in_(bdates_date))

                records: List[QLCurveCache] = query.all()

                for record in records:
                    ql_curve = build_ql_discount_curve(
                        datetime_series=pd.Series(pd.to_datetime(list(record.nodes.keys()), format="%Y-%m-%dT%H:%M:%S", errors="coerce")),
                        discount_factor_series=pd.Series(pd.to_numeric(list(record.nodes.values()), errors="coerce")),
                        ql_dc=ql_day_counter,
                        ql_cal=ql_calendar,
                        interpolation_algo=f"df_{ql_interpolation_algo}",
                    )
                    if enable_extrapolation:
                        ql_curve.enableExtrapolation()

                    ql_curves[pd.to_datetime(record.timestamp, utc=True)] = ql_curve

            finally:
                session.close()

        if bdates:
            provided_dates = pd.to_datetime(bdates).normalize()
        elif start_date and end_date:
            provided_dates = pd.to_datetime(get_bdates_between(start_date=start_date, end_date=end_date, calendar=ql_calendar)).normalize()
        else:
            provided_dates = []

        cached_dates = {dt.date() for dt in ql_curves.keys()}
        missing_dates = [d for d in provided_dates if d.date() not in cached_dates]

        if missing_dates:
            new_start_date = min(missing_dates)
            new_end_date = max(missing_dates)

            timeseries_df = self._fetch_by_col_values(
                table_name=curve,
                search_params_dict={},
                start_date=new_start_date,
                end_date=new_end_date,
                bdates=missing_dates,
                cols_to_look=[self._date_col] + self._bootstrap_cols,
                utc=utc,
            )
            self._ql_bootstrapper.set_timeseries_df(timeseries_df=timeseries_df)
            new_curves = self._ql_bootstrapper.parallel_swap_curve_bootstrapper(
                curve=curve,
                start_date=new_start_date,
                end_date=new_end_date,
                bdates=missing_dates,
                ql_interpolation_algo=ql_interpolation_algo,
                enable_extrapolation=enable_extrapolation,
                show_tqdm=show_tqdm,
                tqdm_message=tqdm_message,
                n_jobs=n_jobs,
            )
            ql_curves |= new_curves

        return ql_curves

    def _run_and_cache_parallel_cash_curve_interpolator_builder(
        self,
        curve: str,
        par_curve_filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        curve_interpolator_func_str: Optional[
            Literal[
                "linear_interpolation",
                "log_linear_interpolation",
                "cubic_spline_interpolation",
                "cubic_hermite_interpolation",
                "pchip_interpolation",
                "akima_interpolation",
                "b_spline1_interpolation",
                "b_spline_with_knots_interpolation",
                "univariate_spline",
                "ppoly_interpolation",
                "monotone_convex",
                "smoothing_spline",
                "lsq_univariate_soline",
                "loess_interpolation",
            ]
        ],
        curve_interpolator_kwargs: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        scipy_x_col_name: Optional[str] = "time_to_maturity",
        scipy_y_col_name: Optional[str] = "eod_ytm",
        scipy_from_par_yields: Optional[bool] = False,
        scipy_from_zero_yields: Optional[bool] = False,
        scipy_from_dfs: Optional[bool] = False,
        utc: Optional[bool] = False,
    ):
        timeseries_df = self._fetch_df_by_dates(
            table_name=curve,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            utc=utc,
        )
        self._ql_bootstrapper.set_timeseries_df_grouper(
            timeseries_df_grouper={
                (
                    self._ql_bootstrapper.to_vanilla_pydt(key.tz_localize(None).to_pydatetime())
                    if key.tzinfo is not None
                    else self._ql_bootstrapper.to_vanilla_pydt(key.to_pydatetime())
                ): par_curve_filter_func(group)
                for key, group in timeseries_df.groupby(self._date_col)
            }
        )

        needed_dates = list(self._ql_bootstrapper._timeseries_df_grouper.keys())
        missing_dates = [d for d in needed_dates if d not in self._ql_cash_curves_cache or d not in self._scipy_cash_splines_cache]
        if missing_dates:
            new_ql, new_scipy = self._ql_bootstrapper.parallel_cash_curve_interpolator_builder(
                curve=curve,
                curve_interpolator_func_str=curve_interpolator_func_str,
                curve_interpolator_kwargs=curve_interpolator_kwargs,
                start_date=start_date,
                end_date=end_date,
                bdates=missing_dates,
                show_tqdm=show_tqdm,
                tqdm_message=tqdm_message,
                x_col_name=scipy_x_col_name,
                y_col_name=scipy_y_col_name,
                from_par_yields=scipy_from_par_yields,
                from_zero_yields=scipy_from_zero_yields,
                from_dfs=scipy_from_dfs,
                n_jobs=n_jobs,
                enable_extrapolation=enable_extrapolation,
            )
            for d in missing_dates:
                if d in new_ql:
                    self._ql_cash_curves_cache[d] = new_ql[d]
                if d in new_scipy:
                    self._scipy_cash_splines_cache[d] = new_scipy[d]

        return needed_dates

    def ql_cash_curve_bootstrap_wrapper(
        self,
        curve: str,
        par_curve_filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        curve_interpolator_func_str: Optional[
            Literal[
                "linear_interpolation",
                "log_linear_interpolation",
                "cubic_spline_interpolation",
                "cubic_hermite_interpolation",
                "pchip_interpolation",
                "akima_interpolation",
                "b_spline1_interpolation",
                "b_spline_with_knots_interpolation",
                "univariate_spline",
                "ppoly_interpolation",
                "monotone_convex",
                "smoothing_spline",
                "lsq_univariate_soline",
                "loess_interpolation",
            ]
        ],
        curve_interpolator_kwargs: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        scipy_x_col_name: Optional[str] = "time_to_maturity",
        scipy_y_col_name: Optional[str] = "eod_ytm",
        scipy_from_par_yields: Optional[bool] = False,
        scipy_from_zero_yields: Optional[bool] = False,
        scipy_from_dfs: Optional[bool] = False,
        utc: Optional[bool] = False,
    ):
        needed_dates = self._run_and_cache_parallel_cash_curve_interpolator_builder(
            curve=curve,
            par_curve_filter_func=par_curve_filter_func,
            curve_interpolator_func_str=curve_interpolator_func_str,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            enable_extrapolation=enable_extrapolation,
            show_tqdm=show_tqdm,
            tqdm_message=tqdm_message,
            n_jobs=n_jobs,
            curve_interpolator_kwargs=curve_interpolator_kwargs,
            scipy_x_col_name=scipy_x_col_name,
            scipy_y_col_name=scipy_y_col_name,
            scipy_from_par_yields=scipy_from_par_yields,
            scipy_from_zero_yields=scipy_from_zero_yields,
            scipy_from_dfs=scipy_from_dfs,
            utc=utc,
        )

        final_ql = {d: self._ql_cash_curves_cache[d] for d in needed_dates if d in self._ql_cash_curves_cache}
        return final_ql

    def scipy_cash_curve_bootstrap_wrapper(
        self,
        curve: str,
        par_curve_filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        curve_interpolator_func_str: Optional[
            Literal[
                "linear_interpolation",
                "log_linear_interpolation",
                "cubic_spline_interpolation",
                "cubic_hermite_interpolation",
                "pchip_interpolation",
                "akima_interpolation",
                "b_spline1_interpolation",
                "b_spline_with_knots_interpolation",
                "univariate_spline",
                "ppoly_interpolation",
                "monotone_convex",
                "smoothing_spline",
                "lsq_univariate_soline",
                "loess_interpolation",
            ]
        ],
        curve_interpolator_kwargs: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        scipy_x_col_name: Optional[str] = "time_to_maturity",
        scipy_y_col_name: Optional[str] = "eod_ytm",
        scipy_from_par_yields: Optional[bool] = False,
        scipy_from_zero_yields: Optional[bool] = False,
        scipy_from_dfs: Optional[bool] = False,
        utc: Optional[bool] = False,
    ) -> Dict[datetime, interp1d]:
        needed_dates = self._run_and_cache_parallel_cash_curve_interpolator_builder(
            curve=curve,
            par_curve_filter_func=par_curve_filter_func,
            curve_interpolator_func_str=curve_interpolator_func_str,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            enable_extrapolation=enable_extrapolation,
            show_tqdm=show_tqdm,
            tqdm_message=tqdm_message,
            n_jobs=n_jobs,
            curve_interpolator_kwargs=curve_interpolator_kwargs,
            scipy_x_col_name=scipy_x_col_name,
            scipy_y_col_name=scipy_y_col_name,
            scipy_from_par_yields=scipy_from_par_yields,
            scipy_from_zero_yields=scipy_from_zero_yields,
            scipy_from_dfs=scipy_from_dfs,
            utc=utc,
        )

        final_scipy = {d: self._scipy_cash_splines_cache[d] for d in needed_dates if d in self._scipy_cash_splines_cache}
        return final_scipy
