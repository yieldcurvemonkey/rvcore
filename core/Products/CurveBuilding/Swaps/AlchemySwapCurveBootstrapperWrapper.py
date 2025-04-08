from datetime import datetime
from typing import Dict, List, Literal, Optional

import pandas as pd
import QuantLib as ql
from sqlalchemy import Date, Engine, cast, inspect
from sqlalchemy.orm import sessionmaker

from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper
from core.Products.CurveBuilding.ql_curve_building_utils import build_ql_discount_curve
from core.Products.CurveBuilding.ql_curve_params import CME_SWAP_CURVE_QL_PARAMS
from core.Products.CurveBuilding.Swaps.swap_data_models import QLCurveCacheBase, get_ql_curve_cache_model
from core.Products.CurveBuilding.Swaps.SwapCurveBootstrapper import SwapCurveBootstrapper
from core.utils.ql_utils import get_bdates_between



class AlchemySwapCurveBootstrapperWrapper(AlchemyWrapper):
    _ql_bootstrapper: SwapCurveBootstrapper = None
    _bootstrap_cols: List[str] = None

    def __init__(
        self,
        engine: Engine,
        date_col: Optional[str] = "Date",
        index_col: Optional[str] = None,
        bootstrap_cols: Optional[str] = None,
        timestamp_col_format: Optional[str] = "%Y-%m-%d %H:%M:%S%z",
    ):
        super().__init__(engine=engine, date_col=date_col, index_col=index_col, timestamp_col_format=timestamp_col_format)

        self._bootstrap_cols = bootstrap_cols
        self._ql_bootstrapper = SwapCurveBootstrapper(pd.DataFrame([], index=pd.DatetimeIndex([])))

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

        QLCurveCache = get_ql_curve_cache_model(curve, ql_interpolation_algo)

        inspector = inspect(self._engine)
        if inspector.has_table(QLCurveCache.__tablename__):
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

                records: List[QLCurveCacheBase] = query.all()

                for record in records:
                    ql_curve = build_ql_discount_curve(
                        datetime_series=pd.Series(pd.to_datetime(list(record.nodes.keys()), format="%Y-%m-%dT%H:%M:%S", errors="coerce")),
                        discount_factor_series=pd.Series(pd.to_numeric(list(record.nodes.values()), errors="coerce")),
                        ql_dc=CME_SWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                        ql_cal=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
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
            provided_dates = pd.to_datetime(
                get_bdates_between(start_date=start_date, end_date=end_date, calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"])
            ).normalize()
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
            new_curves = self._ql_bootstrapper.parallel_ql_swap_curve_bootstrapper(
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
