from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import QuantLib as ql
from sqlalchemy import Date, Engine, cast
from sqlalchemy.orm import sessionmaker

from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper
from core.Products.CurveBuilding.Swaptions.swaption_data_models import SABRParamsCacheBase, SCubeCacheBase, get_sabr_params_cache_model, get_scube_cache_model
from core.Products.CurveBuilding.Swaptions.SwaptionCubeBuilder import SwaptionCubeBuilder
from core.Products.CurveBuilding.Swaptions.types import SABRParams, SCube


class AlchemySwaptionCubeBuilderWrapper(AlchemyWrapper):
    _swaption_builder: SwaptionCubeBuilder = None
    _expiry_col: str = None

    def __init__(
        self,
        engine: Engine,
        date_col: Optional[str] = "Date",
        index_col: Optional[str] = None,
        timestamp_col_format: Optional[str] = "%Y-%m-%d %H:%M:%S%z",
        expiry_col: Optional[str] = "Expiry",
    ):
        super().__init__(engine=engine, date_col=date_col, index_col=index_col, timestamp_col_format=timestamp_col_format)

        self._swaption_builder = SwaptionCubeBuilder({}, {})
        self._expiry_col = expiry_col

    def return_scube(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
    ) -> Dict[datetime, SCube]:
        SCubeCache = get_scube_cache_model(curve=curve)
        Session = sessionmaker(bind=self._engine)
        session = Session()

        try:
            query = session.query(SCubeCache)
            if start_date and end_date:
                query = query.filter(
                    cast(SCubeCache.timestamp, Date) >= start_date.date(),
                    cast(SCubeCache.timestamp, Date) <= end_date.date(),
                )
            elif bdates:
                bdates_date = [d.date() for d in bdates]
                query = query.filter(cast(SCubeCache.timestamp, Date).in_(bdates_date))

            records: List[SCubeCacheBase] = query.all()
            scubes: Dict[datetime, SCube] = {}
            for record in records:
                ts = datetime.fromisoformat(record.timestamp)
                scube = {}
                for offset, grid_records in record.scube.items():
                    try:
                        offset_float = float(offset)
                    except ValueError:
                        continue
                    df = pd.DataFrame(grid_records).set_index("Expiry")
                    scube[offset_float] = df

                scubes[ts] = scube

            return scubes
        finally:
            session.close()

    def return_sabr_params(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
    ) -> Dict[datetime, SABRParams]:
        SABRParamsCache = get_sabr_params_cache_model(curve=curve)
        Session = sessionmaker(bind=self._engine)
        session = Session()

        try:
            query = session.query(SABRParamsCache)
            if start_date and end_date:
                query = query.filter(
                    cast(SABRParamsCache.timestamp, Date) >= start_date.date(),
                    cast(SABRParamsCache.timestamp, Date) <= end_date.date(),
                )
            elif bdates:
                bdates_date = [d.date() for d in bdates]
                query = query.filter(cast(SABRParamsCache.timestamp, Date).in_(bdates_date))

            records: List[SABRParamsCacheBase] = query.all()
            sabr_params: Dict[datetime, SABRParams] = {}
            for record in records:
                ts = datetime.fromisoformat(record.timestamp)
                sabr_params[ts] = record.sabr_params

            return sabr_params
        finally:
            session.close()

    def ql_vol_cube_builder_wrapper(
        self,
        curve: str,
        ql_discount_curves: Dict[datetime, ql.DiscountCurve],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        use_sabr: Optional[bool] = False,
        precalibrate_cube: Optional[bool] = False,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
        self._swaption_builder.set_timeseries(
            swaption_cube_timeseries_dict=self.return_scube(curve=curve, start_date=start_date, end_date=end_date, bdates=bdates),
            sabr_params_timeseries_dict=self.return_sabr_params(curve=curve, start_date=start_date, end_date=end_date, bdates=bdates),
        )
        return self._swaption_builder.parallel_ql_vol_cube_builder(
            curve=curve,
            ql_discount_curves=ql_discount_curves,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            use_sabr=use_sabr,
            precalibrate_cube=precalibrate_cube,
            enable_extrapolation=enable_extrapolation,
            show_tqdm=show_tqdm,
            tqdm_message=tqdm_message,
            n_jobs=n_jobs,
        )
