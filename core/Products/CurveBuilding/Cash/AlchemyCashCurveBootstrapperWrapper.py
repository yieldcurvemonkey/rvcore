from datetime import datetime
from typing import Dict, List, Literal, Optional

import pandas as pd
import QuantLib as ql
from sqlalchemy import Engine

from core.Products.CurveBuilding.AlchemyWrapper import AlchemyWrapper
from core.Products.CurveBuilding.Cash.CashCurveBootstrapper import CashCurveBootstrapper


class AlchemyCashCurveBootstrapperWrapper(AlchemyWrapper):
    _ql_bootstrapper: CashCurveBootstrapper = None
    _bootstrap_cols: List[str] = None
    _timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame]

    def __init__(
        self,
        engine: Engine,
        date_col: Optional[str] = "Date",
        index_col: Optional[str] = None,
        timestamp_col_format: Optional[str] = "%Y-%m-%d %H:%M:%S%z",
    ):
        super().__init__(engine=engine, date_col=date_col, index_col=index_col, timestamp_col_format=timestamp_col_format)

        self._ql_bootstrapper = CashCurveBootstrapper(None)

    def ql_cash_curve_bootstrap_wrapper(
        self,
        curve: str,
        par_curve_model_key: Literal["KEYS OF 'PAR_CURVE_MODELS'"],
        par_curve_model_kwags_ex_cusip_set: Dict,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        utc: Optional[bool] = False,
    ) -> Dict[datetime, ql.DiscountCurve]:
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
                ): group
                for key, group in timeseries_df.groupby(self._date_col)
            }
        )

        return self._ql_bootstrapper.parallel_ql_cash_curve_bootstraper(
            curve=curve,
            par_curve_model_key=par_curve_model_key,
            par_curve_model_kwags_ex_cusip_set=par_curve_model_kwags_ex_cusip_set,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            enable_extrapolation=enable_extrapolation,
            show_tqdm=show_tqdm,
            tqdm_message=tqdm_message,
            n_jobs=n_jobs,
        )

    def scipy_cash_curve_bootstrap_wrapper(
        self,
        curve: str,
        par_curve_model_key: Literal["KEYS OF 'PAR_CURVE_MODELS'"],
        par_curve_model_kwags_ex_cusip_set: Dict,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
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
                ): group
                for key, group in timeseries_df.groupby(self._date_col)
            }
        )

        return self._ql_bootstrapper.parallel_scipy_cash_curve_bootstraper(
            curve=curve,
            par_curve_model_key=par_curve_model_key,
            par_curve_model_kwags_ex_cusip_set=par_curve_model_kwags_ex_cusip_set,
            start_date=start_date,
            end_date=end_date,
            bdates=bdates,
            show_tqdm=show_tqdm,
            tqdm_message=tqdm_message,
            n_jobs=n_jobs,
        )
