from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd
import QuantLib as ql
import tqdm
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from core.Products.CurveBuilding.Cash.ParCurveModel.ParCurveModelBase import ParCurveModelBase
from core.Products.CurveBuilding.Cash.ParCurveModel.ParCurveModels import PAR_CURVE_MODELS
from core.Products.CurveBuilding.ql_curve_building_utils import build_ql_discount_curve, get_nodes_dict
from core.Products.CurveBuilding.ql_curve_params import GOVIE_CURVE_QL_PARAMS
from core.utils.ql_utils import get_bdates_between


def make_hashable(x):
    if isinstance(x, (tuple, list)):
        return tuple(make_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    elif isinstance(x, set):
        return tuple(sorted(make_hashable(e) for e in x))
    else:
        return x


class CashCurveBootstrapper:
    _timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame] = None
    _ql_cash_curve_cache: Dict[datetime, ql.YieldTermStructure] = None
    _scipy_cash_curve_cache: Dict[datetime, interp1d] = None
    _par_curve_models = PAR_CURVE_MODELS

    def __init__(self, timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame]):
        self._timeseries_df_grouper = timeseries_df_grouper
        self._ql_cash_curve_cache = {}
        self._scipy_cash_curve_cache = {}

    def set_timeseries_df_grouper(self, timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame]):
        self._timeseries_df_grouper = timeseries_df_grouper

    def to_vanilla_pydt(self, dt: datetime):
        return datetime(dt.year, dt.month, dt.day)

    def hoist_df(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        timestamp_col: Optional[str] = "timestamp",
        filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        df = df[(df[timestamp_col].dt.date >= start_date.date()) & (df[timestamp_col].dt.date <= end_date.date())]
        if filter_func is not None:
            return filter_func(df)
        return df

    def parallel_ql_cash_curve_bootstraper(
        self,
        curve: str,
        par_curve_model_key: Literal["general_spliner"],
        par_curve_model_kwags_ex_cusip_set: Dict,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, ql.DiscountCurve]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"
        assert self._timeseries_df_grouper is not None, "'_timeseries_df_grouper' is required for cash curve bootstrapping"

        between_bdates = (
            get_bdates_between(
                start_date=self.to_vanilla_pydt(start_date),
                end_date=self.to_vanilla_pydt(end_date),
                calendar=GOVIE_CURVE_QL_PARAMS[curve]["calendar"],
            )
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._timeseries_df_grouper.keys()), set(between_bdates)))

        cached_curves: Dict[datetime, ql.DiscountCurve] = {}
        dates_to_compute = []
        for dt in valid_group_dates:
            if dt in self._ql_cash_curve_cache:
                cached_curves[dt] = self._ql_cash_curve_cache[dt]
            else:
                dates_to_compute.append(dt)

        computed_curves: Dict[datetime, ql.DiscountCurve] = {}
        if dates_to_compute:
            dates_iter = tqdm.tqdm(dates_to_compute, desc=tqdm_message or "FITTING PAR QL CURVES...") if show_tqdm else dates_to_compute
            results = Parallel(n_jobs=n_jobs)(
                delayed(bootstrap_single_cash_curve)(
                    dt,
                    self._par_curve_models[par_curve_model_key](cusip_set_df=self._timeseries_df_grouper[dt], **par_curve_model_kwags_ex_cusip_set),
                )
                for dt in dates_iter
            )
            for result in results:
                if result is None:
                    continue
                dt, discount_curve_nodes_dict, fitted_scipy_func = result
                if dt is None or discount_curve_nodes_dict is None or fitted_scipy_func is None:
                    continue

                datetime_series = pd.Series(pd.to_datetime(list(discount_curve_nodes_dict.keys())).normalize())
                type_series = pd.Series(list(discount_curve_nodes_dict.values()))

                ql_curve = build_ql_discount_curve(
                    datetime_series,
                    type_series,
                    GOVIE_CURVE_QL_PARAMS[curve]["dayCounter"],
                    GOVIE_CURVE_QL_PARAMS[curve]["calendar"],
                    interpolation_algo="df_log_linear",
                )
                if enable_extrapolation:
                    ql_curve.enableExtrapolation()

                computed_curves[dt] = ql_curve

            self._ql_cash_curve_cache.update(computed_curves)

        all_curves = {**cached_curves, **computed_curves}
        return all_curves

    def parallel_scipy_cash_curve_bootstraper(
        self,
        curve: str,
        par_curve_model_key: Literal["general_spliner"],
        par_curve_model_kwags_ex_cusip_set: Dict,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, interp1d]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"
        assert self._timeseries_df_grouper is not None, "'_timeseries_df_grouper' is required for cash curve bootstrapping"

        between_bdates = (
            get_bdates_between(
                start_date=self.to_vanilla_pydt(start_date),
                end_date=self.to_vanilla_pydt(end_date),
                calendar=GOVIE_CURVE_QL_PARAMS[curve]["calendar"],
            )
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._timeseries_df_grouper.keys()), set(between_bdates)))

        cached_funcs: Dict[datetime, interp1d] = {}
        dates_to_compute = []
        for dt in valid_group_dates:
            if dt in self._scipy_cash_curve_cache:
                cached_funcs[dt] = self._scipy_cash_curve_cache[dt]
            else:
                dates_to_compute.append(dt)

        computed_funcs: Dict[datetime, interp1d] = {}
        if dates_to_compute:
            dates_iter = tqdm.tqdm(dates_to_compute, desc=tqdm_message or "FITTING PAR SCIPY SPLINES...") if show_tqdm else dates_to_compute
            results = Parallel(n_jobs=n_jobs)(
                delayed(bootstrap_single_cash_curve)(
                    dt,
                    self._par_curve_models[par_curve_model_key](cusip_set_df=self._timeseries_df_grouper[dt], **par_curve_model_kwags_ex_cusip_set),
                )
                for dt in dates_iter
            )
            for result in results:
                if result is None:
                    continue
                dt, discount_curve_nodes_dict, fitted_scipy_func = result
                if dt is None or discount_curve_nodes_dict is None or fitted_scipy_func is None:
                    continue

                computed_funcs[dt] = fitted_scipy_func

            self._scipy_cash_curve_cache.update(computed_funcs)

        all_funcs = {**cached_funcs, **computed_funcs}
        return all_funcs


def bootstrap_single_cash_curve(curve_dt: datetime, unfitted_par_curve_model: ParCurveModelBase):
    try:
        unfitted_par_curve_model.fit()
        return curve_dt, get_nodes_dict(unfitted_par_curve_model._ql_discount_curve), unfitted_par_curve_model._scipy_par_spline_func
    except Exception as e:
        return None, None, None
