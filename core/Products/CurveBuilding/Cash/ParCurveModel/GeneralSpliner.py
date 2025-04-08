from typing import Optional, Callable, Dict, Literal

import warnings
import pandas as pd
import QuantLib as ql

from core.Analytics.Interpolation.GeneralCurveInterpolator import GeneralCurveInterpolator
from core.Products.CurveBuilding.Cash.ParCurveModel.ParCurveModelBase import ParCurveModelBase
from core.Products.CurveBuilding.ql_curve_building_utils import build_piecewise_curve_from_cmt_scipy_spline


class GeneralSpliner(ParCurveModelBase):
    _custom_filter_func: Callable[[pd.DataFrame], pd.DataFrame] = None
    _GeneralCurveInterpolator_func: str = None
    _GeneralCurveInterpolator_kwargs: Dict[str, str] = None
    _spliner_x_col: str = None
    _spliner_y_col: str = None
    _linspace_x_num: int = None
    _timestamp_col: str = None

    def __init__(
        self,
        cusip_set_df: pd.DataFrame,
        custom_filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        GeneralCurveInterpolator_func: Literal[
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
        ],
        GeneralCurveInterpolator_kwags: Dict[str, str],
        spliner_x_col: Optional[str] = "time_to_maturity",
        spliner_y_col: Optional[str] = "eod_ytm",
        linspace_x_num: Optional[int] = 1000,
        timestamp_col: Optional[str] = "timestamp",
    ):
        self._cusip_set_df = cusip_set_df.copy()
        self._custom_filter_func = custom_filter_func
        self._GeneralCurveInterpolator_func = GeneralCurveInterpolator_func
        self._GeneralCurveInterpolator_kwargs = GeneralCurveInterpolator_kwags
        self._spliner_x_col = spliner_x_col
        self._spliner_y_col = spliner_y_col
        self._linspace_x_num = linspace_x_num
        self._timestamp_col = timestamp_col

    def _filter(self, cusip_set_df: pd.DataFrame) -> pd.DataFrame:
        try:
            return self._custom_filter_func(cusip_set_df)
        except Exception as e:
            self._logger.error(f"'GeneralSpliner._filter': custom filter function had an error: {e}")

        return cusip_set_df

    def fit(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            df = self._filter(cusip_set_df=self._cusip_set_df)
            self._filtered_cusip_set_df = df
            if df.empty:
                raise ValueError(f"'GeneralSpliner.fit': {self._FILTER_TOO_STRONG}")

            gci = GeneralCurveInterpolator(x=df[self._spliner_x_col].to_numpy(), y=df[self._spliner_y_col].to_numpy(), linspace_x_num=self._linspace_x_num)

            gci_func = getattr(gci, self._GeneralCurveInterpolator_func)
            fitted_scipy_func = gci_func(**self._GeneralCurveInterpolator_kwargs)

            self._scipy_par_spline_func = fitted_scipy_func
            self._ql_discount_curve = build_piecewise_curve_from_cmt_scipy_spline(
                spline_func=self._scipy_par_spline_func,
                anchor_date=df[self._timestamp_col].max(),
                ql_calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                ql_business_convention=ql.ModifiedFollowing,
                ql_day_counter=ql.ActualActual(ql.ActualActual.Actual365),
                ql_interpolation_algo="pdf_log_linear",
                enable_extrapolation=True,
            )
