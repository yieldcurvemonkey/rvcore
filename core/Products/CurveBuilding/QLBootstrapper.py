from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import QuantLib as ql
import tqdm
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from core.Analytics.Interpolation.GeneralCurveInterpolator import GeneralCurveInterpolator
from core.Products.CurveBuilding.ql_curve_building import (
    build_piecewise_curve_from_cmt_scipy_spline,
    build_piecewise_curve_from_discount_factor_scipy_spline,
    build_piecewise_curve_from_zero_scipy_spline,
    build_piecewise_ql_discount_curve,
    build_ql_discount_curve,
    get_ql_cash_curve_params,
    get_ql_swaps_curve_params,
)
from core.utils.ql_utils import datetime_to_ql_date, get_bdates_between, ql_date_to_datetime


class QLBootstrapper:
    _timeseries_df: pd.DataFrame = None
    _timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame] = None

    def __init__(self, timeseries_df: pd.DataFrame, timeseries_df_grouper: Dict[datetime | pd.Timestamp, pd.DataFrame]):
        self._timeseries_df = timeseries_df
        if self._timeseries_df is not None:
            assert isinstance(self._timeseries_df.index, pd.DatetimeIndex), "self._timeseries_df index must be a DatetimeIndex"

        self._timeseries_df_grouper = timeseries_df_grouper

    def set_timeseries_df(self, timeseries_df: pd.DataFrame):
        self._timeseries_df = timeseries_df
        if self._timeseries_df is not None:
            assert isinstance(self._timeseries_df.index, pd.DatetimeIndex), "self._timeseries_df index must be a DatetimeIndex"

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

    def parallel_swap_curve_bootstrapper(
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
        ] = "pdf_log_linear",
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, ql.YieldTermStructure]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"
        assert self._timeseries_df is not None, "'timeseries_df' is required for swap curve bootstrapping"

        if not start_date and not end_date:
            start_date = min(bdates)
            end_date = max(bdates)

        timeseries_df = self._timeseries_df[(self._timeseries_df.index.date >= start_date.date()) & (self._timeseries_df.index.date <= end_date.date())]

        data_to_bootstrap = [(date, row.to_dict()) for date, row in timeseries_df.iterrows()]
        data_to_bootstrap_iter = tqdm.tqdm(data_to_bootstrap, desc=tqdm_message or "BOOTSTRAPPING CURVES...") if show_tqdm else data_to_bootstrap
        curves = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_single_swap_curve)(
                as_of_date=date,
                curve=curve,
                market_data=market_data,
                ql_interpolation_algo=f"pdf_{ql_interpolation_algo}",
                enable_extrapolation=enable_extrapolation,
            )
            for date, market_data in data_to_bootstrap_iter
        )
        parallel_results = {date: curve_obj for (date, _), curve_obj in zip(data_to_bootstrap, curves) if curve_obj is not None}

        params = get_ql_swaps_curve_params(curve_str=curve)
        _, _, day_counter, calendar, _, _, _, _ = params

        ql_curve_dict = {}
        for curve_date, discount_curve_nodes_dict in parallel_results.items():
            if not discount_curve_nodes_dict or len(discount_curve_nodes_dict) == 0:
                continue

            curve_date_norm = pd.to_datetime(curve_date).normalize()
            datetime_series = pd.Series(pd.to_datetime(list(discount_curve_nodes_dict.keys())).normalize())
            type_series = pd.Series(list(discount_curve_nodes_dict.values()))

            if not curve_date_norm in datetime_series.values:
                datetime_series = pd.concat([pd.Series([curve_date]), datetime_series])
                type_series = pd.concat([pd.Series([1]), type_series])

            ql_curve = build_ql_discount_curve(
                datetime_series,
                type_series,
                day_counter,
                calendar,
                f"df_{ql_interpolation_algo}",
            )
            if enable_extrapolation:
                ql_curve.enableExtrapolation()

            ql_curve_dict[curve_date] = ql_curve

        return ql_curve_dict

    def parallel_cash_curve_interpolator_builder(
        self,
        curve: str,
        curve_interpolator_func_str: str,
        curve_interpolator_kwargs: Dict,
        enable_extrapolation: Optional[bool] = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        x_col_name: Optional[str] = "time_to_maturity",
        y_col_name: Optional[str] = "eod_ytm",
        from_par_yields: Optional[bool] = False,
        from_zero_yields: Optional[bool] = False,
        from_dfs: Optional[bool] = False,
        n_jobs: Optional[int] = -1,
    ):
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"
        assert self._timeseries_df_grouper is not None, "'_timeseries_df_grouper' is required for cash curve bootstrapping"

        if not (from_par_yields or from_zero_yields or from_dfs):
            from_par_yields = True

        params = get_ql_cash_curve_params(curve_str=curve)

        between_bdates = (
            get_bdates_between(start_date=self.to_vanilla_pydt(start_date), end_date=self.to_vanilla_pydt(end_date), calendar=params[0])
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._timeseries_df_grouper.keys()), set(between_bdates)))
        valid_group_dates_iter = tqdm.tqdm(valid_group_dates, desc=tqdm_message or "FITTING PAR CURVES...") if show_tqdm else valid_group_dates

        ql_par_disc_curves: Dict[datetime, ql.DiscountCurve] = {}
        scipy_interpolated_curves: Dict[datetime, interp1d] = {}

        results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_single_cash_GeneralCurveInterpolator_curve)(
                dt,
                self._timeseries_df_grouper[dt],
                curve,
                curve_interpolator_func_str,
                curve_interpolator_kwargs,
                x_col_name,
                y_col_name,
                from_par_yields,
                from_zero_yields,
                from_dfs,
                enable_extrapolation,
            )
            for dt in valid_group_dates_iter
        )

        ql_par_disc_curves: Dict[datetime, ql.DiscountCurve] = {}
        scipy_interpolated_curves: Dict[datetime, interp1d] = {}
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
                params[3],
                params[0],
                "df_log_linear",
            )
            if enable_extrapolation:
                ql_curve.enableExtrapolation()

            ql_par_disc_curves[dt] = ql_curve
            scipy_interpolated_curves[dt] = fitted_scipy_func

        return ql_par_disc_curves, scipy_interpolated_curves


def _extract_fitted_curve_nodes(fitted_curve: ql.YieldTermStructure, num_points: Optional[int] = 250) -> Dict[datetime, float]:
    start_date = fitted_curve.referenceDate()
    end_date = fitted_curve.maxDate()
    dt_start = start_date.serialNumber()
    dt_end = end_date.serialNumber()
    grid_serials = np.linspace(dt_start, dt_end, num_points).astype(int)
    grid_dates = [ql.Date(int(s)) for s in grid_serials]
    return {ql_date_to_datetime(date): fitted_curve.discount(date) for date in grid_dates}


def _get_nodes_dict(
    ql_curve: ql.YieldTermStructure | ql.DiscountCurve | ql.PiecewiseLogLinearDiscount | ql.FittedBondDiscountCurve, to_ttm: Optional[bool] = False
) -> Dict[datetime, float]:
    if hasattr(ql_curve, "nodes"):
        nodes = ql_curve.nodes()
    elif hasattr(ql_curve, "dates") and hasattr(ql_curve, "discounts"):
        nodes = list(zip(ql_curve.dates(), ql_curve.discounts()))
    else:
        return _extract_fitted_curve_nodes(fitted_curve=ql_curve)

    if to_ttm:
        ref_date: ql.Date = ql_curve.referenceDate()
        day_counter: ql.DayCounter = ql_curve.dayCounter()
        nodes_dict = {day_counter.yearFraction(ref_date, node[0]): node[1] for node in nodes}
    else:
        nodes_dict = {ql_date_to_datetime(node[0]): node[1] for node in nodes}

    return nodes_dict


def bootstrap_single_swap_curve(
    as_of_date: datetime,
    curve: str,
    market_data: Dict[str, float],
    ql_interpolation_algo: Optional[
        List[
            Literal[
                "pdf_log_linear",
                "pdf_mono_log_cubic",
                "pdf_natural_cubic",
                "pdf_kruger_log",
                "pdf_natural_log_cubic",
                "pdf_log_mixed_linear",
                "pdf_log_parabolic_cubic",
                "pdf_spline_cubic_discount",
                "pdf_mono_log_parabolic_cubic",
            ]
        ]
    ] = "log_linear",
    enable_extrapolation: Optional[bool] = False,
) -> Dict[datetime, float]:
    try:
        ql.Settings.instance().evaluationDate = datetime_to_ql_date(as_of_date)

        params = get_ql_swaps_curve_params(curve_str=curve)
        ql_index_class, is_ois, day_counter, calendar, bdc, frequency, index_tenor, currency = params

        dummy_curve = ql.FlatForward(datetime_to_ql_date(as_of_date), 0.0, day_counter)
        dummy_handle = ql.YieldTermStructureHandle(dummy_curve)

        if is_ois:
            ql_floating_index = ql_index_class(dummy_handle)
            rate_helpers = [
                ql.OISRateHelper(2, ql.Period(term), ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)), ql_floating_index) for term, rate in market_data.items()
            ]
        else:
            ql_floating_index = ql.IborIndex(
                curve,
                index_tenor,
                2,
                currency,
                calendar,
                bdc,
                False,
                dummy_handle,
            )
            rate_helpers = [
                ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)), ql.Period(term), calendar, frequency, bdc, day_counter, ql_floating_index)
                for term, rate in market_data.items()
            ]

        ql_discount_curve = build_piecewise_ql_discount_curve(
            swap_rate_helpers=rate_helpers, ql_dc=day_counter, ql_cal=calendar, interpolation_algo=ql_interpolation_algo
        )
        if enable_extrapolation:
            ql_discount_curve.enableExtrapolation()

        return _get_nodes_dict(ql_discount_curve)

    except Exception as e:
        # TODO handle bootstrapping errors
        return None


def bootstrap_single_cash_GeneralCurveInterpolator_curve(
    dt: datetime,
    cash_df: pd.DataFrame,
    curve: str,
    curve_interpolator_func_str: str,
    curve_interpolator_kwargs: Dict,
    x_col_name: Optional[str] = "time_to_maturity",
    y_col_name: Optional[str] = "eod_ytm",
    from_par_yields: Optional[bool] = False,
    from_zero_yields: Optional[bool] = False,
    from_dfs: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True,
):
    try:
        curve_interpolater = GeneralCurveInterpolator(
            x=cash_df[x_col_name].to_numpy(),
            y=cash_df[y_col_name].to_numpy(),
            linspace_x_num=1000,
        )
        curve_interp_func = getattr(curve_interpolater, curve_interpolator_func_str)
        fitted_scipy_func = curve_interp_func(**curve_interpolator_kwargs)

        params = get_ql_cash_curve_params(curve_str=curve)

        if from_par_yields:
            ql_curve = build_piecewise_curve_from_cmt_scipy_spline(
                spline_func=fitted_scipy_func,
                anchor_date=dt,
                settlement_days=params[1],
                ql_coupon_frequency=params[2],
                ql_calendar=params[0],
                ql_business_convention=params[4],
                ql_day_counter=params[3],
                ql_interpolation_algo="pdf_log_linear",
                enable_extrapolation=enable_extrapolation,
            )
        elif from_zero_yields:
            ql_curve = build_piecewise_curve_from_zero_scipy_spline(
                spline_func=fitted_scipy_func,
                anchor_date=dt,
                settlement_days=params[1],
                ql_coupon_frequency=params[2],
                ql_calendar=params[0],
                ql_business_convention=params[4],
                ql_day_counter=params[3],
                ql_interpolation_algo="pdf_log_linear",
                enable_extrapolation=enable_extrapolation,
            )
        elif from_dfs:
            ql_curve = build_piecewise_curve_from_discount_factor_scipy_spline(
                spline_func=fitted_scipy_func,
                anchor_date=dt,
                settlement_days=params[1],
                ql_coupon_frequency=params[2],
                ql_calendar=params[0],
                ql_business_convention=params[4],
                ql_day_counter=params[3],
                ql_interpolation_algo="pdf_log_linear",
                enable_extrapolation=enable_extrapolation,
            )
        else:
            raise ValueError(f"Bad quantlib piecewise curve building from scipy func")

        return dt, _get_nodes_dict(ql_curve), fitted_scipy_func

    except Exception as e:
        # TODO handle bootstrapping errors
        return None, None, None
