from datetime import datetime
from typing import List, Literal, Optional, Dict

import numpy as np
import pandas as pd
import QuantLib as ql

from core.utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime


def build_ql_discount_curve(
    datetime_series: pd.Series,
    discount_factor_series: pd.Series,
    ql_dc: ql.DayCounter,
    ql_cal: ql.Calendar,
    interpolation_algo: Optional[
        Literal[
            "df_log_linear",
            "df_mono_log_cubic",
            "df_natural_cubic",
            "df_kruger_log",
            "df_natural_log_cubic",
            "df_log_mixed_linear",
            "df_log_parabolic_cubic",
            "df_mono_log_parabolic_cubic",
        ]
    ] = "df_log_linear",
) -> ql.DiscountCurve:
    curve_mapping = {
        "df_log_linear": ql.DiscountCurve,
        "df_mono_log_cubic": ql.MonotonicLogCubicDiscountCurve,
        "df_natural_cubic": ql.NaturalCubicDiscountCurve,
        "df_kruger_log": ql.KrugerLogDiscountCurve,
        "df_natural_log_cubic": ql.NaturalLogCubicDiscountCurve,
        "df_log_mixed_linear": ql.LogMixedLinearCubicDiscountCurve,
        "df_log_parabolic_cubic": ql.LogParabolicCubicDiscountCurve,
        "df_mono_log_parabolic_cubic": ql.MonotonicLogParabolicCubicDiscountCurve,
    }
    try:
        curve_class = curve_mapping[interpolation_algo]
    except KeyError:
        raise ValueError(f"QuantLib has no discount curve with {interpolation_algo} interpolation")

    ql_dates = datetime_series.apply(datetime_to_ql_date).to_list()
    discount_factors = discount_factor_series.to_list()

    return curve_class(ql_dates, discount_factors, ql_dc, ql_cal)


def build_piecewise_ql_discount_curve(
    swap_rate_helpers: List[ql.RateHelper],
    ql_dc: ql.DayCounter,
    ql_cal: ql.Calendar,
    settlement_day: int,
    interpolation_algo: Optional[
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
    ] = "pdf_log_linear",
) -> ql.DiscountCurve:
    curve_mapping = {
        "pdf_log_linear": ql.PiecewiseLogLinearDiscount,
        "pdf_mono_log_cubic": ql.PiecewiseLogCubicDiscount,
        "pdf_natural_cubic": ql.PiecewiseNaturalCubicZero,
        "pdf_kruger_log": ql.PiecewiseKrugerLogDiscount,
        "pdf_natural_log_cubic": ql.PiecewiseNaturalLogCubicDiscount,
        "pdf_log_mixed_linear": ql.PiecewiseLogMixedLinearCubicDiscount,
        "pdf_log_parabolic_cubic": ql.PiecewiseLogParabolicCubicDiscount,
        "pdf_spline_cubic_discount": ql.PiecewiseSplineCubicDiscount,
        "pdf_mono_log_parabolic_cubic": ql.PiecewiseMonotonicLogParabolicCubicDiscount,
    }
    try:
        curve_class = curve_mapping[interpolation_algo]
    except KeyError:
        raise ValueError(f"QuantLib has no discount curve with {interpolation_algo} interpolation")

    return curve_class(settlement_day, ql_cal, swap_rate_helpers, ql_dc)


def build_ql_zero_curve(
    datetime_series: pd.Series,
    zero_rate_series: pd.Series,
    ql_dc: ql.DayCounter,
    ql_cal: ql.Calendar,
    interpolation_algo: Optional[
        Literal[
            "z_linear",
            "z_log_linear",
            "z_cubic",
            "z_natural_cubic",
            "z_log_cubic",
            "z_monotonic_cubic",
            "z_kruger",
            "z_parabolic_cubic",
            "z_monotonic_parabolic_cubic",
        ]
    ] = "z_log_linear",
) -> ql.ZeroCurve:
    curve_mapping = {
        "z_linear": ql.ZeroCurve,
        "z_log_linear": ql.LogLinearZeroCurve,
        "z_cubic": ql.CubicZeroCurve,
        "z_natural_cubic": ql.NaturalCubicZeroCurve,
        "z_log_cubic": ql.LogCubicZeroCurve,
        "z_monotonic_cubic": ql.MonotonicCubicZeroCurve,
        "z_kruger": ql.KrugerZeroCurve,
        "z_parabolic_cubic": ql.ParabolicCubicZeroCurve,
        "z_monotonic_parabolic_cubic": ql.MonotonicParabolicCubicZeroCurve,
    }
    try:
        curve_class = curve_mapping[interpolation_algo]
    except KeyError:
        raise ValueError(f"QuantLib has no zero curve with {interpolation_algo} interpolation.")

    ql_dates = datetime_series.apply(datetime_to_ql_date).to_list()
    rates = zero_rate_series.to_list()
    return curve_class(ql_dates, rates, ql_dc, ql_cal)


def extract_fitted_curve_nodes(fitted_curve: ql.YieldTermStructure, num_points: Optional[int] = 250) -> Dict[datetime, float]:
    start_date = fitted_curve.referenceDate()
    end_date = fitted_curve.maxDate()
    dt_start = start_date.serialNumber()
    dt_end = end_date.serialNumber()
    grid_serials = np.linspace(dt_start, dt_end, num_points).astype(int)
    grid_dates = [ql.Date(int(s)) for s in grid_serials]
    return {ql_date_to_datetime(date): fitted_curve.discount(date) for date in grid_dates}


def get_nodes_dict(
    ql_curve: ql.YieldTermStructure | ql.DiscountCurve | ql.PiecewiseLogLinearDiscount | ql.FittedBondDiscountCurve, to_ttm: Optional[bool] = False
) -> Dict[datetime, float]:
    if hasattr(ql_curve, "nodes"):
        nodes = ql_curve.nodes()
    elif hasattr(ql_curve, "dates") and hasattr(ql_curve, "discounts"):
        nodes = list(zip(ql_curve.dates(), ql_curve.discounts()))
    else:
        return extract_fitted_curve_nodes(fitted_curve=ql_curve)

    if to_ttm:
        ref_date: ql.Date = ql_curve.referenceDate()
        day_counter: ql.DayCounter = ql_curve.dayCounter()
        nodes_dict = {day_counter.yearFraction(ref_date, node[0]): node[1] for node in nodes}
    else:
        nodes_dict = {ql_date_to_datetime(node[0]): node[1] for node in nodes}

    return nodes_dict


def get_fixings_dict(swap_index: ql.SwapIndex) -> Dict[datetime, float]:
    ts = swap_index.timeSeries()
    return {ql_date_to_datetime(d): v for d, v in zip(ts.dates(), ts.values()) if not np.isnan(v)}
