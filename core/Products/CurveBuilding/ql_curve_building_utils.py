from datetime import datetime
from typing import List, Literal, Optional, Callable, Dict

import numpy as np
import pandas as pd
import QuantLib as ql

from core.utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime


def build_ql_discount_curve(
    datetime_series: pd.Series,
    discount_factor_series: pd.Series,
    ql_dc: Optional[ql.DayCounter] = ql.Actual360(),
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
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
) -> ql.YieldTermStructure:
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
    swap_rate_helpers: List[ql.SwapRateHelper],
    ql_dc: Optional[ql.DayCounter] = ql.Actual360(),
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
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
    settlement_day: Optional[int] = 2,
) -> ql.YieldTermStructure:
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
    ql_dc: Optional[ql.DayCounter] = ql.Actual360(),
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
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
) -> ql.YieldTermStructure:
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


def build_fitted_bond_curve(
    bond_helpers: List[ql.BondHelper],
    as_of_date: datetime,
    fitting_method: Literal["NelsonSiegelFitting", "SvenssonFitting", "SimplePolynomialFitting", "ExponentialSplinesFitting", "CubicBSplinesFitting"],
    cubic_knots: Optional[List[float]] = [-50, -20, 0, 1, 2, 3, 5, 7, 10, 20, 30, 50],
    poly_degree: Optional[int] = 2,
    ql_dc: Optional[ql.DayCounter] = ql.Actual360(),
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    settlement_day: Optional[int] = 2,
) -> ql.YieldTermStructure:
    fitting_mapping = {
        "NelsonSiegelFitting": ql.NelsonSiegelFitting(),
        "SvenssonFitting": ql.SvenssonFitting(),
        "SimplePolynomialFitting": ql.SimplePolynomialFitting(poly_degree),
        "ExponentialSplinesFitting": ql.ExponentialSplinesFitting(),
        "CubicBSplinesFitting": ql.CubicBSplinesFitting(cubic_knots),
    }
    try:
        method = fitting_mapping[fitting_method]
    except KeyError:
        raise ValueError(f"Fitting method '{fitting_method}' is not available. " f"Choose one of {list(fitting_mapping.keys())}.")

    fitted_curve = ql.FittedBondDiscountCurve(*[ql_cal.advance(datetime_to_ql_date(as_of_date), ql.Period(settlement_day, ql.Days)), bond_helpers, ql_dc], method)
    return fitted_curve


def build_piecewise_curve_from_cmt_scipy_spline(
    spline_func: Callable[[float], float],
    anchor_date: datetime,
    maturity_dates: Optional[List[datetime]] = None,
    settlement_days: Optional[int] = 0,
    ql_coupon_frequency: Optional[int] = ql.Semiannual,
    ql_calendar: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    ql_business_convention: Optional[int] = ql.ModifiedFollowing,
    ql_day_counter: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),
    face_amount: Optional[float] = 100.0,
    ql_interpolation_algo: Optional[
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
    enable_extrapolation: Optional[bool] = True,
) -> ql.YieldTermStructure:
    ql_anchor_date = datetime_to_ql_date(anchor_date)
    ql.Settings.instance().evaluationDate = ql_anchor_date

    if not maturity_dates:
        maturity_dates = [
            ql_date_to_datetime(ql_dt)
            for ql_dt in ql.Schedule(
                ql_calendar.advance(datetime_to_ql_date(anchor_date), ql.Period("1D")),
                ql_calendar.advance(datetime_to_ql_date(anchor_date), ql.Period("30Y")),
                ql.Period(ql.Weekly),
                ql_calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            ).dates()
        ]

    maturity_dates = sorted(set(maturity_dates))

    yf_set = set()
    yf_set_dup_tol = 1e-3
    bond_helpers = []
    for mat_date in sorted(maturity_dates):
        ql_maturity_date = datetime_to_ql_date(mat_date)

        yf = ql_day_counter.yearFraction(ql_anchor_date, ql_maturity_date)
        yf_duplicate_exists = any(abs(yf - existing) < yf_set_dup_tol for existing in yf_set)
        if yf_duplicate_exists:
            continue

        yf_set.add(yf)
        par_yield = spline_func(yf)
        if not par_yield or np.isnan(par_yield):
            continue

        schedule = ql.Schedule(
            ql_anchor_date,
            ql_maturity_date,
            ql.Period(ql_coupon_frequency),
            ql_calendar,
            ql_business_convention,
            ql_business_convention,
            ql.DateGeneration.Backward,
            False,
        )

        clean_price = 100.0
        coupons = [par_yield / 100]

        bond_helper = ql.FixedRateBondHelper(
            ql.QuoteHandle(ql.SimpleQuote(clean_price)),
            settlement_days,
            face_amount,
            schedule,
            coupons,
            ql_day_counter,
            ql_business_convention,
            face_amount,
            ql_anchor_date,
        )
        bond_helpers.append(bond_helper)

    piecewise_curve = build_piecewise_ql_discount_curve(
        swap_rate_helpers=bond_helpers, ql_dc=ql_day_counter, ql_cal=ql_calendar, interpolation_algo=ql_interpolation_algo, settlement_day=settlement_days
    )
    if enable_extrapolation:
        piecewise_curve.enableExtrapolation()
    return piecewise_curve


def build_piecewise_curve_from_zero_scipy_spline(
    spline_func: Callable[[float], float],
    anchor_date: datetime,
    maturity_dates: Optional[List[datetime]] = None,
    ql_calendar: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    ql_business_convention: Optional[int] = ql.ModifiedFollowing,
    ql_day_counter: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),
    ql_interpolation_algo: Optional[
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
    ] = "log_linear",
    return_ql_zero_curve: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True,
) -> ql.YieldTermStructure:
    ql_anchor_date = datetime_to_ql_date(anchor_date)
    ql.Settings.instance().evaluationDate = ql_anchor_date

    if not maturity_dates:
        maturity_dates = [
            ql_date_to_datetime(ql_dt)
            for ql_dt in ql.Schedule(
                ql_calendar.advance(ql_anchor_date, ql.Period("1D")),
                ql_calendar.advance(ql_anchor_date, ql.Period("30Y")),
                ql.Period(ql.Weekly),
                ql_calendar,
                ql_business_convention,
                ql_business_convention,
                ql.DateGeneration.Backward,
                False,
            ).dates()
        ]
    maturity_dates = sorted(set(maturity_dates))

    valid_dates = []
    discount_factors = []
    zero_rates = []
    yf_set = set()
    tol = 1e-3

    for mat_date in maturity_dates:
        ql_mat_date = datetime_to_ql_date(mat_date)

        yf = ql_day_counter.yearFraction(ql_anchor_date, ql_mat_date)
        if any(abs(yf - existing) < tol for existing in yf_set):
            continue
        yf_set.add(yf)

        zr = spline_func(yf)
        if zr is None or np.isnan(zr):
            continue

        df = np.exp(-zr * yf)
        valid_dates.append(mat_date)
        discount_factors.append(df)
        zero_rates.append(zr)

    if 0.0 not in yf_set:
        valid_dates.insert(0, anchor_date)
        discount_factors.insert(0, 1.0)
        if return_ql_zero_curve:
            zero_rates.insert(0, 1e-8)
        else:
            zero_rates.insert(0, 0.0)

    dates_series = pd.Series(valid_dates)
    df_series = pd.Series(discount_factors)
    zr_series = pd.Series(zero_rates)

    if return_ql_zero_curve:
        curve = build_ql_zero_curve(
            datetime_series=dates_series,
            zero_rate_series=zr_series,
            ql_dc=ql_day_counter,
            ql_cal=ql_calendar,
            interpolation_algo=f"z_{ql_interpolation_algo}",
        )
    else:
        curve = build_ql_discount_curve(
            datetime_series=dates_series,
            discount_factor_series=df_series,
            ql_dc=ql_day_counter,
            ql_cal=ql_calendar,
            interpolation_algo=f"df_{ql_interpolation_algo}",
        )

    if enable_extrapolation:
        curve.enableExtrapolation()

    return curve


def build_piecewise_curve_from_discount_factor_scipy_spline(
    spline_func: Callable[[float], float],
    anchor_date: datetime,
    maturity_dates: Optional[List[datetime]] = None,
    settlement_days: Optional[int] = 1,
    ql_calendar: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    ql_business_convention: Optional[int] = ql.ModifiedFollowing,
    ql_day_counter: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),
    ql_interpolation_algo: Optional[
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
    ] = "log_linear",
    return_ql_zero_curve: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True,
    max_ql_period: Optional[ql.Period] = ql.Period("30Y"),
) -> ql.YieldTermStructure:
    ql_anchor_date = datetime_to_ql_date(anchor_date)
    ql.Settings.instance().evaluationDate = ql_anchor_date

    if not maturity_dates:
        maturity_dates = [
            ql_date_to_datetime(ql_dt)
            for ql_dt in ql.Schedule(
                ql_calendar.advance(ql_anchor_date, ql.Period(settlement_days, ql.Days)),
                ql_calendar.advance(ql_anchor_date, max_ql_period),
                ql.Period(ql.Weekly),
                ql_calendar,
                ql_business_convention,
                ql_business_convention,
                ql.DateGeneration.Backward,
                False,
            ).dates()
        ]
    maturity_dates = sorted(set(maturity_dates))

    valid_dates = []
    discount_factors = []
    zero_rates = []
    yf_set = set()
    tol = 1e-3

    for mat_date in maturity_dates:
        ql_mat_date = datetime_to_ql_date(mat_date)
        yf = ql_day_counter.yearFraction(ql_anchor_date, ql_mat_date)

        if any(abs(yf - existing) < tol for existing in yf_set):
            continue
        yf_set.add(yf)

        df_val = float(spline_func(yf))
        if df_val is None or np.isnan(df_val):
            continue

        valid_dates.append(mat_date)
        discount_factors.append(df_val)

        if yf > 0 and df_val > 0:
            implied_zr = -np.log(df_val) / yf
        else:
            implied_zr = 0.0
        zero_rates.append(implied_zr)

    if 0.0 not in yf_set:
        valid_dates.insert(0, anchor_date)
        discount_factors.insert(0, 1.0)
        if return_ql_zero_curve:
            zero_rates.insert(0, 1e-8)
        else:
            zero_rates.insert(0, 0.0)

    dates_series = pd.Series(valid_dates)
    df_series = pd.Series(discount_factors)
    zr_series = pd.Series(zero_rates)

    if return_ql_zero_curve:
        curve = build_ql_zero_curve(
            datetime_series=dates_series,
            zero_rate_series=zr_series,
            ql_dc=ql_day_counter,
            ql_cal=ql_calendar,
            interpolation_algo=f"z_{ql_interpolation_algo}",
        )
    else:
        curve = build_ql_discount_curve(
            datetime_series=dates_series,
            discount_factor_series=df_series,
            ql_dc=ql_day_counter,
            ql_cal=ql_calendar,
            interpolation_algo=f"df_{ql_interpolation_algo}",
        )

    if enable_extrapolation:
        curve.enableExtrapolation()

    return curve


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
