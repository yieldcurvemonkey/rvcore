from datetime import datetime
from typing import List, Literal, Optional, Callable

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


# TODO: make curve params more structured


def get_ql_swaps_curve_params(
    curve_str: Literal[
        "USD-SOFR-1D",
        "USD-FEDFUNDS",
        "USD-OIS",
        "JPY-TONAR",
        "CAD-CORRA",
        "EUR-ESTR",
        "EUR-EURIBOR-1M",
        "EUR-EURIBOR-3M",
        "EUR-EURIBOR-6M",
        "GBP-SONIA",
        # Additional curves could be added here.
        "CHF-SARON-1D",
        "NOK-NIBOR-6M",
        "HKD-HIBOR-3M",
        "AUD-AONIA",
        "SGD-SORA-1D",
    ],
):
    """
    QL_CURVE_PARAMS returns a tuple:
      IndexClass,         -> e.g. ql.Sofr, ql.FedFunds, etc.
      is_ois,             -> True for overnight indexed swaps, False for Ibor-based indices.
      DayCounter,         -> e.g. ql.Actual360()
      Calendar,           -> e.g. ql.UnitedStates(ql.UnitedStates.SOFR)
      BusinessDayConvention, -> e.g. ql.ModifiedFollowing
      Frequency,          -> e.g. ql.Annual for OIS, or Monthly/Quarterly/Semiannual for Ibor indices.
      IndexTenor,         -> a QuantLib Period representing the index tenor (e.g. ql.Period("1D") for OIS, ql.Period("1M") for Euribor1M)
      Currency            -> a QuantLib currency object.
    """
    QL_SWAP_CURVE_PARAMS = {
        "USD-SOFR-1D": (ql.Sofr, True, ql.Actual360(), ql.UnitedStates(ql.UnitedStates.SOFR), ql.ModifiedFollowing, ql.Annual, ql.Period("1D"), ql.USDCurrency()),
        "USD-FEDFUNDS": (
            ql.FedFunds,
            True,
            ql.Actual360(),
            ql.UnitedStates(ql.UnitedStates.FederalReserve),
            ql.ModifiedFollowing,
            ql.Annual,
            ql.Period("1D"),
            ql.USDCurrency(),
        ),
        "USD-OIS": (
            ql.FedFunds,
            True,
            ql.Actual360(),
            ql.UnitedStates(ql.UnitedStates.FederalReserve),
            ql.ModifiedFollowing,
            ql.Annual,
            ql.Period("1D"),
            ql.USDCurrency(),
        ),
        "CAD-CORRA": (ql.Corra, True, ql.Actual365Fixed(), ql.Canada(ql.Canada.TSX), ql.ModifiedFollowing, ql.Annual, ql.Period("1D"), ql.CADCurrency()),
        "GBP-SONIA": (
            ql.Sonia,
            True,
            ql.Actual365Fixed(),
            ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
            ql.ModifiedFollowing,
            ql.Annual,
            ql.Period("1D"),
            ql.GBPCurrency(),
        ),
        "EUR-ESTR": (ql.Estr, True, ql.Actual360(), ql.TARGET(), ql.ModifiedFollowing, ql.Annual, ql.Period("1D"), ql.EURCurrency()),
        "EUR-EURIBOR-1M": (
            ql.Euribor1M,
            False,
            ql.Thirty360(ql.Thirty360.European),
            ql.TARGET(),
            ql.ModifiedFollowing,
            ql.Monthly,
            ql.Period("1M"),
            ql.EURCurrency(),
        ),
        "EUR-EURIBOR-3M": (
            ql.Euribor3M,
            False,
            ql.Thirty360(ql.Thirty360.European),
            ql.TARGET(),
            ql.ModifiedFollowing,
            ql.Quarterly,
            ql.Period("3M"),
            ql.EURCurrency(),
        ),
        "EUR-EURIBOR-6M": (
            ql.Euribor6M,
            False,
            ql.Thirty360(ql.Thirty360.European),
            ql.TARGET(),
            ql.ModifiedFollowing,
            ql.Semiannual,
            ql.Period("6M"),
            ql.EURCurrency(),
        ),
        "JPY-TONAR": (ql.Tona, True, ql.Actual365Fixed(), ql.Japan(), ql.ModifiedFollowing, ql.Annual, ql.Period("1D"), ql.JPYCurrency()),
    }

    return QL_SWAP_CURVE_PARAMS[curve_str]


def get_ql_cash_curve_params(curve_str: Literal["USD"]):
    QL_CASH_CURVE_PARAMS = {
        "USD": (
            ql.UnitedStates(ql.UnitedStates.GovernmentBond),
            1,
            ql.Semiannual,
            ql.ActualActual(ql.ActualActual.Actual365),
            ql.ModifiedFollowing,
            100,
            ["CB12", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"],
        )
    }

    return QL_CASH_CURVE_PARAMS[curve_str]
