import functools
from datetime import datetime
from typing import Callable, Literal, Optional

import math
import QuantLib as ql
from rateslib.calendars import get_imm
from scipy.optimize import newton

from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime


def build_irswap(
    curve: str,
    curve_handle: ql.YieldTermStructureHandle,
    swap_index: Optional[ql.SwapIndex] = None,
    fwd: Optional[ql.Period] = None,
    tenor: Optional[ql.Period] = None,
    effective_date: Optional[datetime] = None,
    maturity_date: Optional[datetime] = None,
    fixed_rate: Optional[float] = -0.00,
    notional: Optional[float] = 100_000_000,
    bpv: Optional[float] = None,
    r_p: Optional[Literal["rec", "r", "pay", "p"]] = None,
) -> ql.VanillaSwap | ql.OvernightIndexedSwap:

    def _make_swap(nominal: float) -> ql.VanillaSwap | ql.OvernightIndexedSwap:
        period_tenor = fwd and tenor
        mm_tenor = effective_date and maturity_date
        assert period_tenor or mm_tenor, "Must specify either tenor (fwd+tenor) or dates (effective+termination)"

        if period_tenor:
            fwd_start = ql.Period(fwd) if isinstance(fwd, str) else fwd
            swap_tenor = ql.Period(tenor) if isinstance(tenor, str) else tenor

            if CME_IRSWAP_CURVE_QL_PARAMS[curve]["is_ois"]:
                swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                    fwdStart=fwd_start,
                    swapTenor=swap_tenor,
                    fixedRate=fixed_rate / 100,
                    overnightIndex=swap_index or CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle),
                    paymentLag=CME_IRSWAP_CURVE_QL_PARAMS[curve]["paymentLag"],
                    settlementDays=CME_IRSWAP_CURVE_QL_PARAMS[curve]["settlementDays"],
                    calendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    paymentCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    fixedLegDayCount=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                    fixedLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    paymentAdjustmentConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    paymentFrequency=CME_IRSWAP_CURVE_QL_PARAMS[curve]["frequency"],
                    nominal=abs(nominal),
                    receiveFixed=r_p.startswith("r") if r_p else nominal > 0,
                )
            else:
                swap: ql.VanillaSwap = ql.MakeVanillaSwap(
                    forwardStart=fwd_start,
                    swapTenor=swap_tenor,
                    iborIndex=swap_index or CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle),
                    fixedRate=fixed_rate / 100,
                    fixedLegTenor=CME_IRSWAP_CURVE_QL_PARAMS[curve]["period"],
                    fixedLegCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    fixedLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    fixedLegTerminationDateConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    fixedLegDayCount=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                    floatingLegTenor=CME_IRSWAP_CURVE_QL_PARAMS[curve]["period"],
                    floatingLegCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    floatingLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    floatingLegTerminationDateConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    nominal=abs(nominal),
                    receiveFixed=r_p.startswith("r") if r_p else nominal > 0,
                )
        else:
            eff = datetime_to_ql_date(effective_date)
            term = datetime_to_ql_date(maturity_date)
            if CME_IRSWAP_CURVE_QL_PARAMS[curve]["is_ois"]:
                swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                    fwdStart=ql.Period("-0D"),
                    swapTenor=ql.Period("-0D"),
                    fixedRate=fixed_rate / 100,
                    overnightIndex=swap_index or CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle),
                    paymentLag=CME_IRSWAP_CURVE_QL_PARAMS[curve]["paymentLag"],
                    settlementDays=CME_IRSWAP_CURVE_QL_PARAMS[curve]["settlementDays"],
                    calendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    paymentCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    fixedLegDayCount=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                    fixedLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    paymentAdjustmentConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    paymentFrequency=CME_IRSWAP_CURVE_QL_PARAMS[curve]["frequency"],
                    nominal=abs(nominal),
                    receiveFixed=r_p.startswith("r") if r_p else nominal > 0,
                    effectiveDate=eff,
                    terminationDate=term,
                )
            else:
                swap: ql.VanillaSwap = ql.MakeVanillaSwap(
                    forwardStart=ql.Period("-0D"),
                    swapTenor=ql.Period("-0D"),
                    iborIndex=swap_index or CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](curve_handle),
                    fixedRate=fixed_rate / 100,
                    fixedLegTenor=CME_IRSWAP_CURVE_QL_PARAMS[curve]["period"],
                    fixedLegCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    fixedLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    fixedLegTerminationDateConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    fixedLegDayCount=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                    floatingLegTenor=CME_IRSWAP_CURVE_QL_PARAMS[curve]["period"],
                    floatingLegCalendar=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    floatingLegConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    floatingLegTerminationDateConvention=CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    nominal=abs(nominal),
                    receiveFixed=r_p.startswith("r") if r_p else nominal > 0,
                    effectiveDate=eff,
                    terminationDate=term,
                )

        swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
        return swap

    if fixed_rate == -0.00 or (isinstance(fixed_rate, str) and fixed_rate.lower() == "par"):
        temp = _make_swap(1.0)
        fixed_rate = temp.fairRate() * 100

    swap0 = _make_swap(notional)
    if bpv is not None:
        bps0 = swap0.fixedLegBPS()
        if not bps0:
            raise ValueError("Cannot scale notional: swap fixedLegBPS is zero")
        scale = bpv / bps0
        return _make_swap(math.copysign(notional * scale, bpv))
    return swap0


def _delta_to_strike_qlloop(
    *,
    build_swaption_with_strike: Callable[[float], ql.Swaption],
    pricing_engine: ql.BachelierSwaptionEngine,
    target_abs_delta: float,
    guess: float,
    bump: Optional[float] = 10 / 10_000,  # 10 bp bump
    tol: Optional[float] = 0.05,
    max_iter: Optional[int] = 100,
) -> float:
    @functools.lru_cache(maxsize=256)
    def _g_cached(K: float) -> float:
        sw = build_swaption_with_strike(K * 100)
        sw.setPricingEngine(pricing_engine)
        delta_val = abs(sw.delta() / sw.annuity())
        return delta_val - target_abs_delta

    def g(K: float) -> float:
        return _g_cached(K)

    def g_prime(K: float) -> float:
        return (_g_cached(K + bump) - _g_cached(K - bump)) / (2 * bump)

    def g_double(K: float) -> float:
        return (_g_cached(K + bump) - 2 * _g_cached(K) + _g_cached(K - bump)) / (bump**2)

    try:
        root = newton(
            func=g,
            x0=guess,
            fprime=g_prime,
            fprime2=g_double,
            rtol=tol,
            maxiter=max_iter,
        )
    except RuntimeError as e:
        raise RuntimeError(f"delta-to-strike did not converge in {max_iter} iterations: {e}")

    return root


def build_irswaption(
    curve: str,
    curve_handle: ql.YieldTermStructureHandle,
    pricing_engine: ql.BachelierSwaptionEngine,
    swap_index: Optional[ql.SwapIndex] = None,
    *,
    expiry: Optional[ql.Period] = None,
    tail: Optional[str | ql.Period] = None,
    exercise_date: Optional[datetime] = None,
    underlying_effective_date: Optional[datetime] = None,
    underlying_maturity_date: Optional[datetime] = None,
    strike: str | float | int = "ATMF",
    notional: float = 100_000_000,
    vega: Optional[float] = None,
    r_p: Literal["rec", "r", "pay", "p"] = "rec",
) -> ql.Swaption:
    ref_date: ql.Date = curve_handle.referenceDate()
    cal: ql.Calendar = curve_handle.calendar()
    ql.Settings.instance().evaluationDate = ref_date

    period_tenor = expiry and tail
    mm_tenor = exercise_date and underlying_effective_date and underlying_maturity_date
    if not (period_tenor or mm_tenor):
        raise ValueError("Need either (expiry, tail) or explicit dates")

    def _convert_strike(spec: str | float | int, atmf: float) -> float:
        if isinstance(spec, (int, float)):
            spec = float(spec)
            return float(spec) / 100

        up = str(spec).strip().upper()
        if up == "ATMF" or up == "ATM":
            return atmf

        if up.startswith("ATMS"):
            spot_starting_swap = build_irswap(
                curve=curve,
                curve_handle=curve_handle,
                swap_index=swap_index,
                fwd=ql.Period("0D"),
                tenor=tail,
                effective_date=ql_date_to_datetime(ref_date),
                maturity_date=underlying_maturity_date,
                r_p=r_p,
            )
            atms = spot_starting_swap.fairRate()
            if up == "ATMS":
                return atms
            sign = 1 if "+" in up else -1
            bps = float(up.split("+")[-1] if "+" in up else up.split("-")[-1])
            return atms + sign * bps / 10_000.0

        if up.startswith("ATMF"):
            sign = 1 if "+" in up else -1
            bps = float(up.split("+")[-1] if "+" in up else up.split("-")[-1])
            return atmf + sign * bps / 10_000.0

        if up.endswith("D"):  # delta spec
            target_delta = float(up[:-1]) / 100.0

            def _builder(k: float) -> ql.Swaption:
                return build_irswaption(
                    curve=curve,
                    curve_handle=curve_handle,
                    pricing_engine=pricing_engine,
                    expiry=expiry,
                    tail=tail,
                    exercise_date=exercise_date,
                    underlying_effective_date=underlying_effective_date,
                    underlying_maturity_date=underlying_maturity_date,
                    strike=k,
                    notional=notional,
                    r_p=r_p,
                )

            return _delta_to_strike_qlloop(
                build_swaption_with_strike=_builder,
                pricing_engine=pricing_engine,
                target_abs_delta=target_delta,
                guess=atmf,
            )

        raise ValueError(f"Unrecognised strike spec {spec!r}")

    def _to_dt(d):
        if isinstance(d, ql.Period):
            d = cal.advance(ref_date, d)
        if isinstance(d, ql.Date):
            return ql_date_to_datetime(d)
        if isinstance(d, str) and d.upper().startswith("IMM_"):
            return get_imm(code=d.split("IMM_")[-1])
        return d

    if period_tenor:
        if isinstance(expiry, str) and not expiry.upper().startswith("IMM_"):
            expiry = ql.Period(expiry)

        if isinstance(tail, str) and "X" in tail.upper():
            fwd_tenor, swap_tenor = map(ql.Period, tail.upper().split("X"))
        else:
            fwd_tenor, swap_tenor = ql.Period("0D"), ql.Period(tail) if isinstance(tail, str) else tail

        exercise_date = _to_dt(expiry)
        underlying_effective_date = _to_dt(cal.advance(datetime_to_ql_date(exercise_date), fwd_tenor))
        underlying_maturity_date = _to_dt(cal.advance(datetime_to_ql_date(underlying_effective_date), swap_tenor))
    else:
        exercise_date = _to_dt(exercise_date)
        underlying_effective_date = _to_dt(underlying_effective_date)
        underlying_maturity_date = _to_dt(underlying_maturity_date)

    k = (
        _convert_strike(
            strike,
            build_irswap(
                curve=curve,
                curve_handle=curve_handle,
                swap_index=swap_index,
                effective_date=underlying_effective_date,
                maturity_date=underlying_maturity_date,
                notional=notional,
                r_p=r_p,
            ).fairRate(),
        )
        * 100
    )
    underlying = build_irswap(
        curve=curve,
        curve_handle=curve_handle,
        swap_index=swap_index,
        effective_date=underlying_effective_date,
        maturity_date=underlying_maturity_date,
        fixed_rate=k,
        notional=notional,
        r_p=r_p,
    )
    ql_swaption = ql.Swaption(underlying, ql.EuropeanExercise(datetime_to_ql_date(exercise_date)))
    ql_swaption.setPricingEngine(pricing_engine)

    if vega is not None:
        V0 = ql_swaption.vega() / 10_000
        if V0 == 0:
            raise ValueError("Cannot scale notional: swaption vega is zero")

        ql_swaption = build_irswaption(
            curve=curve,
            curve_handle=curve_handle,
            pricing_engine=pricing_engine,
            swap_index=swap_index,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
            strike=strike,
            notional=notional * (vega / V0),
            r_p=r_p,
        )

    return ql_swaption
