from core.utils.ql_loader import ql

from typing import Optional

from core.TimeseriesBuilding.Base.utils import build_irswap
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.utils.ql_utils import ql_date_to_datetime


def calc_fair_rate(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.fairRate()


def calc_npv(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.NPV()


def calc_pv01(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.fixedLegBPS()  # bumps fixed rate/coupon


def calc_dv01(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, curve: str, shift: Optional[float] = 1e-4) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))

    embedded_fixed_rate = swap.fixedRate()
    fixed_rate_to_use = embedded_fixed_rate if embedded_fixed_rate > 0 else swap.fairRate()

    bumped_up_ql_curve = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(ql.SimpleQuote(shift)))
    bumped_up_curve_handle = ql.YieldTermStructureHandle(bumped_up_ql_curve)
    bumped_up_swap = build_irswap(
        curve=curve,
        curve_handle=bumped_up_curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate()),
        maturity_date=ql_date_to_datetime(swap.maturityDate()),
        fixed_rate=fixed_rate_to_use * 100,
        notional=swap.nominal() if swap.fixedLegBPS() > 0 else -swap.nominal(),
    )

    bumped_down_ql_curve = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(ql.SimpleQuote(-shift)))
    bumped_down_curve_handle = ql.YieldTermStructureHandle(bumped_down_ql_curve)
    bumped_down_swap = build_irswap(
        curve=curve,
        curve_handle=bumped_down_curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate()),
        maturity_date=ql_date_to_datetime(swap.maturityDate()),
        fixed_rate=fixed_rate_to_use * 100,
        notional=swap.nominal() if swap.fixedLegBPS() > 0 else -swap.nominal(),
    )

    return (calc_npv(swap=bumped_down_swap, curve_handle=bumped_down_curve_handle) - calc_npv(swap=bumped_up_swap, curve_handle=bumped_up_curve_handle)) / 2


def calc_gamma(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, curve: str, shift: Optional[float] = 1e-4):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    npv_0 = calc_npv(swap, curve_handle)

    bumped_up_ql_curve = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(ql.SimpleQuote(shift)))
    bumped_up_curve_handle = ql.YieldTermStructureHandle(bumped_up_ql_curve)
    bumped_up_swap = build_irswap(
        curve=curve,
        curve_handle=bumped_up_curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate()),
        maturity_date=ql_date_to_datetime(swap.maturityDate()),
        fixed_rate=swap.fixedRate() * 100,
        notional=swap.nominal() if swap.fixedLegBPS() > 0 else -swap.nominal(),
    )
    npv_up = calc_npv(swap=bumped_up_swap, curve_handle=bumped_up_curve_handle)

    bumped_down_ql_curve = ql.ZeroSpreadedTermStructure(curve_handle, ql.QuoteHandle(ql.SimpleQuote(-shift)))
    bumped_down_curve_handle = ql.YieldTermStructureHandle(bumped_down_ql_curve)
    bumped_down_swap = build_irswap(
        curve=curve,
        curve_handle=bumped_down_curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate()),
        maturity_date=ql_date_to_datetime(swap.maturityDate()),
        fixed_rate=swap.fixedRate() * 100,
        notional=swap.nominal() if swap.fixedLegBPS() > 0 else -swap.nominal(),
    )
    npv_down = calc_npv(swap=bumped_down_swap, curve_handle=bumped_down_curve_handle)

    return ((npv_up - 2 * npv_0 + npv_down) / (shift**2)) / swap.nominal()


def calc_dollar_carry(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, horizon: Optional[ql.Period] = ql.Period("1D")):
    ref_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = ref_date
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))

    npv0 = swap.NPV()
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate() + horizon
    dollar_carry = swap.NPV() - npv0
    ql.Settings.instance().evaluationDate = ref_date

    return dollar_carry


def calc_carry_bps_running(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, curve: str, horizon: Optional[ql.Period] = ql.Period("1D")) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    if swap.startDate() > curve_handle.calendar().advance(curve_handle.referenceDate(), ql.Period(CME_IRSWAP_CURVE_QL_PARAMS[curve]["settlementDays"], ql.Days)):
        return 0.0

    fwd_rolled_swap = build_irswap(
        curve=curve,
        curve_handle=curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate() + horizon),
        maturity_date=ql_date_to_datetime(swap.maturityDate() - horizon),
    )
    return (fwd_rolled_swap.fixedRate() - swap.fixedRate()) * 10_000


def calc_roll_bps_running(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, curve: str, horizon: Optional[ql.Period] = ql.Period("1D")) -> float:
    rolled_swap = build_irswap(
        curve=curve,
        curve_handle=curve_handle,
        effective_date=ql_date_to_datetime(swap.startDate()),
        maturity_date=ql_date_to_datetime(swap.maturityDate() - horizon),
    )
    return (swap.fixedRate() - rolled_swap.fixedRate()) * 10_000


def calc_carry_and_roll_bps_running(
    swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle, curve: str, horizon: Optional[ql.Period] = ql.Period("1D")
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()

    return calc_carry_bps_running(swap=swap, curve_handle=curve_handle, horizon=horizon) + calc_roll_bps_running(
        swap=swap, curve_handle=curve_handle, curve=curve, horizon=horizon
    )


def calc_notional(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.nominal()
