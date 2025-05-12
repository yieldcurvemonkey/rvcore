import QuantLib as ql


def calc_fair_rate(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.fairRate()


def calc_npv(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.NPV()


def calc_bpv(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.fixedLegBPS()

def calc_notional(swap: ql.VanillaSwap, curve_handle: ql.YieldTermStructureHandle):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swap.setPricingEngine(ql.DiscountingSwapEngine(curve_handle))
    return swap.nominal()