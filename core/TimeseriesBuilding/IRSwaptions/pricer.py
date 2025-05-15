import math
from typing import Optional, List, Callable, Tuple, Iterable

import numpy as np
import QuantLib as ql

from core.TimeseriesBuilding.Base.utils import build_irswaption
from core.utils.ql_utils import ql_date_to_datetime


def calc_spot_prem(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    notional = swaption.underlying().nominal()
    return swaption.NPV() / (notional / 100)


def calc_fwd_prem(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    notional = swaption.underlying().nominal()
    return swaption.forwardPrice() / (notional / 100)


def calc_spot_npv(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    return swaption.NPV()


def calc_fwd_npv(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    return swaption.forwardPrice()


def calc_nvol(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, use_fwd_prem: Optional[bool] = False):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    if use_fwd_prem:
        return swaption.impliedVolatility(swaption.forwardPrice(), curve_handle, guess=0.01, type=ql.Normal, priceType=ql.Swaption.Forward) * 10_000
    else:
        return swaption.impliedVolatility(swaption.NPV(), curve_handle, guess=0.01, type=ql.Normal) * 10_000


def calc_delta(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swaption.setPricingEngine(engine)
    abs_delta = abs(swaption.delta() / swaption.annuity())
    return abs_delta if swaption.underlying().fixedLegBPS() > 0 else -abs_delta


def calc_dv01(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date
    swaption.setPricingEngine(engine)
    abs_dv01 = abs(swaption.delta() / 10_000) # bps
    return abs_dv01 if swaption.underlying().fixedLegBPS() > 0 else -abs_dv01


def calc_vega(
    swaption: ql.Swaption,
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
):
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    swaption.setPricingEngine(engine)
    return swaption.vega() / 10_000


def calc_gamma(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, curve: str, bump_bps: Optional[int] = 1):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date

    orig_strike = swaption.underlying().fixedRate()
    notional = swaption.underlying().nominal()
    r_p = "rec" if swaption.underlying().fixedLegBPS() > 0 else "pay"
    dS = bump_bps / 10_000.0

    s_up = build_irswaption(
        curve=curve,
        curve_handle=curve_handle,
        pricing_engine=engine,
        exercise_date=ql_date_to_datetime(swaption.exercise().dates()[0]),
        underlying_effective_date=ql_date_to_datetime(swaption.underlying().startDate()),
        underlying_maturity_date=ql_date_to_datetime(swaption.underlying().maturityDate()),
        strike=(orig_strike + dS) * 100,
        notional=notional,
        r_p=r_p,
    )
    s_dn = build_irswaption(
        curve=curve,
        curve_handle=curve_handle,
        pricing_engine=engine,
        exercise_date=ql_date_to_datetime(swaption.exercise().dates()[0]),
        underlying_effective_date=ql_date_to_datetime(swaption.underlying().startDate()),
        underlying_maturity_date=ql_date_to_datetime(swaption.underlying().maturityDate()),
        strike=(orig_strike - dS) * 100,
        notional=notional,
        r_p=r_p,
    )

    g_up = calc_delta(s_up, curve_handle, engine)
    g_dn = calc_delta(s_dn, curve_handle, engine)
    return abs((g_up - g_dn) / (2 * dS)) / 10_000


def calc_dollar_gamma(swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, curve: str, bump_bps: Optional[int] = 1):
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date

    orig_strike = swaption.underlying().fixedRate()
    notional = swaption.underlying().nominal()
    r_p = "rec" if swaption.underlying().fixedLegBPS() > 0 else "pay"
    dS = bump_bps / 10_000.0

    s_up = build_irswaption(
        curve=curve,
        curve_handle=curve_handle,
        pricing_engine=engine,
        exercise_date=ql_date_to_datetime(swaption.exercise().dates()[0]),
        underlying_effective_date=ql_date_to_datetime(swaption.underlying().startDate()),
        underlying_maturity_date=ql_date_to_datetime(swaption.underlying().maturityDate()),
        strike=(orig_strike + dS) * 100,
        notional=notional,
        r_p=r_p,
    )
    s_dn = build_irswaption(
        curve=curve,
        curve_handle=curve_handle,
        pricing_engine=engine,
        exercise_date=ql_date_to_datetime(swaption.exercise().dates()[0]),
        underlying_effective_date=ql_date_to_datetime(swaption.underlying().startDate()),
        underlying_maturity_date=ql_date_to_datetime(swaption.underlying().maturityDate()),
        strike=(orig_strike - dS) * 100,
        notional=notional,
        r_p=r_p,
    )

    g_up = calc_dv01(s_up, curve_handle, engine)
    g_dn = calc_dv01(s_dn, curve_handle, engine)
    return abs((g_up - g_dn) / (2 * dS)) / 10_000


def calc_theta(
    swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, horizon: Optional[ql.Period] = ql.Period("1D")
) -> float:
    swaption.setPricingEngine(engine)
    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date

    price_today = swaption.NPV()
    ql.Settings.instance().evaluationDate = curve_date + 1
    theta = swaption.NPV() - price_today
    ql.Settings.instance().evaluationDate = curve_date
    return theta


def calc_charm(
    swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, horizon: Optional[ql.Period] = ql.Period("1D")
) -> float:
    swaption.setPricingEngine(engine)

    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date

    delta = swaption.delta()
    ql.Settings.instance().evaluationDate = curve_date + horizon
    charm = swaption.delta() - delta
    ql.Settings.instance().evaluationDate = curve_date
    return charm


def calc_veta(
    swaption: ql.Swaption, curve_handle: ql.YieldTermStructureHandle, engine: ql.BachelierSwaptionEngine, horizon: Optional[ql.Period] = ql.Period("1D")
) -> float:
    swaption.setPricingEngine(engine)

    curve_date = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = curve_date

    vega = swaption.vega()
    ql.Settings.instance().evaluationDate = curve_date + horizon
    veta = swaption.vega() - vega
    ql.Settings.instance().evaluationDate = curve_date
    return veta


# TODO second order greeks functions

# def build_swaption_with_vol_shift(
#     swaption: ql.Swaption,
#     curve_handle: ql.YieldTermStructureHandle,
#     vol_handle: ql.SwaptionVolatilityStructureHandle,
#     shift: float,
#     ql_bday_convention=ql.ModifiedFollowing,
# ) -> ql.Swaption:
#     curve_date = curve_handle.referenceDate()
#     ql.Settings.instance().evaluationDate = curve_date

#     original_vol: ql.SwaptionVolatilityStructure = vol_handle.currentLink()

#     exercise_date = swaption.exercise().dates()[0]
#     atm_vol = original_vol.volatility(
#         original_vol.timeFromReference(exercise_date),
#         original_vol.dayCounter().yearFraction(exercise_date, swaption.underlying().maturityDate()),
#         swaption.underlying().fairRate(),
#     )

#     bumped_vol = atm_vol + shift
#     bumped_surface = ql.ConstantSwaptionVolatility(curve_date, original_vol.calendar(), ql_bday_convention, bumped_vol, original_vol.dayCounter(), ql.Normal)
#     bumped_handle = ql.RelinkableSwaptionVolatilityStructureHandle()
#     bumped_handle.linkTo(bumped_surface)
#     bumped_engine = ql.BachelierSwaptionEngine(curve_handle, bumped_handle)

#     swaption_copy = ql.Swaption(swaption.underlying(), swaption.exercise())
#     swaption_copy.setPricingEngine(bumped_engine)
#     return swaption_copy


# def calc_volga(
#     swaption: ql.Swaption,
#     curve_handle: ql.YieldTermStructureHandle,
#     vol_handle: ql.SwaptionVolatilityStructureHandle,
#     bump_bps: Optional[float] = 1.0,
# ) -> float:
#     curve_date = curve_handle.referenceDate()
#     ql.Settings.instance().evaluationDate = curve_date

#     dVol = bump_bps / 10_000.0

#     s_up = build_swaption_with_vol_shift(
#         swaption=swaption,
#         curve_handle=curve_handle,
#         vol_handle=vol_handle,
#         shift=+dVol,
#     )
#     P_up = s_up.NPV()
#     print(s_up.vega())
#     # vega_up = s_up.vega() / (s_up.underlying().nominal() / 100)

#     s_down = build_swaption_with_vol_shift(
#         swaption=swaption,
#         curve_handle=curve_handle,
#         vol_handle=vol_handle,
#         shift=-dVol,
#     )
#     P_down = s_down.NPV()
#     print(s_down.vega())
#     # vega_down = s_down.vega() / (s_down.underlying().nominal() / 100)

#     P0 = swaption.NPV()

#     # print(P0)
#     # print(P_up)
#     # print(P_down)

#     return (P_up - 2.0 * P0 + P_down) / (dVol * dVol)


def calc_portfolio_vega_weighted_nvol(
    book: List[ql.Swaption],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()

    vols = np.array([calc_nvol(sw, curve_handle, engine) for sw in book], dtype=float)
    vegas = np.array([calc_vega(sw, curve_handle, engine) * math.copysign(1.0, sw.underlying().nominal()) for sw in book], dtype=float)
    total_vega = vegas.sum()
    if np.isclose(total_vega, 0.0):
        raise ValueError("Total signed vega is zero; cannot compute weighted vol")

    return float(np.dot(vols, vegas) / total_vega)


def calc_straddle_annual_breakeven_nvol(
    straddle: Iterable[ql.Swaption],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    today = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = today
    return np.sqrt(2 / np.pi) * calc_portfolio_vega_weighted_nvol(straddle, curve_handle, engine)


def calc_straddle_daily_breakeven_nvol(
    straddle: Iterable[ql.Swaption],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    today = curve_handle.referenceDate()
    ql.Settings.instance().evaluationDate = today
    return np.sqrt(2 / np.pi) * (calc_portfolio_vega_weighted_nvol(straddle, curve_handle, engine) / np.sqrt(252))


def _w_sum(
    fn: Callable[[ql.Swaption], float],
    fn_args: Tuple,
    book: List[ql.Swaption],
    weights: List[float],
) -> float:
    if weights is None:
        weights = [1.0] * len(book)
    if len(weights) != len(book):
        raise ValueError("risk_weights length must match book length")

    return sum(w * fn(sw, *fn_args) for sw, w in zip(book, weights))


def calc_book_spot_prem(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_spot_prem, (curve_handle, engine), book, risk_weights)


def calc_book_fwd_prem(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_fwd_prem, (curve_handle, engine), book, risk_weights)


def calc_book_spot_npv(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_spot_npv, (curve_handle, engine), book, risk_weights)


def calc_book_fwd_npv(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_fwd_npv, (curve_handle, engine), book, risk_weights)


def calc_book_delta(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_delta, (curve_handle, engine), book, risk_weights)


def calc_book_dv01(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_dv01, (curve_handle, engine), book, risk_weights)


def calc_book_vega(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_vega, (curve_handle, engine), book, risk_weights)


def calc_book_gamma(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    curve: str,
    bump_bps: Optional[float] = 1.0,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_gamma, (curve_handle, engine, curve, bump_bps), book, risk_weights)


def calc_book_dollar_gamma(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    curve: str,
    bump_bps: Optional[float] = 1.0,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_dollar_gamma, (curve_handle, engine, curve, bump_bps), book, risk_weights)


def calc_book_theta(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    horizon: Optional[ql.Period] = ql.Period("1D"),
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_theta, (curve_handle, engine, horizon), book, risk_weights)


def calc_book_charm(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    horizon: Optional[ql.Period] = ql.Period("1D"),
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_charm, (curve_handle, engine, horizon), book, risk_weights)


def calc_book_veta(
    book: List[ql.Swaption],
    risk_weights: List[float],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    horizon: Optional[ql.Period] = ql.Period("1D"),
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    return _w_sum(calc_veta, (curve_handle, engine, horizon), book, risk_weights)


# def calc_custom_spreadable(
#     book: Dict[ql.Swaption, float],  # swaption => risk weight
#     curve_handle: ql.YieldTermStructureHandle,
#     engine: ql.BachelierSwaptionEngine,
#     value_func: Optional[Callable[[ql.Swaption, ql.YieldTermStructureHandle, ql.BachelierSwaptionEngine], float]] = calc_nvol,
# ):
#     today = curve_handle.referenceDate()
#     ql.Settings.instance().evaluationDate = today

#     weighted_sum = 0.0
#     total_long_weight = 0.0
#     for sw, weight in book.items():
#         v = value_func(sw, curve_handle, engine)
#         weighted_sum += weight * v
#         if weight > 0:
#             total_long_weight += weight

#     if math.isclose(total_long_weight, 0.0):
#         raise ValueError("Total long weight is zero; cannot normalize")

#     return weighted_sum / total_long_weight


# def calc_fwd_vol(
#     book: List[ql.Swaption],  # ordered e.g. 1Yx2Yx5Y => [3Yx5Y, 1Yx2Y, 1Yx7Y]
#     curve_handle: ql.YieldTermStructureHandle,
#     engine: ql.BachelierSwaptionEngine,
#     rho: Optional[float] = 1.0,
# ):
#     pass


# def calc_midcurve_implied_corr(
#     book: List[ql.Swaption],  # ordered e.g. 3Mx1Yx2Y => [3Yx1Y, 3Mx3Y]
#     curve_handle: ql.YieldTermStructureHandle,
#     engine: ql.BachelierSwaptionEngine,
# ):
#     pass
