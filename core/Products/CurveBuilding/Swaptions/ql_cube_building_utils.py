import pandas as pd
import QuantLib as ql
from datetime import datetime
from typing import Optional, List, Literal, Dict, Tuple, Optional, Literal, Annotated


from core.utils.ql_utils import datetime_to_ql_date, period_to_string, period_to_months, parse_tenor_string
from core.Products.CurveBuilding.Swaptions.types import SCube


def build_ql_swaption_with_vol_shift(
    ql_curve: ql.DiscountCurve,
    swaption: ql.Swaption,
    base_vol_handle: ql.SwaptionVolatilityStructureHandle,
    shift: float,
    expiry_str: str | int,
    ql_bday_convention=ql.ModifiedFollowing,
) -> ql.Swaption:
    swaption_copy = ql.Swaption(swaption.underlying(), swaption.exercise())
    original_vol_structure: ql.SwaptionVolatilityStructureHandle = base_vol_handle.currentLink()
    ref_date = original_vol_structure.referenceDate()
    bumped_vol_value = original_vol_structure.volatility(ref_date, ql.Period(expiry_str) if type(expiry_str) == str else ql.Period(expiry_str, ql.Days), 0.0) + shift
    bumped_vol_surface = ql.ConstantSwaptionVolatility(
        ref_date,
        original_vol_structure.calendar(),
        ql_bday_convention,
        bumped_vol_value,
        original_vol_structure.dayCounter(),
        ql.Normal,
    )
    bumped_vol_handle = ql.RelinkableSwaptionVolatilityStructureHandle()
    bumped_vol_handle.linkTo(bumped_vol_surface)
    bumped_engine = ql.BachelierSwaptionEngine(ql.RelinkableYieldTermStructureHandle(ql_curve), bumped_vol_handle)
    swaption_copy.setPricingEngine(bumped_engine)
    return swaption_copy


def calc_swaption_greeks_helper(
    ql_date: ql.Date,
    original_swaption: ql.Swaption,
    swaption_strike_bumped_up: ql.Swaption,
    swaption_strike_bumped_down: ql.Swaption,
    swaption_vol_bumped_up: ql.Swaption,
    swaption_vol_bumped_down: ql.Swaption,
    swaption_strike_up_vol_up: ql.Swaption,
    swaption_strike_up_vol_down: ql.Swaption,
    swaption_strike_down_vol_up: ql.Swaption,
    swaption_strike_down_vol_down: ql.Swaption,
    dStrike: float,
    dVol: float,
    is_receiver: bool,
    is_long: bool,
) -> Tuple[Dict[str, float], float]:
    ql.Settings.instance().evaluationDate = ql_date

    dv01 = original_swaption.delta() / 10_000.0
    vega = original_swaption.vega() / 10_000.0
    gamma = ((swaption_strike_bumped_up.delta() / 10_000.0) - (swaption_strike_bumped_down.delta() / 10_000.0)) / (2 * dStrike) / 100
    volga = ((swaption_vol_bumped_up.vega() / 10_000.0) - (swaption_vol_bumped_down.vega() / 10_000.0)) / (2 * dVol)
    vanna = (
        (swaption_strike_up_vol_up.NPV() - swaption_strike_up_vol_down.NPV() - swaption_strike_down_vol_up.NPV() + swaption_strike_down_vol_down.NPV())
        / (4.0 * dStrike * dVol)
    ) / 10_000.0

    price_today = original_swaption.NPV()
    original_price = price_today
    ql.Settings.instance().evaluationDate = ql_date + 1
    theta = original_price - original_swaption.NPV()
    charm = dv01 - (original_swaption.delta() / 10_000.0)
    veta = vega - (original_swaption.vega() / 10_000.0)
    ql.Settings.instance().evaluationDate = ql_date

    greeks: Dict[str, float] = {
        "DV01": (abs(dv01) if is_receiver else -abs(dv01)) if is_long else (-abs(dv01) if is_receiver else abs(dv01)),
        "Gamma_01": abs(gamma) if is_long else -abs(gamma),
        "Vega_01": abs(vega) if is_long else -abs(vega),
        "Volga_01": abs(volga) if is_long else -abs(volga),
        "Vanna_01": float(vanna),
        "Theta_1d": -abs(theta) if is_long else abs(theta) * -1,
        "Charm_1d": -abs(charm) if is_long else abs(charm) * -1,
        "Veta_1d": -abs(veta) if is_long else abs(veta) * -1,
    }

    return greeks, price_today


# def calc_swaption_greeks(
#     as_of_date: datetime,
#     expiry_str: str,
#     tail_str: str,
#     swaption_expiry_date: datetime,
#     strike: float,
#     notional: float,
#     ql_overnight_index: ql.Sofr,
#     normal_bp_vol: float,
#     dStrike: float,
#     dVol: float,
#     ql_discount_curve: ql.DiscountCurve,
#     backed_out_vol_ql_swaption: ql.Swaption,
#     backed_out_vol_ql_swaption_pricing_engine: ql.BachelierSwaptionEngine,
#     ql_underlying_pricing_engine: ql.DiscountingSwapEngine,
#     ql_cal: ql.UnitedStates,
#     is_long: bool,
#     swap_type: Literal["Payer", "Receiver"],
#     ql_bday_convention: Literal["Quantlib BDay Conventions"],
#     swaption_underlying_tplus: Optional[str] = "0D",
# ):
#     ql_underlying_ois_strike_up = build_ql_ois(
#         overnight_index=ql_overnight_index,
#         fwd_tenor_str=expiry_str,
#         swap_tenor_str=tail_str,
#         trade_date=swaption_expiry_date,
#         t_plus=swaption_underlying_tplus,
#         fixed_rate=strike + dStrike,
#         notional=notional,
#         swap_type=swap_type,
#     )
#     ql_underlying_ois_strike_up.setPricingEngine(ql_underlying_pricing_engine)
#     ql_underlying_ois_strike_down = build_ql_ois(
#         overnight_index=ql_overnight_index,
#         fwd_tenor_str=expiry_str,
#         swap_tenor_str=tail_str,
#         trade_date=swaption_expiry_date,
#         t_plus=swaption_underlying_tplus,
#         fixed_rate=strike - dStrike,
#         notional=notional,
#         swap_type=swap_type,
#     )
#     ql_underlying_ois_strike_down.setPricingEngine(ql_underlying_pricing_engine)

#     ql_swaption_strike_bumped_up = ql.Swaption(ql_underlying_ois_strike_up, ql.EuropeanExercise(datetime_to_ql_date(swaption_expiry_date)))
#     ql_swaption_strike_bumped_down = ql.Swaption(ql_underlying_ois_strike_down, ql.EuropeanExercise(datetime_to_ql_date(swaption_expiry_date)))
#     ql_swaption_strike_bumped_up.setPricingEngine(backed_out_vol_ql_swaption_pricing_engine)
#     ql_swaption_strike_bumped_down.setPricingEngine(backed_out_vol_ql_swaption_pricing_engine)

#     vol_surface = ql.ConstantSwaptionVolatility(datetime_to_ql_date(as_of_date), ql_cal, ql.Normal, normal_bp_vol, ql.Actual360())
#     vol_handle = ql.RelinkableSwaptionVolatilityStructureHandle()
#     vol_handle.linkTo(vol_surface)

#     # volatility-bumped swaptions.
#     ql_swaption_vol_bumped_up = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=backed_out_vol_ql_swaption,
#         base_vol_handle=vol_handle,
#         shift=dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )
#     ql_swaption_vol_bumped_down = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=backed_out_vol_ql_swaption,
#         base_vol_handle=vol_handle,
#         shift=-dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )
#     ql_swaption_strike_up_vol_up = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=ql_swaption_strike_bumped_up,
#         base_vol_handle=vol_handle,
#         shift=dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )
#     ql_swaption_strike_up_vol_down = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=ql_swaption_strike_bumped_up,
#         base_vol_handle=vol_handle,
#         shift=-dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )
#     ql_swaption_strike_down_vol_up = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=ql_swaption_strike_bumped_down,
#         base_vol_handle=vol_handle,
#         shift=dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )
#     ql_swaption_strike_down_vol_down = build_ql_swaption_with_vol_shift(
#         ql_curve=ql_discount_curve,
#         swaption=ql_swaption_strike_bumped_down,
#         base_vol_handle=vol_handle,
#         shift=-dVol,
#         expiry_str=expiry_str,
#         ql_bday_convention=ql_bday_convention,
#     )

#     greeks_dict, _ = calc_swaption_greeks_helper(
#         ql_date=datetime_to_ql_date(as_of_date),
#         original_swaption=backed_out_vol_ql_swaption,
#         swaption_strike_bumped_up=ql_swaption_strike_bumped_up,
#         swaption_strike_bumped_down=ql_swaption_strike_bumped_down,
#         swaption_vol_bumped_up=ql_swaption_vol_bumped_up,
#         swaption_vol_bumped_down=ql_swaption_vol_bumped_down,
#         swaption_strike_up_vol_up=ql_swaption_strike_up_vol_up,
#         swaption_strike_up_vol_down=ql_swaption_strike_up_vol_down,
#         swaption_strike_down_vol_up=ql_swaption_strike_down_vol_up,
#         swaption_strike_down_vol_down=ql_swaption_strike_down_vol_down,
#         dStrike=dStrike,
#         dVol=dVol,
#         is_receiver=swap_type == "Receiver",
#         is_long=is_long,
#     )

#     return greeks_dict


# def calc_swaption_book_greeks(
#     as_of_date: datetime,
#     swaptions: List[Tuple[ql.Swaption, Annotated[bool, "Is_Long"]]],
#     dStrike: float,
#     dVol: float,
#     ql_discount_curve: ql.YieldTermStructureHandle,
#     vol_handle: ql.SwaptionVolatilityStructureHandle,
#     ql_swaption_pricing_engine: ql.BachelierSwaptionEngine,
#     ql_underlying_pricing_engine: ql.DiscountingSwapEngine,
#     ql_overnight_index: ql.Sofr,
#     ql_bday_convention,
#     swaption_underlying_tplus: Optional[str] = "0D",
# ) -> Dict[str, float]:
#     aggregated_greeks = {
#         "DV01": 0.0,
#         "Gamma_01": 0.0,
#         "Vega_01": 0.0,
#         "Volga_01": 0.0,
#         "Vanna_01": 0.0,
#         "Theta_1d": 0.0,
#         "Charm_1d": 0.0,
#         "Veta_1d": 0.0,
#     }
#     book_npv = 0.0

#     for swaption, is_long in swaptions:
#         underlying_swap: ql.OvernightIndexedSwap = swaption.underlying()
#         strike = underlying_swap.fixedRate()
#         try:
#             notional = underlying_swap.nominal()
#         except AttributeError:
#             notional = abs(underlying_swap.leg(0)[0].amount())

#         effective_date = underlying_swap.startDate()
#         maturity_date = underlying_swap.maturityDate()
#         underlying_tenor_days = maturity_date - effective_date

#         exercise_qldate = swaption.exercise().date(0)
#         days_to_expiry = exercise_qldate - datetime_to_ql_date(as_of_date)
#         trade_date = datetime(exercise_qldate.year(), exercise_qldate.month(), exercise_qldate.dayOfMonth())
#         swap_type = "Receiver" if underlying_swap.fixedLegBPS() > 0 else "Payer"

#         bumped_underlying_up = build_ql_ois(
#             fwd_tenor_str=days_to_expiry,
#             swap_tenor_str=underlying_tenor_days,
#             overnight_index=ql_overnight_index,
#             trade_date=trade_date,
#             fixed_rate=(strike * 100) + dStrike,
#             notional=notional,
#             swap_type=swap_type,
#             t_plus=swaption_underlying_tplus,
#         )
#         bumped_underlying_down = build_ql_ois(
#             fwd_tenor_str=days_to_expiry,
#             swap_tenor_str=underlying_tenor_days,
#             overnight_index=ql_overnight_index,
#             trade_date=trade_date,
#             fixed_rate=(strike * 100) - dStrike,
#             notional=notional,
#             swap_type=swap_type,
#             t_plus=swaption_underlying_tplus,
#         )
#         bumped_underlying_up.setPricingEngine(ql_underlying_pricing_engine)
#         bumped_underlying_down.setPricingEngine(ql_underlying_pricing_engine)

#         swaption_strike_bumped_up = ql.Swaption(bumped_underlying_up, swaption.exercise())
#         swaption_strike_bumped_down = ql.Swaption(bumped_underlying_down, swaption.exercise())
#         swaption_strike_bumped_up.setPricingEngine(ql_swaption_pricing_engine)
#         swaption_strike_bumped_down.setPricingEngine(ql_swaption_pricing_engine)

#         swaption_vol_bumped_up = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption,
#             base_vol_handle=vol_handle,
#             shift=dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )
#         swaption_vol_bumped_down = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption,
#             base_vol_handle=vol_handle,
#             shift=-dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )
#         swaption_strike_up_vol_up = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption_strike_bumped_up,
#             base_vol_handle=vol_handle,
#             shift=dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )
#         swaption_strike_up_vol_down = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption_strike_bumped_up,
#             base_vol_handle=vol_handle,
#             shift=-dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )
#         swaption_strike_down_vol_up = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption_strike_bumped_down,
#             base_vol_handle=vol_handle,
#             shift=dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )
#         swaption_strike_down_vol_down = build_ql_swaption_with_vol_shift(
#             ql_curve=ql_discount_curve,
#             swaption=swaption_strike_bumped_down,
#             base_vol_handle=vol_handle,
#             shift=-dVol,
#             expiry_str=days_to_expiry,
#             ql_bday_convention=ql_bday_convention,
#         )

#         greeks, curr_npv = calc_swaption_greeks_helper(
#             ql_date=datetime_to_ql_date(as_of_date),
#             original_swaption=swaption,
#             swaption_strike_bumped_up=swaption_strike_bumped_up,
#             swaption_strike_bumped_down=swaption_strike_bumped_down,
#             swaption_vol_bumped_up=swaption_vol_bumped_up,
#             swaption_vol_bumped_down=swaption_vol_bumped_down,
#             swaption_strike_up_vol_up=swaption_strike_up_vol_up,
#             swaption_strike_up_vol_down=swaption_strike_up_vol_down,
#             swaption_strike_down_vol_up=swaption_strike_down_vol_up,
#             swaption_strike_down_vol_down=swaption_strike_down_vol_down,
#             dStrike=dStrike,
#             dVol=dVol,
#             is_receiver=True if swap_type == "Receiver" else False,
#             is_long=is_long,
#         )

#         for key in aggregated_greeks:
#             aggregated_greeks[key] += greeks.get(key, 0.0)
#         book_npv += curr_npv

#     aggregated_greeks["Book_NPV"] = book_npv
#     return aggregated_greeks


def build_ql_interpolated_vol_cube(
    vol_cube: SCube | Dict[int | float, pd.DataFrame],
    ql_swap_index: ql.SwapIndex,
    ql_calendar: ql.Calendar,
    ql_day_counter: ql.DayCounter,
    ql_bday_convention: Optional[Annotated[int, "ql.BusinessConvention"]] = ql.ModifiedFollowing,
    pre_calibrate: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True
) -> ql.InterpolatedSwaptionVolatilityCube:
    atm_vol_grid = vol_cube[0]
    expiries = [ql.Period(e) for e in atm_vol_grid.index]
    tails = [ql.Period(t) for t in atm_vol_grid.columns]

    ql_atm_swaption_vol_matrix = ql.SwaptionVolatilityMatrix(
        ql_calendar,
        ql_bday_convention,
        expiries,
        tails,
        ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_grid.values]),
        ql_day_counter,
        False,
        ql.Normal,
    )

    vol_spreads = []
    strike_spreads = [float(k) / 10_000 for k in vol_cube.keys()]
    strike_offsets = sorted(vol_cube.keys(), key=lambda x: int(x))
    for option_tenor in atm_vol_grid.index:
        for swap_tenor in atm_vol_grid.columns:
            vol_spread_row = [
                ql.QuoteHandle(ql.SimpleQuote((vol_cube[strike].loc[option_tenor, swap_tenor] - atm_vol_grid.loc[option_tenor, swap_tenor]) / 10_000))
                for strike in strike_offsets
            ]
            vol_spreads.append(vol_spread_row)

    ql_vol_cube = ql.InterpolatedSwaptionVolatilityCube(
        ql.SwaptionVolatilityStructureHandle(ql_atm_swaption_vol_matrix),
        expiries,
        tails,
        strike_spreads,
        vol_spreads,
        ql_swap_index,
        ql_swap_index,
        True,
    )

    if pre_calibrate:
        pre_calibration_expiry = 0.25
        pre_calibration_tails = 10
        pre_calibration_strike = 0.04
        try:
            _ = ql_vol_cube.volatility(pre_calibration_expiry, pre_calibration_tails, pre_calibration_strike, True)
        except Exception:
            pass

    if enable_extrapolation:
        ql_vol_cube.enableExtrapolation()
    return ql_vol_cube


def _build_sabr_guess(
    sabr_params_dict: Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]], spread_opt_tenors: List[ql.Period], spread_swap_tenors: List[ql.Period]
):
    guess_list = []
    for opt in spread_opt_tenors:
        opt_str = period_to_string(opt)
        for swap in spread_swap_tenors:
            swap_str = period_to_string(swap)
            key_candidate = opt_str + "x" + swap_str
            if key_candidate in sabr_params_dict:
                params = sabr_params_dict[key_candidate]
            else:
                candidates = []
                for key in sabr_params_dict:
                    if key.endswith("x" + swap_str):
                        candidate_opt_str = key.split("x")[0]
                        candidate_opt = parse_tenor_string(candidate_opt_str)
                        diff = abs(period_to_months(candidate_opt) - period_to_months(opt))
                        candidates.append((diff, key))
                if candidates:
                    best_key = sorted(candidates, key=lambda x: x[0])[0][1]
                    params = sabr_params_dict[best_key]
                else:
                    params = {"alpha": 0.2, "beta": 0.5, "nu": 0.4, "rho": 0.0}

            guess_tuple = (
                ql.QuoteHandle(ql.SimpleQuote(params["alpha"])),
                ql.QuoteHandle(ql.SimpleQuote(params["beta"])),
                ql.QuoteHandle(ql.SimpleQuote(params["nu"])),
                ql.QuoteHandle(ql.SimpleQuote(params["rho"])),
            )
            guess_list.append(guess_tuple)

    return guess_list


def build_ql_sabr_vol_cube(
    vol_cube: Dict[int, pd.DataFrame],
    sabr_params_dict: Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]],
    ql_swap_index: ql.SwapIndex,
    ql_calendar: ql.Calendar,
    ql_day_counter: ql.DayCounter,
    ql_bday_convention: Optional[Annotated[int, "ql.BusinessConvention"]] = ql.ModifiedFollowing,
    pre_calibrate: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True,
) -> ql.SabrSwaptionVolatilityCube:
    atm_vol_grid = vol_cube[0]

    spread_opt_tenors = [ql.Period(e) for e in atm_vol_grid.index]
    spread_swap_tenors = [ql.Period(t) for t in atm_vol_grid.columns]

    atm_swaption_vol_matrix = ql.SwaptionVolatilityMatrix(
        ql_calendar,
        ql_bday_convention,
        spread_opt_tenors,
        spread_swap_tenors,
        ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_grid.values]),
        ql_day_counter,
        False,
        ql.Normal,
    )

    vol_spreads = []
    strike_spreads = [float(k) / 10_000 for k in vol_cube.keys()]
    strike_offsets = sorted(vol_cube.keys(), key=lambda x: int(x))
    for option_tenor in atm_vol_grid.index:
        for swap_tenor in atm_vol_grid.columns:
            vol_spread_row = [
                ql.QuoteHandle(ql.SimpleQuote((vol_cube[strike].loc[option_tenor, swap_tenor] - atm_vol_grid.loc[option_tenor, swap_tenor]) / 10_000))
                for strike in strike_offsets
            ]
            vol_spreads.append(vol_spread_row)

    guess = _build_sabr_guess(sabr_params_dict, spread_opt_tenors, spread_swap_tenors)
    is_parameter_fixed = (False, True, False, False)

    # https://quant.stackexchange.com/questions/75537/sabrswaptionvolcube-class-in-quantilib-python
    ql_sabr_vol_cube = ql.SabrSwaptionVolatilityCube(
        ql.SwaptionVolatilityStructureHandle(atm_swaption_vol_matrix),
        spread_opt_tenors,
        spread_swap_tenors,
        strike_spreads,
        vol_spreads,
        ql_swap_index,
        ql_swap_index,
        False,
        guess,
        is_parameter_fixed,
        True,
    )

    if pre_calibrate:
        pre_calibration_expiry = 0.25
        pre_calibration_tails = 10
        pre_calibration_strike = 0.04
        try:
            _ = ql_sabr_vol_cube.volatility(pre_calibration_expiry, pre_calibration_tails, pre_calibration_strike, True)
        except Exception:
            pass

    if enable_extrapolation:
        ql_sabr_vol_cube.enableExtrapolation()
    return ql_sabr_vol_cube
