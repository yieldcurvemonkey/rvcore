from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d

from core.Analytics.HedgeHog.Positions import LinearPosition
from core.Analytics.HedgeHog.utils import linear_solve_for_risk_weighted_notionals

from core.utils.ql_utils import ql_period_to_years, ql_date_to_datetime


def calc_bond_metrics(
    ql_bond: ql.FixedRateBond,
    ql_curve: ql.DiscountCurve | ql.YieldTermStructure,
    scipy_spline: interp1d,
    roll_horizons: Optional[List[ql.Period]] = [ql.Period("1M")],
    gc_repo_rates_dict: Optional[Dict[ql.Period, int]] = None,
    observed_bond_row: Optional[pd.Series] = None,
    observed_price_col: Optional[str] = "eod_price",
    observed_ytm_col: Optional[str] = "eod_ytm",
    observed_coup_col: Optional[str] = "coupon",
):
    if not gc_repo_rates_dict:
        gc_repo_rates_dict = {}

    ql_day_counter: ql.DayCounter = ql_bond.dayCounter()
    ql_calendar: ql.Calendar = ql_bond.calendar()
    ql_compounding = ql.Compounded
    ql_frequency = ql_bond.frequency()

    curve_handle = ql.YieldTermStructureHandle(ql_curve)
    engine = ql.DiscountingBondEngine(curve_handle)
    ql_bond.setPricingEngine(engine)

    computed_npv = ql_bond.NPV()
    computed_dirty_price = ql_bond.dirtyPrice()
    computed_clean_price = ql_bond.cleanPrice()
    computed_yield = ql.BondFunctions.bondYield(ql_bond, computed_dirty_price, ql_day_counter, ql_compounding, ql_frequency)
    modified_duration = ql.BondFunctions.duration(ql_bond, computed_yield, ql_day_counter, ql_compounding, ql_frequency)

    bpv = ql.BondFunctions.basisPointValue(ql_bond, computed_yield, ql_day_counter, ql_compounding, ql_frequency)
    ql_bond_notional = ql_bond.notional()
    if ql_bond_notional:
        bpv = bpv / ql_bond_notional
        # bpv += bpv * 1_000_000

    convexity = ql.BondFunctions.convexity(ql_bond, computed_yield, ql_day_counter, ql_compounding, ql_frequency)
    z_spread = ql.BondFunctions.zSpread(
        ql_bond,
        computed_dirty_price,
        curve_handle.currentLink(),
        ql_day_counter,
        ql_compounding,
        ql_frequency,
        ql_bond.settlementDate(),
    )

    as_of_ql_date = ql_curve.nodes()[0][0]
    ttm = ql_day_counter.yearFraction(as_of_ql_date, ql_bond.maturityDate())

    computed_yield *= 100
    metrics = {
        "observed_price": observed_bond_row[observed_price_col] if observed_bond_row is not None else None,
        "observed_ytm": observed_bond_row[observed_ytm_col] if observed_bond_row is not None else None,
        "model_npv": computed_npv,
        "model_dirty_price": computed_dirty_price,
        "model_clean_price": computed_clean_price,
        "model_ytm": computed_yield,
        "modified_duration": modified_duration,
        "bpv": bpv,
        "convexity": convexity,
        "z_spread": z_spread,
        "time_to_maturity_years": ttm,
    }

    for roll_horizon in roll_horizons:
        rolled_date = ql_calendar.advance(as_of_ql_date, roll_horizon)

        rolled_yield = scipy_spline(ql_day_counter.yearFraction(rolled_date, ql_bond.maturityDate()))
        roll_bps_running = (scipy_spline(ttm) - rolled_yield) * 100

        metrics[f"{roll_horizon}_rolled_ytm"] = rolled_yield
        metrics[f"{roll_horizon}_roll"] = roll_bps_running

        if observed_bond_row is not None and roll_horizon in gc_repo_rates_dict:
            cost_of_carry = observed_bond_row[observed_coup_col] - (observed_bond_row[observed_price_col] * (gc_repo_rates_dict[roll_horizon] / 100))
            metrics[f"{roll_horizon}_cost_of_carry"] = cost_of_carry

            repo = (
                ql.InterestRate(gc_repo_rates_dict[roll_horizon] / 100, ql.Actual360(), ql.Simple, ql.Annual)
                .equivalentRate(ql_compounding, ql_frequency, ql_period_to_years(roll_horizon, ql_day_counter, ql_date_to_datetime(as_of_ql_date)))
                .rate()
                * 100
            )
            carry_bps_running = observed_bond_row[observed_ytm_col] - repo
            metrics[f"{roll_horizon}_carry"] = carry_bps_running

            metrics[f"{roll_horizon}_carry_and_roll_bps_running"] = carry_bps_running + roll_bps_running

        try:
            roll_years = ql_day_counter.yearFraction(as_of_ql_date, ql_calendar.advance(as_of_ql_date, roll_horizon))
            adjusted_cashflows: List[ql.CashFlow] = []
            for cf in ql_bond.cashflows():
                if not cf.hasOccurred(as_of_ql_date):
                    t = ql_day_counter.yearFraction(as_of_ql_date, cf.date())
                    if t > roll_years:
                        adjusted_date = ql_calendar.advance(cf.date(), -roll_horizon)
                        if adjusted_date >= rolled_date:
                            adjusted_cf = ql.SimpleCashFlow(cf.amount(), adjusted_date)
                            adjusted_cashflows.append(adjusted_cf)

            if not adjusted_cashflows:
                coupon_adjusted_rolldown = float("nan")
            else:
                P_prime = 0.0
                for acf in adjusted_cashflows:
                    t_adj = ql_day_counter.yearFraction(as_of_ql_date, acf.date())
                    d_adj = ql_curve.discount(acf.date())
                    P_prime += acf.amount() * d_adj * np.exp(-z_spread * t_adj)

                face_amount = ql_bond.notional() if hasattr(ql_bond, "notional") else 100.0
                maturity_roll = ql_calendar.advance(ql_bond.maturityDate(), -roll_horizon)
                rolled_bond = ql.Bond(0, ql_calendar, face_amount, maturity_roll, rolled_date, adjusted_cashflows)

                try:
                    y_prime = ql.BondFunctions.bondYield(rolled_bond, P_prime, ql_day_counter, ql_compounding, ql_frequency, rolled_date) * 100
                except Exception as e:
                    y_prime = float("nan")

                coupon_adjusted_rolldown = (computed_yield - y_prime) * 100
                metrics[f"{roll_horizon}_coupon_adjusted_rolldown"] = coupon_adjusted_rolldown
        except:
            metrics[f"{roll_horizon}_coupon_adjusted_rolldown"] = None

    return metrics


def calc_swap_metrics(
    ql_swap: ql.Swap | ql.VanillaSwap | ql.OvernightIndexedSwap,
    ql_curve: ql.YieldTermStructure,
    roll_horizons: List[ql.Period] = [ql.Period(30, ql.Days)],
    observed_swap_row: Optional[pd.Series] = None,
    observed_rate_col: Optional[str] = "eod_rate",
    use_observed: Optional[bool] = False,
) -> dict:
    original_eval_date = ql.Settings.instance().evaluationDate
    is_ois = isinstance(ql_swap, ql.OvernightIndexedSwap)

    ql_calendar: ql.Calendar = ql_curve.calendar()
    ql_day_counter: ql.DayCounter = ql_curve.dayCounter()
    ql_curve_handle = ql.YieldTermStructureHandle(ql_curve)
    ql_eff_date: ql.Date = ql_swap.startDate()
    ql_mat_date: ql.Date = ql_swap.maturityDate()
    ql_on_index: ql.SwapIndex = ql_swap.overnightIndex() if is_ois else ql_swap.iborIndex()
    ql_disc_engine = ql.DiscountingSwapEngine(ql_curve_handle)
    ql_swap.setPricingEngine(ql_disc_engine)

    computed_npv = ql_swap.NPV()
    computed_fair_rate = ql_swap.fairRate() * 100
    bpv = ql_swap.fixedLegBPS()

    ql_swap_notional = ql_swap.nominal()
    if ql_swap_notional:
        bpv = bpv / ql_swap_notional
        # bpv += bpv * 1_000_000

    ttm = ql_day_counter.yearFraction(original_eval_date, ql_mat_date)
    tts = ql_day_counter.yearFraction(original_eval_date, ql_eff_date)

    metrics = {
        "observed_rate": observed_swap_row[observed_rate_col] if observed_swap_row is not None else None,
        "curve_npv": computed_npv,
        "curve_fair_rate": computed_fair_rate,
        "bpv": bpv,
        "time_to_maturity_years": ttm,
        "time_to_start_years": tts,
    }

    yield_to_use = observed_swap_row[observed_rate_col] if observed_swap_row is not None and use_observed else computed_fair_rate

    for roll_horizon in roll_horizons:
        fwd_eff_date = ql_calendar.advance(ql_eff_date, roll_horizon)
        rolled_mat_date = ql_calendar.advance(ql_mat_date, -roll_horizon)

        if is_ois:
            if tts > 0:
                spot_rolled_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    overnightIndex=ql_on_index,
                    effectiveDate=ql_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )
            else:
                fwd_rolled_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    overnightIndex=ql_on_index,
                    effectiveDate=fwd_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )
                spot_rolled_swap: ql.OvernightIndexedSwap = ql.MakeOIS(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    overnightIndex=ql_on_index,
                    effectiveDate=ql_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )
        else:
            if tts > 0:
                spot_rolled_swap: ql.VanillaSwap = ql.MakeVanillaSwap(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    iborIndex=ql_on_index,
                    effectiveDate=ql_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )
            else:
                fwd_rolled_swap: ql.VanillaSwap = ql.MakeVanillaSwap(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    iborIndex=ql_on_index,
                    effectiveDate=fwd_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )
                spot_rolled_swap: ql.VanillaSwap = ql.MakeVanillaSwap(
                    swapTenor=ql.Period("-0D"),
                    fixedRate=-0,
                    iborIndex=ql_on_index,
                    effectiveDate=ql_eff_date,
                    terminationDate=rolled_mat_date,
                    pricingEngine=ql_disc_engine,
                )

        metrics[f"{roll_horizon}_rolled_rate"] = spot_rolled_swap.fairRate() * 100
        metrics[f"{roll_horizon}_fwd_rolled_rate"] = fwd_rolled_swap.fairRate() * 100 if tts <= 0 else None
        metrics[f"{roll_horizon}_carry"] = (metrics[f"{roll_horizon}_fwd_rolled_rate"] - yield_to_use) * 100 if tts <= 0 else None
        metrics[f"{roll_horizon}_roll"] = (yield_to_use - metrics[f"{roll_horizon}_rolled_rate"]) * 100
        metrics[f"{roll_horizon}_carry_and_roll_bps_running"] = (
            metrics[f"{roll_horizon}_carry"] + metrics[f"{roll_horizon}_roll"] if tts <= 0 else metrics[f"{roll_horizon}_roll"]
        )

    ql_swap.setPricingEngine(ql_disc_engine)
    return metrics


def get_instrument_metrics(
    linear_pos: LinearPosition,
    roll_horizons: Optional[List[ql.Period]] = [ql.Period("1M"), ql.Period("3M")],
    gc_repo_rates_dict: Optional[Dict[ql.Period, int]] = None,
) -> float:
    if isinstance(linear_pos.ql_instrument, ql.FixedRateBond):
        return calc_bond_metrics(
            ql_bond=linear_pos.ql_instrument,
            ql_curve=linear_pos.ql_curve,
            scipy_spline=linear_pos.scipy_spline,
            roll_horizons=roll_horizons,
            observed_bond_row=linear_pos.obs_bond_row,
            gc_repo_rates_dict=gc_repo_rates_dict,
        )
    elif (
        isinstance(linear_pos.ql_instrument, ql.Swap)
        or isinstance(linear_pos.ql_instrument, ql.VanillaSwap)
        or isinstance(linear_pos.ql_instrument, ql.OvernightIndexedSwap)
    ):
        return calc_swap_metrics(ql_swap=linear_pos.ql_instrument, ql_curve=linear_pos.ql_curve, roll_horizons=roll_horizons)
    else:
        raise NotImplementedError("Metric calculation not implemented for instrument type: " + str(type(linear_pos.ql_instrument)))


def calc_linear_book_metrics(
    book: List[LinearPosition],
    roll_horizons: Optional[List[ql.Period]] = [ql.Period("1M"), ql.Period("3M")],
    gc_repo_rates_dict: Optional[Dict[ql.Period, int]] = None,
) -> Dict[str, Dict]:
    pos_metrics: Dict[str, Dict] = {}
    book_metrics: Dict[str, float | dict] = {}
    pos_unit: Dict[str, float] = {}  # instrument BPV per unit notional (ignoring risk weight)

    risk_weights_to_solve = []
    bpvs_to_solve = []
    constrained_leg_index = None
    constrained_leg_notional = None
    pos_ids = []

    for i, pos in enumerate(book):
        metrics = get_instrument_metrics(pos, roll_horizons=roll_horizons, gc_repo_rates_dict=gc_repo_rates_dict)
        bpv = metrics.get("bpv", np.nan)
        pos_unit[pos.id] = bpv
        r = pos.risk_weight
        r_sign = np.sign(r)

        model_ytm = metrics.get("observed_ytm", metrics.get("model_ytm", metrics.get("curve_fair_rate")))
        weighted_spread = model_ytm * r if model_ytm is not None else np.nan

        carry_and_roll_dict = {}
        for rh in roll_horizons:
            key = f"{rh}_carry_and_roll_bps_running"
            carry_and_roll_dict[str(rh)] = abs(metrics.get(key, np.nan)) * r_sign

        pos_metrics[pos.id] = {
            "risk_weight": r,
            "notional": abs(pos.notional) * r_sign if pos.notional else None,
            "unit_bpv": abs(bpv) * r_sign,
            "effective_bpv": abs(pos.notional * bpv * r) * r_sign if pos.notional is not None else None,
            "model_ytm": model_ytm,
            "weighted_ytm": weighted_spread,
            "carry_and_roll_bps_running": carry_and_roll_dict,
        }

        risk_weights_to_solve.append(pos.risk_weight)
        bpvs_to_solve.append(bpv)
        pos_ids.append(pos.id)
        if pos.notional and not constrained_leg_index:
            constrained_leg_index = i
            constrained_leg_notional = pos.notional

    if constrained_leg_index is None:
        raise ValueError(f"Underdetermined System - must pass in a notional value to one of the legs in the book")

    risk_weighted_notionals = dict(
        zip(
            pos_ids,
            linear_solve_for_risk_weighted_notionals(
                risk_weights=risk_weights_to_solve,
                bpvs=bpvs_to_solve,
                constrained_leg_index=constrained_leg_index,
                constrained_leg_notional=constrained_leg_notional,
            ),
        )
    )

    total_notional = 0.0
    total_risk_weights = 0.0
    total_weighted_carry_and_roll = {}
    weighted_spread = 0.0
    for pos in book:
        risk_weighted_notional = abs(risk_weighted_notionals[pos.id]) * np.sign(pos.risk_weight)
        pos_metrics[pos.id]["notional"] = risk_weighted_notional
        pos_metrics[pos.id]["effective_bpv"] = abs(risk_weighted_notional) * pos_metrics[pos.id]["unit_bpv"]

        total_notional += risk_weighted_notional
        total_risk_weights += pos.risk_weight
        weighted_spread += pos_metrics[pos.id]["weighted_ytm"]

        for roll_horizon in roll_horizons:
            horizon_str = str(roll_horizon)
            if horizon_str not in total_weighted_carry_and_roll:
                total_weighted_carry_and_roll[horizon_str] = 0
            total_weighted_carry_and_roll[horizon_str] += (
                abs(pos_metrics[pos.id]["carry_and_roll_bps_running"][str(roll_horizon)]) * pos_metrics[pos.id]["risk_weight"]
            )

    book_metrics["book_effective_bpv"] = sum(pos_metrics[p.id]["effective_bpv"] for p in book if pos_metrics[p.id]["effective_bpv"] is not None)
    book_metrics["book_notional"] = total_notional
    book_metrics["weighted_spread"] = (weighted_spread) * 100
    for roll_horizon in roll_horizons:
        book_metrics[f"weighted_{roll_horizon}_carry_and_roll_bps_running"] = total_weighted_carry_and_roll[str(roll_horizon)]

    return {"position_metrics": pos_metrics, "book_metrics": book_metrics}
