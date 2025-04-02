from datetime import datetime
from typing import List, Optional

import pandas as pd
import QuantLib as ql

from core.utils.ql_utils import datetime_to_ql_date


def calc_swap_metrics(
    curve_date: datetime,
    ql_swap: ql.Swap | ql.VanillaSwap | ql.OvernightIndexedSwap,
    ql_curve: ql.YieldTermStructure,
    roll_horizons: List[ql.Period] = [ql.Period(30, ql.Days)],
    observed_swap_row: Optional[pd.Series] = None,
    observed_rate_col: Optional[str] = "eod_rate",
    use_observed: Optional[bool] = False,
) -> dict:
    original_eval_date = datetime_to_ql_date(curve_date)
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
        "bpv_per_mm": bpv * 1_000_000, 
        "time_to_maturity_years": ttm,
        "time_to_start_years": tts,
    }

    yield_to_use = observed_swap_row[observed_rate_col] if observed_swap_row is not None and use_observed else computed_fair_rate

    for roll_horizon in roll_horizons:
        fwd_eff_date = ql_calendar.advance(ql_eff_date, roll_horizon)
        rolled_mat_date = ql_calendar.advance(ql_mat_date, -roll_horizon)

        if is_ois:
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
        metrics[f"{roll_horizon}_fwd_rolled_rate"] = fwd_rolled_swap.fairRate() * 100
        metrics[f"{roll_horizon}_carry"] = (metrics[f"{roll_horizon}_fwd_rolled_rate"] - yield_to_use) * 100
        metrics[f"{roll_horizon}_roll"] = (yield_to_use - metrics[f"{roll_horizon}_rolled_rate"]) * 100
        metrics[f"{roll_horizon}_carry_and_roll_bps_running"] = metrics[f"{roll_horizon}_carry"] + metrics[f"{roll_horizon}_roll"]

    ql_swap.setPricingEngine(ql_disc_engine)
    return metrics
