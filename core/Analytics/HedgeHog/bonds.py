from typing import Dict, List, Optional

import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d

from utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime, ql_period_to_years


def make_bond_from_observed_row(
    row: pd.Series,
    settlement_days: Optional[int] = 1,
    issue_date_col: Optional[str] = "issue_date",
    mat_date_col: Optional[str] = "maturity_date",
    coupon_col: Optional[str] = "coupon",
    ql_bday_convention: Optional[int] = ql.ModifiedFollowing,
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    ql_day_counter: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),
):
    ust_issue_date = pd.to_datetime(row[issue_date_col])
    ust_mat_date = pd.to_datetime(row[mat_date_col])
    ust_coup = pd.to_numeric(row[coupon_col])
    ql_schedule = ql.Schedule(
        datetime_to_ql_date(ust_issue_date),
        datetime_to_ql_date(ust_mat_date),
        ql.Period(ql.Semiannual),
        ql_cal,
        ql_bday_convention,
        ql_bday_convention,
        ql.DateGeneration.Backward,
        False,
    )
    return ql.FixedRateBond(settlement_days, 100, ql_schedule, [ust_coup / 100], ql_day_counter)


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

    return metrics
