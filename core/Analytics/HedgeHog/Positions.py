from dataclasses import dataclass
from typing import Optional

import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d

from core.utils.ql_utils import datetime_to_ql_date


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


@dataclass
class LinearPosition:
    ql_curve: ql.YieldTermStructure
    risk_weight: float
    id: float
    ql_instrument: Optional[ql.FixedRateBond | ql.Swap | ql.VanillaSwap | ql.OvernightIndexedSwap] = None
    obs_bond_row: Optional[pd.Series] = None
    scipy_spline: Optional[interp1d] = None
    notional: Optional[float] = None
    issue_date_col: Optional[str] = "issue_date"
    mat_date_col: Optional[str] = "maturity_date"
    coupon_col: Optional[str] = "coupon"
    ql_bday_convention: Optional[int] = ql.ModifiedFollowing
    ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    ql_day_counter: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365)

    def __post_init__(self):
        assert self.ql_instrument or self.obs_bond_row is not None, "MUST PASS IN BUILT 'ql_instrument' OR 'obs_bond_row'"
        if isinstance(self.ql_instrument, ql.FixedRateBond) or self.obs_bond_row is not None:
            assert self.scipy_spline, "MUST PASS IN SCIPY PAR SPLINE"

        if self.obs_bond_row is not None:
            self.ql_instrument = make_bond_from_observed_row(
                row=self.obs_bond_row,
                issue_date_col=self.issue_date_col,
                mat_date_col=self.mat_date_col,
                coupon_col=self.coupon_col,
                ql_bday_convention=self.ql_bday_convention,
                ql_cal=self.ql_cal,
                ql_day_counter=self.ql_day_counter,
            )

        if self.notional is None:
            self.notional = self.check_ql_notional()

        self.ql_instrument.setPricingEngine(
            ql.DiscountingBondEngine(ql.YieldTermStructureHandle(self.ql_curve))
            if isinstance(self.ql_instrument, ql.FixedRateBond)
            else self.ql_instrument.setPricingEngine(ql.DiscountingSwapEngine(ql.YieldTermStructureHandle(self.ql_curve)))
        )

    def check_ql_notional(self) -> float:
        if self.notional is not None:
            return self.notional

        try:
            if isinstance(self.ql_instrument, ql.FixedRateBond):
                bond_notional = self.ql_instrument.notional()
                if bond_notional and bond_notional > 100:
                    return bond_notional
            elif (
                isinstance(self.ql_instrument, ql.Swap) or isinstance(self.ql_instrument, ql.VanillaSwap) or isinstance(self.ql_instrument, ql.OvernightIndexedSwap)
            ):
                nomial = self.ql_instrument.nominal()
                if nomial and nomial > 100:
                    return nomial

        except Exception:
            return None
