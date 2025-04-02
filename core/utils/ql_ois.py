import QuantLib as ql
import pandas as pd
from datetime import datetime
from typing import Optional, List, Literal, Optional, Literal
from utils.ql_utils import datetime_to_ql_date


def build_ql_ois(
    swap_tenor_str: str | int,
    overnight_index: ql.Sofr,
    trade_date: datetime,
    fixed_rate: float,
    notional: float,
    swap_type: Literal["Payer", "Receiver"],
    cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.SOFR),
    fwd_tenor_str: Optional[str | int] = "0D",
    t_plus: Optional[str] = "2D",
) -> ql.OvernightIndexedSwap:
    effective_date = cal.advance(datetime_to_ql_date(trade_date), ql.Period(t_plus))
    return ql.MakeOIS(
        swapTenor=ql.Period(swap_tenor_str) if type(swap_tenor_str) == str else ql.Period(swap_tenor_str, ql.Days),
        overnightIndex=overnight_index,
        fixedRate=fixed_rate / 100,
        fwdStart=ql.Period("0D" if fwd_tenor_str == "Spot" else fwd_tenor_str) if type(fwd_tenor_str) == str else ql.Period(fwd_tenor_str, ql.Days),
        swapType=ql.OvernightIndexedSwap.Payer if swap_type == "Payer" else ql.OvernightIndexedSwap.Receiver,
        effectiveDate=effective_date,
        paymentAdjustmentConvention=ql.ModifiedFollowing,
        paymentLag=2,
        fixedLegDayCount=ql.Actual360(),
        nominal=notional,
    )


def build_ql_sofr_ois(
    fwd_tenor_str: str,
    swap_tenor_str: str,
    swap_type: Literal["pay", "rec"],
    ql_discount_curve: ql.DiscountCurve,
) -> ql.OvernightIndexedSwap:
    ql_sofr_ois: ql.OvernightIndexedSwap = ql.MakeOIS(
        swapTenor=ql.Period(swap_tenor_str),
        overnightIndex=ql.Sofr(ql.YieldTermStructureHandle(ql_discount_curve)),
        fwdStart=ql.Period("0D" if fwd_tenor_str == "Spot" else fwd_tenor_str) if type(fwd_tenor_str) == str else ql.Period(fwd_tenor_str, ql.Days),
        swapType=ql.OvernightIndexedSwap.Payer if swap_type == "pay" else ql.OvernightIndexedSwap.Receiver,
        paymentAdjustmentConvention=ql.ModifiedFollowing,
        paymentLag=2,
        fixedLegDayCount=ql.Actual360(),
        fixedRate=0
    )
    ql_sofr_ois.setPricingEngine(ql.DiscountingSwapEngine(ql.YieldTermStructureHandle(ql_discount_curve)))
    return ql_sofr_ois
