import QuantLib as ql
from typing import Dict

CME_SWAP_CURVE_QL_PARAMS: Dict[str, Dict[str, ql.SwapIndex | bool | ql.DayCounter | ql.Calendar | int | int | ql.Period | ql.Currency | int | int]] = {
    "USD-SOFR-1D": {
        "swapIndex": ql.Sofr,
        "is_ois": True,
        "dayCounter": ql.Actual360(),
        "calendar": ql.UnitedStates(ql.UnitedStates.SOFR),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,
        "period": ql.Period("1D"),
        "currency": ql.USDCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "USD-FEDFUNDS": {
        "swapIndex": ql.FedFunds,
        "is_ois": True,
        "dayCounter": ql.Actual360(),
        "calendar": ql.UnitedStates(ql.UnitedStates.FederalReserve),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,
        "period": ql.Period("1D"),
        "currency": ql.USDCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "USD-OIS-1D": {
        "swapIndex": ql.FedFunds,
        "is_ois": True,
        "dayCounter": ql.Actual360(),
        "calendar": ql.UnitedStates(ql.UnitedStates.FederalReserve),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,
        "period": ql.Period("1D"),
        "currency": ql.USDCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "CAD-CORRA-1D": {
        "swapIndex": ql.Corra,
        "is_ois": True,
        "dayCounter": ql.Actual365Fixed(),  # Correct
        "calendar": ql.Canada(ql.Canada.TSX),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Semiannual,  # Changed from Annual — CORRA OIS typically pays semiannually
        "period": ql.Period("1D"),
        "currency": ql.CADCurrency(),
        "paymentLag": 2,
        "settlementDays": 1,  # CAD OIS is typically T+1
    },
    "GBP-SONIA-1D": {
        "swapIndex": ql.Sonia,
        "is_ois": True,
        "dayCounter": ql.Actual365Fixed(),
        "calendar": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,  # Sonia swaps are often annual pay
        "period": ql.Period("1D"),
        "currency": ql.GBPCurrency(),
        "paymentLag": 0,  # SONIA OIS typically have no payment lag
        "settlementDays": 0,  # T+0 common for SONIA OIS
    },
    "EUR-ESTR-1D": {
        "swapIndex": ql.Estr,
        "is_ois": True,
        "dayCounter": ql.Actual360(),  # Correct
        "calendar": ql.TARGET(),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,  # ESTR swaps typically pay annually
        "period": ql.Period("1D"),
        "currency": ql.EURCurrency(),
        "paymentLag": 0,  # Usually 0 for ESTR
        "settlementDays": 2,  # T+2 is standard for EUR
    },
    "EUR-EURIBOR-1M": {
        "swapIndex": ql.Euribor1M,
        "is_ois": False,
        "dayCounter": ql.Actual360(),  # Changed to standard for EURIBOR swaps
        "calendar": ql.TARGET(),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,  # 1M Euribor usually referenced in annual fixed leg
        "period": ql.Period("1M"),
        "currency": ql.EURCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "EUR-EURIBOR-3M": {
        "swapIndex": ql.Euribor3M,
        "is_ois": False,
        "dayCounter": ql.Actual360(),  # Changed
        "calendar": ql.TARGET(),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Semiannual,  # Common for 3M Euribor fixed leg
        "period": ql.Period("3M"),
        "currency": ql.EURCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "EUR-EURIBOR-6M": {
        "swapIndex": ql.Euribor6M,
        "is_ois": False,
        "dayCounter": ql.Actual360(),  # Changed
        "calendar": ql.TARGET(),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,  # Fixed leg frequency for 6M Euribor is often annual
        "period": ql.Period("6M"),
        "currency": ql.EURCurrency(),
        "paymentLag": 2,
        "settlementDays": 2,
    },
    "JPY-TONAR-1D": {
        "swapIndex": ql.Tona,
        "is_ois": True,
        "dayCounter": ql.Actual365Fixed(),
        "calendar": ql.Japan(),
        "businessConvention": ql.ModifiedFollowing,
        "frequency": ql.Annual,  # TONAR OIS usually annual
        "period": ql.Period("1D"),
        "currency": ql.JPYCurrency(),
        "paymentLag": 0,  # TONAR OIS often has no lag
        "settlementDays": 0,  # T+0 settlement is common
    },
}


GOVIE_CURVE_QL_PARAMS = {
    "USD": {
        "calendar": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "settlementDays": 1,
        "frequency": ql.Semiannual,
        "dayCounter": ql.ActualActual(ql.ActualActual.Actual365),
        "businessConvention": ql.ModifiedFollowing,
        "faceAmount": 100,
        "OTR_TENORS": {
            ql.Period("1Y"): "CB12",
            ql.Period("2Y"): "CT2",
            ql.Period("3Y"): "CT3",
            ql.Period("5Y"): "CT5",
            ql.Period("7Y"): "CT7",
            ql.Period("10Y"): "CT10",
            ql.Period("20Y"): "CT20",
            ql.Period("30Y"): "CT30",
        },
    }
}
