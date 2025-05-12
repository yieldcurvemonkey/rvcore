import QuantLib as ql

from enum import Enum, auto
from typing import List, Any, Dict, Callable

from core.TimeseriesBuilding.Base.BaseValue import BaseValueFunctionMap
from core.TimeseriesBuilding.IRSwaptions.pricer import (
    calc_nvol,
    calc_portfolio_vega_weighted_nvol,
    calc_straddle_daily_breakeven_nvol,
    calc_straddle_annual_breakeven_nvol,
    calc_book_spot_npv,
    calc_book_fwd_npv,
    calc_book_spot_prem,
    calc_book_fwd_prem,
    calc_book_dv01,
    calc_book_delta,
    calc_book_vega,
    calc_book_gamma,
    calc_book_dollar_gamma,
    calc_book_theta,
    calc_book_charm,
    calc_book_veta,
)
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionStructure import IRSwaptionStructure


class IRSwaptionValue(Enum):
    NVOL = auto()
    SPREAD_NVOL = auto()
    DAILY_BREAKEVEN_NVOL = auto()
    ANNUAL_BREAKEVEN_NVOL = auto()
    # FWD_NVOL = auto()
    # IMPLIED_CORRELATION = auto()
    SPOT_NPV = auto()
    FWD_NPV = auto()
    SPOT_PREM = auto()
    FWD_PREM = auto()
    DV01 = auto()
    DELTA = auto()
    GAMMA = auto()
    GAMMA_01 = auto()
    VEGA_01 = auto()
    THETA_1D = auto()
    CHARM = auto()
    VETA = auto()
    # RETURNS = auto()


class SwaptionValueFunctionMap(BaseValueFunctionMap[IRSwaptionValue, float]):
    def __init__(
        self,
        package: List[ql.Swaption],
        risk_weights: List[float],
        curve: str,
        curve_handle: ql.YieldTermStructureHandle,
        pricing_engine: ql.BachelierSwaptionEngine,
        swaption_structure: IRSwaptionStructure,
        # bps_bump_size: int = 1,
        # horizon: ql.Period = ql.Period("1D"),
    ):
        super().__init__(
            IRSwaptionValue,
            package=package,
            risk_weights=risk_weights,
            curve=curve,
            curve_handle=curve_handle,
            pricing_engine=pricing_engine,
            swaption_structure=swaption_structure,
            # bps_bump_size=bps_bump_size,
            # horizon=horizon,
        )

    def _create_map(self) -> Dict[IRSwaptionValue, Callable[..., float]]:
        return {
            IRSwaptionValue.NVOL: self._nvol,
            IRSwaptionValue.SPREAD_NVOL: self._spread_nvol,
            IRSwaptionValue.DAILY_BREAKEVEN_NVOL: self._daily_breakeven,
            IRSwaptionValue.ANNUAL_BREAKEVEN_NVOL: self._annual_breakeven,
            IRSwaptionValue.SPOT_NPV: self._spot_npv,
            IRSwaptionValue.FWD_NPV: self._fwd_npv,
            IRSwaptionValue.SPOT_PREM: self._spot_prem,
            IRSwaptionValue.FWD_PREM: self._fwd_prem,
            IRSwaptionValue.DV01: self._dv01,
            IRSwaptionValue.DELTA: self._delta,
            IRSwaptionValue.GAMMA: self._gamma,
            IRSwaptionValue.GAMMA_01: self._dollar_gamma,
            IRSwaptionValue.VEGA_01: self._vega,
            IRSwaptionValue.THETA_1D: self._theta,
            IRSwaptionValue.CHARM: self._charm,
            IRSwaptionValue.VETA: self._veta,
        }

    def _nvol(self, **kwargs: Any) -> float:
        return calc_portfolio_vega_weighted_nvol(kwargs["package"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _spread_nvol(self, **kwargs: Any) -> float:
        return calc_spread_nvol(kwargs["package"], kwargs["curve_handle"], kwargs["pricing_engine"], kwargs["swaption_structure"])

    def _daily_breakeven(self, **kwargs: Any) -> float:
        return calc_straddle_daily_breakeven_nvol(kwargs["package"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _annual_breakeven(self, **kwargs: Any) -> float:
        return calc_straddle_annual_breakeven_nvol(kwargs["package"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _spot_npv(self, **kwargs: Any) -> float:
        return calc_book_spot_npv(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _fwd_npv(self, **kwargs: Any) -> float:
        return calc_book_fwd_npv(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _spot_prem(self, **kwargs: Any) -> float:
        return calc_book_spot_prem(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _fwd_prem(self, **kwargs: Any) -> float:
        return calc_book_fwd_prem(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _dv01(self, **kwargs: Any) -> float:
        return calc_book_dv01(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _delta(self, **kwargs: Any) -> float:
        return calc_book_delta(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _vega(self, **kwargs: Any) -> float:
        return calc_book_vega(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"])

    def _gamma(self, **kwargs: Any) -> float:
        return calc_book_gamma(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"], curve=kwargs["curve"], bump_bps=1)

    def _dollar_gamma(self, **kwargs: Any) -> float:
        return calc_book_dollar_gamma(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"], curve=kwargs["curve"], bump_bps=1)

    def _theta(self, **kwargs: Any) -> float:
        return calc_book_theta(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"], horizon=ql.Period("1D"))

    def _charm(self, **kwargs: Any) -> float:
        return calc_book_charm(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"], horizon=ql.Period("1D"))

    def _veta(self, **kwargs: Any) -> float:
        return calc_book_veta(kwargs["package"], kwargs["risk_weights"], kwargs["curve_handle"], kwargs["pricing_engine"], horizon=ql.Period("1D"))


def calc_spread_nvol(
    book: List[ql.Swaption],
    curve_handle: ql.YieldTermStructureHandle,
    engine: ql.BachelierSwaptionEngine,
    structure: IRSwaptionStructure,
) -> float:
    ql.Settings.instance().evaluationDate = curve_handle.referenceDate()
    n = len(book)
    if n == 0:
        raise ValueError("Empty swaption book")

    vols = [calc_nvol(sw, curve_handle, engine) for sw in book]
    distances = [abs(sw.underlying().fixedRate() - sw.underlying().fairRate()) for sw in book]

    if structure is IRSwaptionStructure.RECEIVER:
        return float(vols[0])
    elif structure is IRSwaptionStructure.PAYER:
        return float(vols[0])
    elif structure is IRSwaptionStructure.STRADDLE:
        return float(0.5 * (vols[0] + vols[1]))
    elif structure is IRSwaptionStructure.STRANGLE:
        raise NotImplementedError("STRANGLE logic goes here")

    elif structure is IRSwaptionStructure.RECEIVER_SPREAD:
        otm = int(distances.index(max(distances)))
        near = 1 - otm
        return float(vols[otm] - vols[near])

    elif structure is IRSwaptionStructure.PAYER_SPREAD:
        otm = int(distances.index(max(distances)))
        near = 1 - otm
        return float(vols[otm] - vols[near])

    elif structure is IRSwaptionStructure.RECEIVER_FLY:
        belly = int(distances.index(min(distances)))
        wings = [i for i in range(3) if i != belly]
        return float(2 * vols[belly] - vols[wings[0]] - vols[wings[1]])

    elif structure is IRSwaptionStructure.PAYER_FLY:
        belly = int(distances.index(min(distances)))
        wings = [i for i in range(3) if i != belly]
        return float(2 * vols[belly] - vols[wings[0]] - vols[wings[1]])

    elif structure is IRSwaptionStructure.RECEIVER_1x2:
        long_vol = vols[0]
        short_vol = vols[1]
        return float(long_vol - 2 * short_vol)

    elif structure is IRSwaptionStructure.PAYER_1x2:
        long_vol = vols[0]
        short_vol = vols[1]
        return float(long_vol - 2 * short_vol)

    elif structure is IRSwaptionStructure.RECEIVER_LADDER:
        raise NotImplementedError("RECEIVER_LADDER logic goes here")

    elif structure is IRSwaptionStructure.PAYER_LADDER:
        raise NotImplementedError("PAYER_LADDER logic goes here")

    elif structure is IRSwaptionStructure.RISK_REVERSAL:
        return float(vols[1] - vols[0])

    else:
        raise ValueError(f"Unsupported structure: {structure}")
