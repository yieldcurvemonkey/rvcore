import math
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import QuantLib as ql

from core.TimeseriesBuilding.Base.BaseValue import BaseValueFunctionMap
from core.TimeseriesBuilding.IRSwaps.pricer import (
    calc_fair_rate,
    calc_pv01,
    calc_dv01,
    calc_gamma,
    calc_notional,
    calc_npv,
    calc_carry_bps_running,
    calc_roll_bps_running,
    calc_carry_and_roll_bps_running,
)
from core.TimeseriesBuilding.IRSwaps.IRSwapStructure import IRSwapStructure


class IRSwapValue(Enum):
    RATE = auto()
    PV01 = auto()
    DV01 = auto()
    GAMMA_01 = auto()
    NPV = auto()
    NOTIONAL = auto()
    CARRY_BPS_RUNNING = auto()
    ROLL_BPS_RUNNING = auto()
    CARRY_AND_ROLL_BPS_RUNNING = auto()


class IRSwapValueFunctionMap(BaseValueFunctionMap[IRSwapValue, float]):
    def __init__(
        self,
        package: List[ql.Swap],
        risk_weights: List[float],
        curve: str,
        curve_handle: ql.YieldTermStructureHandle,
        swap_index: Optional[ql.SwapIndex] = None,
    ):
        super().__init__(IRSwapValue, package=package, risk_weights=risk_weights, curve=curve, curve_handle=curve_handle, swap_index=swap_index)

    def _create_map(self) -> Dict[IRSwapValue, Callable[..., float]]:
        return {
            IRSwapValue.RATE: self._rate,
            IRSwapValue.PV01: self._pv01,
            IRSwapValue.DV01: self._dv01,
            IRSwapValue.GAMMA_01: self._gamma,
            IRSwapValue.NPV: self._npv,
            IRSwapValue.NOTIONAL: self._notional,
            IRSwapValue.CARRY_BPS_RUNNING: self._carry_bps_running,
            IRSwapValue.ROLL_BPS_RUNNING: self._rolldown_bps_running,
            IRSwapValue.CARRY_AND_ROLL_BPS_RUNNING: self._carry_and_roll_bps_running,
        }

    def _rate(self, **kwargs: Any) -> float:
        return calc_spread_rate(kwargs["package"], kwargs["curve_handle"], kwargs["risk_weights"]) * _swap_structure_legs_mapper[len(kwargs["package"])][1]

    def _npv(self, **kwargs: Any) -> float:
        return sum(calc_npv(s, kwargs["curve_handle"]) for s in kwargs["package"])

    def _pv01(self, **kwargs: Any) -> float:
        return sum(calc_pv01(s, kwargs["curve_handle"]) for s in kwargs["package"])

    def _dv01(self, **kwargs: Any) -> float:
        return sum(calc_dv01(s, kwargs["curve_handle"], kwargs["curve"]) for s in kwargs["package"])

    def _gamma(self, **kwargs: Any) -> float:
        return sum(calc_gamma(s, kwargs["curve_handle"], kwargs["curve"]) for s in kwargs["package"])

    def _notional(self, **kwargs: Any) -> float:
        return sum(math.copysign(calc_notional(s, kwargs["curve_handle"]), s.fixedLegBPS()) for s in kwargs["package"])

    def _carry_bps_running(self, **kwargs: Any) -> float:
        assert "horizon" in kwargs, 'Expecting an "horizon" ql.Period in args e.g. `ql.Period("1M")`'
        return sum(calc_carry_bps_running(s, kwargs["curve_handle"], kwargs["curve"], kwargs["horizon"]) for s in kwargs["package"])

    def _rolldown_bps_running(self, **kwargs: Any) -> float:
        assert "horizon" in kwargs, 'Expecting an "horizon" ql.Period in args e.g. `ql.Period("1M")`'
        return sum(calc_roll_bps_running(s, kwargs["curve_handle"], kwargs["curve"], kwargs["horizon"]) for s in kwargs["package"])

    def _carry_and_roll_bps_running(self, **kwargs: Any) -> float:
        assert "horizon" in kwargs, 'Expecting an "horizon" ql.Period in args e.g. `ql.Period("1M")`'
        return sum(calc_carry_and_roll_bps_running(s, kwargs["curve_handle"], kwargs["curve"], kwargs["horizon"]) for s in kwargs["package"])


_swap_structure_sign_mapper = {
    IRSwapStructure.OUTRIGHT: lambda rws: [abs(rws[0])],
    IRSwapStructure.CURVE: lambda rws: [-1 * abs(rws[0]), abs(rws[1])],
    IRSwapStructure.FLY: lambda rws: [-1 * abs(rws[0]), abs(rws[1]), -1 * abs(rws[2])],
}

_swap_structure_legs_mapper = {
    1: (IRSwapStructure.OUTRIGHT, 100),
    2: (IRSwapStructure.CURVE, 10_000),
    3: (IRSwapStructure.FLY, 10_000),
}


def calc_spread_rate(
    package: List[ql.VanillaSwap],
    curve_handle: ql.YieldTermStructureHandle,
    risk_weights: List[float],
) -> float:

    risk_weights = _swap_structure_sign_mapper[_swap_structure_legs_mapper[len(package)][0]](risk_weights)
    return sum([risk_weights[i] * abs(calc_fair_rate(sw, curve_handle)) for i, sw in enumerate(package)])
