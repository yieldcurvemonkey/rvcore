import math
from enum import Enum, auto
from typing import Any, Callable, Dict, List

import QuantLib as ql

from core.TimeseriesBuilding.Base.BaseValue import BaseValueFunctionMap
from core.TimeseriesBuilding.IRSwaps.pricer import calc_bpv, calc_fair_rate, calc_notional, calc_npv
from core.TimeseriesBuilding.IRSwaps.IRSwapStructure import IRSwapStructure


class IRSwapValue(Enum):
    RATE = auto()
    BPV = auto()
    NPV = auto()
    NOTIONAL = auto()
    CARRY = auto()
    ROLL = auto()
    CARRY_AND_ROLL = auto()
    DOLLAR_CARRY_AND_ROLL = auto()


class IRSwapValueFunctionMap(BaseValueFunctionMap[IRSwapValue, float]):
    def __init__(
        self,
        package: List[ql.Swap],
        risk_weights: List[float],
        curve: str,
        curve_handle: ql.YieldTermStructureHandle,
    ):
        super().__init__(IRSwapValue, package=package, risk_weights=risk_weights, curve=curve, curve_handle=curve_handle)

    def _create_map(self) -> Dict[IRSwapValue, Callable[..., float]]:
        return {
            IRSwapValue.RATE: self._rate,
            IRSwapValue.NPV: self._npv,
            IRSwapValue.BPV: self._bpv,
            IRSwapValue.NOTIONAL: self._notional,
        }

    def _rate(self, **kwargs: Any) -> float:
        return calc_spread_rate(kwargs["package"], kwargs["curve_handle"], kwargs["risk_weights"]) * _swap_structure_legs_mapper[len(kwargs["package"])][1]

    def _npv(self, **kwargs: Any) -> float:
        return sum(calc_npv(s, kwargs["curve_handle"]) for s in kwargs["package"])

    def _bpv(self, **kwargs: Any) -> float:
        return sum(calc_bpv(s, kwargs["curve_handle"]) for s in kwargs["package"])

    def _notional(self, **kwargs: Any) -> float:
        return sum(math.copysign(calc_notional(s, kwargs["curve_handle"]), s.fixedLegBPS()) for s in kwargs["package"])


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
