import collections
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union, Literal

import QuantLib as ql

from core.TimeseriesBuilding.Base.BaseQuery import BaseQuery
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionStructure import IRSwaptionStructure
from core.TimeseriesBuilding.IRSwaptions.IRSwaptionValue import IRSwaptionValue
from core.TimeseriesBuilding.IRSwaps.IRSwapStructure import IRSwapStructureFunctionMap, IRSwapStructure

from core.utils.ql_utils import ql_date_to_datetime

STRUCTURE_KWARGS_FORMATTERS: Dict[IRSwaptionStructure, Callable[[Optional[Dict]], str]] = {
    IRSwaptionStructure.RECEIVER: lambda kw: f"{kw['strike']}" if kw else "",
    IRSwaptionStructure.PAYER: lambda kw: f"{kw['strike']}" if kw else "",
    IRSwaptionStructure.STRADDLE: lambda kw: f"{kw['strike']}" if kw else "",
    IRSwaptionStructure.STRANGLE: lambda kw: f"{kw['strike']}" if kw else "",
    IRSwaptionStructure.RECEIVER_SPREAD: lambda kw: f"{kw['low_strike']}, {kw['high_strike']}" if kw else "",
    IRSwaptionStructure.PAYER_SPREAD: lambda kw: f"{kw['high_strike']}, {kw['low_strike']}" if kw else "",
    IRSwaptionStructure.RECEIVER_FLY: lambda kw: f"{kw['low_strike']}, {kw['belly_strike']}, {kw['high_strike']}" if kw else "",
    IRSwaptionStructure.PAYER_FLY: lambda kw: f"{kw['high_strike']}, {kw['belly_strike']}, {kw['low_strike']}" if kw else "",
    IRSwaptionStructure.RECEIVER_1x2: lambda kw: f"{kw['low_strike']}, {kw['high_strike']}" if kw else "",
    IRSwaptionStructure.PAYER_1x2: lambda kw: f"{kw['high_strike']}, {kw['low_strike']}" if kw else "",
    IRSwaptionStructure.RECEIVER_LADDER: lambda kw: f"{kw['strike']} Costless" if kw else "",
    IRSwaptionStructure.PAYER_LADDER: lambda kw: f"{kw['strike']} Costless" if kw else "",
    IRSwaptionStructure.RISK_REVERSAL: lambda kw: f"{kw['rec_strike']}, {kw['pay_strike']}" if kw else "",
}


@dataclass
class IRSwaptionQuery(BaseQuery):
    expiry: Optional[Union[str, ql.Period]] = None
    tail: Optional[Union[str, ql.Period]] = None
    exercise_date: Optional[datetime] = None
    underlying_effective_date: Optional[datetime] = None
    underlying_maturity_date: Optional[datetime] = None

    trade_date: Optional[datetime] = None  # convert CMT query into 'actual' trade i.e. converts query to dates and fixed strike
    curve: Optional[str] = None
    curve_handle: Optional[datetime] = None

    value: IRSwaptionValue | List[IRSwaptionValue] = IRSwaptionValue.NVOL
    structure: IRSwaptionStructure = IRSwaptionStructure.STRADDLE
    structure_kwargs: Optional[Dict] = field(default_factory=dict)

    name: Optional[str] = None
    side: Optional[Literal["buy", "b", "sell", "s"]] = "buy"
    risk_weight: Optional[float | Literal["buy", "b", "sell", "s"]] = 1

    def __post_init__(self):
        assert (self.expiry and self.tail) or (
            self.exercise_date and self.underlying_effective_date and self.underlying_maturity_date
        ), "MUST PASS IN expiry-tail or datetimes"

        if isinstance(self.value, collections.abc.Iterable) and (
            any(self.value) == IRSwaptionValue.DAILY_BREAKEVEN_NVOL or any(self.value) == IRSwaptionValue.ANNUAL_BREAKEVEN_NVOL
        ):
            assert self.structure == IRSwaptionStructure.STRADDLE, "BREAKEVEN_NVOL only implemented for straddles"
        elif self.value == IRSwaptionValue.DAILY_BREAKEVEN_NVOL or self.value == IRSwaptionValue.ANNUAL_BREAKEVEN_NVOL:
            assert self.structure == IRSwaptionStructure.STRADDLE, "BREAKEVEN_NVOL only implemented for straddles"

        if self.side.lower().startswith("b"):
            self.risk_weight = 1
        elif self.side.lower().startswith("s"):
            self.risk_weight = -1
        else:
            raise ValueError(f"Invalid risk weight: {self.risk_weight}")

        # TODO generalize/support more structures - will need to pass in vol handle
        if self.trade_date:
            assert self.expiry and self.tail, "'expiry' and 'tail' expected for 'trade_date'"
            assert self.curve and self.curve_handle, "'curve' and 'curve_handle' expected for 'trade_date'"
            assert self.structure in [
                IRSwaptionStructure.STRADDLE,
                IRSwaptionStructure.RECEIVER,
                IRSwaptionStructure.PAYER,
            ], "'structure' must be a 'STRADDLE', 'RECEIVER', or 'PAYER'"
            assert "x" not in self.tail, "midcurves are not supported for 'trade_date'"
            assert ql_date_to_datetime(self.curve_handle.referenceDate()).date() == self.trade_date.date(), "'curve_handle' ref date must match 'trade_date'"
            ql.Settings.instance().evaluationDate = self.curve_handle.referenceDate()

            tenor = f"{self.expiry}x{self.tail}"
            pkg, _ = IRSwapStructureFunctionMap(curve=self.curve, curve_handle=self.curve_handle).apply(tenor=tenor, structure=IRSwapStructure.OUTRIGHT)
            pkg: ql.FixedVsFloatingSwap = pkg[0]

            self.expiry = None
            self.tail = None
            self.trade_date = None
            self.curve = None
            self.curve_handle = None

            self.exercise_date = ql_date_to_datetime(pkg.startDate())
            self.underlying_effective_date = ql_date_to_datetime(pkg.startDate())
            self.underlying_maturity_date = ql_date_to_datetime(pkg.maturityDate())
            self.structure_kwargs = self.structure_kwargs | {"strike": pkg.fairRate() * 100}

    def return_query(self):
        if isinstance(self.value, List):
            to_ret = []
            for sw_val in self.value:
                to_ret.append(
                    IRSwaptionQuery(
                        expiry=self.expiry,
                        tail=self.tail,
                        exercise_date=self.exercise_date,
                        underlying_effective_date=self.underlying_effective_date,
                        underlying_maturity_date=self.underlying_maturity_date,
                        value=sw_val,
                        structure=self.structure,
                        structure_kwargs=self.structure_kwargs,
                        name=self.name,
                        risk_weight=self.risk_weight,
                    )
                )
            return to_ret

        return [self]

    def col_name(self, cube_name: Optional[str] = None):
        swaption_name = (
            f"{self.expiry}x{self.tail}"
            if self.expiry
            else f"Exercise: {self.exercise_date.date()}, Start: {self.underlying_effective_date.date()}, End: {self.underlying_maturity_date.date()}"
        )
        if self.name is None:
            return f"{cube_name} {swaption_name} {STRUCTURE_KWARGS_FORMATTERS[self.structure](self.structure_kwargs)} {self.structure.name} {self.value.name}"
        return self.name

    def eval_expression(self, cube_name: Optional[str] = None, ignore_risk_weight: Optional[bool] = False):
        if self.risk_weight != None and not ignore_risk_weight:
            return f"{self.risk_weight} * `{self.col_name(cube_name=cube_name)}`"
        return f"`{self.col_name(cube_name=cube_name)}`"

    def __pos__(self) -> "IRSwaptionQuery":
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        return self

    def __neg__(self) -> "IRSwaptionQuery":
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        new_weight = -(self.risk_weight or 1)
        return replace(self, risk_weight=new_weight)

    def __add__(self, other: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        if not isinstance(other, IRSwaptionQuery):
            return NotImplemented
        return [self * 1, other * 1]

    def __radd__(self, other: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        return self.__add__(other)

    def __sub__(self, other: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        if not isinstance(other, IRSwaptionQuery):
            return NotImplemented
        return [self * 1, other * -1]

    def __rsub__(self, other: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        if not isinstance(other, IRSwaptionQuery):
            return NotImplemented
        return [other * 1, self * -1]

    def __mul__(self, scalar: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        new_weight = (self.risk_weight or 1) * scalar
        return replace(self, risk_weight=new_weight)

    def __rmul__(self, scalar: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return self * (1.0 / scalar)

    def __rtruediv__(self, scalar: object) -> List["IRSwaptionQuery"]:
        # assert isinstance(self.value, IRSwaptionValue), "Must have single IRSwaptionValue"
        return NotImplemented


@dataclass
class IRSwaptionQueryWrapper:
    queries: List[IRSwaptionQuery]
    name: str
    ignore_risk_weights: Optional[bool] = True

    def __init__(
        self,
        queries: Union[IRSwaptionQuery, List[IRSwaptionQuery]],
        name: str,
    ):
        self.queries = [queries] if isinstance(queries, IRSwaptionQuery) else queries
        self.name = name

    def return_query(self) -> List[IRSwaptionQuery]:
        return self.queries

    def col_name(self, cube_name: Optional[str] = None) -> str:
        # use the wrapper’s own name
        return self.name

    def eval_expression(self, cube_name: Optional[str] = None) -> str:
        exp = ""
        for q in self.queries:
            rw = q.risk_weight
            exp += f"{rw} * `{q.col_name(cube_name=cube_name)}` + "
        return exp[:-3]
