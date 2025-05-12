from enum import Enum, auto
from functools import partial
from typing import Callable, List, Optional, Dict, Union, Literal
from datetime import datetime

import numpy as np
import QuantLib as ql
from scipy.optimize import newton

from core.TimeseriesBuilding.Base.BaseStructure import BaseStructureFunctionMap
from core.TimeseriesBuilding.Base.utils import build_irswaption


class IRSwaptionStructure(Enum):
    RECEIVER = auto()
    PAYER = auto()
    STRADDLE = auto()
    STRANGLE = auto()
    RECEIVER_SPREAD = auto()
    PAYER_SPREAD = auto()
    RECEIVER_FLY = auto()
    PAYER_FLY = auto()
    RECEIVER_1x2 = auto()
    PAYER_1x2 = auto()
    RECEIVER_LADDER = auto()
    PAYER_LADDER = auto()
    RISK_REVERSAL = auto()


class SwaptionStructureFunctionMap(BaseStructureFunctionMap[IRSwaptionStructure, ql.Swaption]):
    _long_risk_weighted_structure_map = {
        IRSwaptionStructure.RECEIVER: np.array([1]),
        IRSwaptionStructure.PAYER: np.array([1]),
        IRSwaptionStructure.STRADDLE: np.array([1, 1]),
        IRSwaptionStructure.STRANGLE: np.array([1, 1]),
        IRSwaptionStructure.RECEIVER_SPREAD: np.array([1, -1]),
        IRSwaptionStructure.PAYER_SPREAD: np.array([1, -1]),
        IRSwaptionStructure.RECEIVER_FLY: np.array([1, -2, 1]),
        IRSwaptionStructure.PAYER_FLY: np.array([1, -2, 1]),
        IRSwaptionStructure.RECEIVER_1x2: np.array([1, -2]),
        IRSwaptionStructure.PAYER_1x2: np.array([1, -2]),
        IRSwaptionStructure.RECEIVER_LADDER: np.array([1, -1, -1]),
        IRSwaptionStructure.PAYER_LADDER: np.array([1, -1, -1]),
        IRSwaptionStructure.RISK_REVERSAL: np.array([-1, 1]),  # sell rec, buy pay
    }

    def __init__(
        self,
        curve: str,
        curve_handle: ql.YieldTermStructureHandle,
        pricing_engine: ql.BachelierSwaptionEngine,
        swap_index: Optional[ql.SwapIndex] = None,
        side: Optional[Literal["buy", "sell"]] = "buy",
        notional: Optional[float] = 100_000_000.0,
    ):
        super().__init__(
            IRSwaptionStructure, curve=curve, curve_handle=curve_handle, pricing_engine=pricing_engine, swap_index=swap_index, side=side, notional=notional
        )
        self._rw_buy_or_side = 1 if str(side).lower().startswith("b") else -1
        self._map = self._create_map()

    def _create_map(self) -> Dict[IRSwaptionStructure, Callable[..., List[ql.Swaption]]]:
        return {
            IRSwaptionStructure.RECEIVER: partial(self._build_receiver),
            IRSwaptionStructure.PAYER: partial(self._build_payer),
            IRSwaptionStructure.STRADDLE: partial(self._build_straddle, strike="ATMF"),
            IRSwaptionStructure.STRANGLE: partial(self._build_strangle),
            IRSwaptionStructure.RECEIVER_SPREAD: partial(self._build_receiver_spread),
            IRSwaptionStructure.PAYER_SPREAD: partial(self._build_payer_spread),
            IRSwaptionStructure.RECEIVER_FLY: partial(self._build_receiver_fly),
            IRSwaptionStructure.PAYER_FLY: partial(self._build_payer_fly),
            IRSwaptionStructure.RECEIVER_1x2: partial(self._build_receiver_1x2),
            IRSwaptionStructure.PAYER_1x2: partial(self._build_payer_1x2),
            IRSwaptionStructure.RECEIVER_LADDER: partial(self._build_receiver_ladder),
            IRSwaptionStructure.PAYER_LADDER: partial(self._build_payer_ladder),
            IRSwaptionStructure.RISK_REVERSAL: partial(self._build_risk_reversal),
        }

    def _leg(
        self,
        kind: str,
        strike: float,
        notional: Optional[float] = None,
        vega: Optional[float] = None,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
    ) -> ql.Swaption:
        assert notional or vega, "MUST PASS IN `notional` or `vega`"
        sw = build_irswaption(
            curve=self.common_kwargs["curve"],
            curve_handle=self.common_kwargs["curve_handle"],
            pricing_engine=self.common_kwargs["pricing_engine"],
            swap_index=self.common_kwargs["swap_index"],
            strike=strike,
            notional=notional,
            vega=vega,
            r_p=kind,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
        )
        return sw

    def _find_costless_strike(
        self,
        long_swaption: ql.Swaption,
        short_swaption_builder: Callable[[float, float], ql.Swaption],
        short_notional: float,
        initial_guess: Optional[float] = None,
        tol: float = 1e-8,
        maxiter: int = 50,
    ) -> float:
        pv_long = long_swaption.NPV()
        guess = initial_guess or long_swaption.underlying().fairRate()

        def _f(K: float) -> float:
            short_sw = short_swaption_builder(K, short_notional)
            return pv_long - short_sw.NPV()

        return newton(func=_f, x0=guess, tol=tol, maxiter=maxiter)

    def _build_receiver(
        self,
        *,
        strike: float,
        notional: Optional[float] = None,
        vega: Optional[float] = None,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "r",
                strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.RECEIVER] * self._rw_buy_or_side

    def _build_payer(
        self,
        *,
        strike: float,
        notional: Optional[float] = None,
        vega: Optional[float] = None,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "p",
                strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.PAYER] * self._rw_buy_or_side

    def _build_straddle(
        self,
        *,
        strike: float,
        notional: Optional[float] = None,
        vega: Optional[float] = None,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        vega = vega / 2 if vega else None
        return [
            self._leg(
                "r",
                strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.STRADDLE] * self._rw_buy_or_side

    def _build_strangle(
        self,
        *,
        rec_strike: float,
        pay_strike: float,
        notional: Optional[float] = None,
        vega: Optional[float] = None,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "r",
                rec_strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                pay_strike,
                notional=notional,
                vega=vega,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.STRANGLE] * self._rw_buy_or_side

    def _build_receiver_spread(
        self,
        *,
        low_strike: float,
        high_strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "r",
                high_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "r",
                low_strike,
                -notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.RECEIVER_SPREAD] * self._rw_buy_or_side

    def _build_payer_spread(
        self,
        *,
        low_strike: float,
        high_strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "p",
                low_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                high_strike,
                -notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.PAYER_SPREAD] * self._rw_buy_or_side

    def _build_receiver_fly(
        self,
        *,
        low_strike: float,
        belly_strike: float,
        high_strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "r",
                high_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "r",
                belly_strike,
                -2 * notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "r",
                low_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.RECEIVER_FLY] * self._rw_buy_or_side

    def _build_payer_fly(
        self,
        *,
        low_strike: float,
        belly_strike: float,
        high_strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "p",
                low_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                belly_strike,
                -2 * notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                high_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.PAYER_SPREAD] * self._rw_buy_or_side

    def _build_receiver_1x2(
        self,
        *,
        strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        def _short(k, n):
            return self._leg(
                "r",
                k,
                n,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )

        long_sw = self._leg(
            "r",
            strike,
            notional,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
        )
        low = self._find_costless_strike(long_sw, _short, -2 * notional)
        return [long_sw, _short(low, -2 * notional)], self._long_risk_weighted_structure_map[IRSwaptionStructure.RECEIVER_1x2] * self._rw_buy_or_side

    def _build_payer_1x2(
        self,
        *,
        strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        def _short(k, n):
            return self._leg(
                "p",
                k,
                n,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )

        long_sw = self._leg(
            "p",
            strike,
            notional,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
        )
        high = self._find_costless_strike(long_sw, _short, -2 * notional)
        return [long_sw, _short(high, -2 * notional)], self._long_risk_weighted_structure_map[IRSwaptionStructure.PAYER_1x2] * self._rw_buy_or_side

    def _build_receiver_ladder(
        self,
        *,
        strike: float,
        notional: float,
        initial_bump: Optional[float] = None,
        tol: float = 1e-8,
        maxiter: int = 50,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        def _build_leg(K: float, n: float):
            return self._leg(
                "r",
                K * 100,
                n,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )

        long_sw = self._leg(
            "r",
            strike,
            notional,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
        )
        K0 = long_sw.underlying().fairRate()
        pv = long_sw.NPV()
        bump0 = initial_bump or 1e-3

        def f(b):
            return pv - (_build_leg(K0 - b, notional).NPV() + _build_leg(K0 - 2 * b, notional).NPV())

        bump = newton(func=f, x0=bump0, tol=tol, maxiter=maxiter)
        return [
            long_sw,
            _build_leg(K0 - bump, -notional),
            _build_leg(K0 - 2 * bump, -notional),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.RECEIVER_LADDER] * self._rw_buy_or_side

    def _build_payer_ladder(
        self,
        *,
        strike: float,
        notional: float,
        initial_bump: Optional[float] = None,
        tol: float = 1e-8,
        maxiter: int = 50,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        def _build_leg(K: float, n: float):
            return self._leg(
                "p",
                K * 100,
                n,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            )

        long_sw = self._leg(
            "p",
            strike,
            notional,
            expiry=expiry,
            tail=tail,
            exercise_date=exercise_date,
            underlying_effective_date=underlying_effective_date,
            underlying_maturity_date=underlying_maturity_date,
        )
        K0 = long_sw.underlying().fairRate()
        pv = long_sw.NPV()
        bump0 = initial_bump or 1e-3

        def f(b):
            return pv - (_build_leg(K0 + b, notional).NPV() + _build_leg(K0 + 2 * b, notional).NPV())

        bump = newton(func=f, x0=bump0, tol=tol, maxiter=maxiter)
        return [
            long_sw,
            _build_leg(K0 + bump, -notional),
            _build_leg(K0 + 2 * bump, -notional),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.PAYER_LADDER] * self._rw_buy_or_side

    def _build_risk_reversal(
        self,
        *,
        rec_strike: float,
        pay_strike: float,
        notional: float,
        expiry: Optional[Union[str, ql.Period]] = None,
        tail: Optional[Union[str, ql.Period]] = None,
        exercise_date: Optional[datetime] = None,
        underlying_effective_date: Optional[datetime] = None,
        underlying_maturity_date: Optional[datetime] = None,
        **_,
    ):
        return [
            self._leg(
                "r",
                rec_strike,
                -notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
            self._leg(
                "p",
                pay_strike,
                notional,
                expiry=expiry,
                tail=tail,
                exercise_date=exercise_date,
                underlying_effective_date=underlying_effective_date,
                underlying_maturity_date=underlying_maturity_date,
            ),
        ], self._long_risk_weighted_structure_map[IRSwaptionStructure.RISK_REVERSAL] * self._rw_buy_or_side
