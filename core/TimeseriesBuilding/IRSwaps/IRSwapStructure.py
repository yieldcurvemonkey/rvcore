import math
from datetime import datetime
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from core.utils.ql_loader import ql
from rateslib.calendars import get_imm

from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.TimeseriesBuilding.Base.BaseStructure import BaseStructureFunctionMap
from core.TimeseriesBuilding.Base.utils import build_irswap
from core.utils.ql_utils import datetime_to_ql_date, ql_date_to_datetime


def _linear_solve_for_risk_weighted_notionals(
    risk_weights: np.ndarray,
    bpvs: np.ndarray,
    constrained_leg_index: int,
    constrained_leg_contribution: float,
    contribution_is_bpv: Optional[bool] = False,
) -> np.ndarray:
    idx = constrained_leg_index
    constrained_leg_contribution = math.copysign(constrained_leg_contribution, risk_weights[idx])
    if contribution_is_bpv:
        notional_contrib = constrained_leg_contribution / bpvs[idx]
    else:
        notional_contrib = constrained_leg_contribution

    R = notional_contrib * bpvs[idx] / risk_weights[idx]
    notionals = (risk_weights * R) / bpvs
    return notionals


class IRSwapStructure(Enum):
    OUTRIGHT = auto()
    CURVE = auto()
    FLY = auto()


class IRSwapStructureFunctionMap(BaseStructureFunctionMap[IRSwapStructure, ql.VanillaSwap]):
    def __init__(
        self,
        curve: str,
        curve_handle: ql.YieldTermStructureHandle,
        swap_index: Optional[ql.SwapIndex] = None,
    ):
        super().__init__(
            IRSwapStructure,
            curve=curve,
            curve_handle=curve_handle,
            swap_index=swap_index,
        )
        self._map = self._create_map()
        self._cal = CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"]
        self._curve_ref_date = curve_handle.referenceDate()

    def _create_map(self) -> Dict[IRSwapStructure, Callable[..., List[ql.VanillaSwap]]]:
        return {
            IRSwapStructure.OUTRIGHT: partial(self._build_outright),
            IRSwapStructure.CURVE: partial(self._build_curve),
            IRSwapStructure.FLY: partial(self._build_fly),
        }

    def _to_dt(self, d: datetime | ql.Period | str, ref_date: datetime | ql.Date):
        if isinstance(d, ql.Period):
            d = ql_date_to_datetime(self._cal.advance(datetime_to_ql_date(ref_date), d))
        if isinstance(d, ql.Date):
            return ql_date_to_datetime(d)
        if isinstance(d, str) and d.upper().startswith("IMM_"):
            return get_imm(code=d.split("IMM_")[-1])
        if isinstance(d, str):
            d = ql_date_to_datetime(self._cal.advance(datetime_to_ql_date(ref_date), ql.Period(d)))
        return d

    def _leg(
        self,
        tenor: Optional[str] = None,
        effective_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        fixed_rate: Optional[float] = -0.00,
        notional: Optional[float] = 100_000_000,
        bpv: Optional[float] = None,
    ) -> ql.VanillaSwap:
        if isinstance(tenor, str) and tenor.startswith("IMM_"):
            imm_date, mat_date = tenor.split("x")
            effective_date = self._to_dt(imm_date, ql_date_to_datetime(self._curve_ref_date))
            maturity_date = self._to_dt(mat_date, effective_date)
            fwd, tenor = None, None
        else:
            if tenor and "x" in tenor:
                fwd, tenor = tenor.split("x")
                fwd, tenor = ql.Period(fwd), ql.Period(tenor)
            elif tenor:
                fwd, tenor = ql.Period("0D"), ql.Period(tenor) if tenor else None
            elif effective_date and maturity_date:
                fwd, tenor = None, None
            else:
                raise ValueError("Need to define tenor or dates")

        sw = build_irswap(
            curve=self.common_kwargs["curve"],
            curve_handle=self.common_kwargs["curve_handle"],
            swap_index=self.common_kwargs["swap_index"],
            notional=notional,
            bpv=bpv,
            fwd=fwd,
            tenor=tenor,
            effective_date=effective_date,
            maturity_date=maturity_date,
            fixed_rate=fixed_rate,
        )
        return sw

    def _build_spreadable(
        self,
        leg_specs: List[Dict[str, Any]],
        risk_weights: Optional[List[float]] = None,
        constrained_leg_index: Optional[int] = None,
        constrained_notional: Optional[float] = None,
        constrained_bpv: Optional[float] = None,
    ) -> List[ql.VanillaSwap]:
        n = len(leg_specs)
        rw = np.array(risk_weights or ([1.0, -1.0] + [0.0] * (n - 2))[:n], dtype=float)

        if sum(x is not None for x in (constrained_notional, constrained_bpv)) != 1 or constrained_leg_index is None:
            raise ValueError("Must specify constrained_leg_index and exactly one of constrained_notional/bpv")

        unit_swaps = [self._leg(**spec, notional=1.0) for spec in leg_specs]
        bpvs = np.array([s.fixedLegBPS() for s in unit_swaps], dtype=float)

        contribution_is_bpv = False
        if constrained_notional is not None:
            contribution = constrained_notional
        else:
            contribution = constrained_bpv
            contribution_is_bpv = True

        notionals = _linear_solve_for_risk_weighted_notionals(
            risk_weights=rw,
            bpvs=bpvs,
            constrained_leg_index=constrained_leg_index,
            constrained_leg_contribution=contribution,
            contribution_is_bpv=contribution_is_bpv,
        )
        result = []
        for spec, n in zip(leg_specs, notionals):
            result.append(self._leg(**spec, notional=float(n)))
        return result

    def _build_outright(
        self,
        *,
        tenor: Optional[str] = None,
        effective_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        fixed_rate: Optional[float] = -0.00,
        notional: Optional[float] = 100_000_000,
        bpv: Optional[float] = None,
        **_,
    ) -> Tuple[List[ql.VanillaSwap], List[float]]:
        return (
            [
                self._leg(
                    tenor=tenor,
                    effective_date=effective_date,
                    maturity_date=maturity_date,
                    fixed_rate=fixed_rate,
                    notional=notional,
                    bpv=bpv,
                )
            ],
            [1 if notional > 0 or (bpv and bpv > 0) else -1],
        )

    def _build_curve(
        self,
        *,
        front_tenor: Optional[str] = None,
        front_effective_date: Optional[datetime] = None,
        front_maturity_date: Optional[datetime] = None,
        back_tenor: Optional[str] = None,
        back_effective_date: Optional[datetime] = None,
        back_maturity_date: Optional[datetime] = None,
        front_notional: Optional[float] = None,
        back_notional: Optional[float] = None,
        bpv: Optional[float] = None,
        risk_weights: Optional[List[float]] = [1, 1],
        front_fixed_rate: Optional[float] = -0,
        back_fixed_rate: Optional[float] = -0,
        **_,
    ) -> Tuple[List[ql.VanillaSwap], List[float]]:
        assert sum(x is not None for x in (front_notional, back_notional, bpv)) == 1, "Exactly one of front_notional, back_notional or bpv must be provided"
        assert len(risk_weights) == 2, "CURVE 2 RISK WEIGHTS"

        # `_leg` specs
        leg0 = dict(tenor=front_tenor, effective_date=front_effective_date, maturity_date=front_maturity_date, fixed_rate=front_fixed_rate)
        leg1 = dict(tenor=back_tenor, effective_date=back_effective_date, maturity_date=back_maturity_date, fixed_rate=back_fixed_rate)

        # determine constraint
        if front_notional is not None:
            idx, cn, cp = 0, front_notional, None
            risk_weights[0] = math.copysign(risk_weights[0], cn)
            risk_weights[1] = math.copysign(risk_weights[1], risk_weights[0] * -1)
        elif back_notional is not None:
            idx, cn, cp = 1, back_notional, None
            risk_weights[1] = math.copysign(risk_weights[1], cn)
            risk_weights[0] = math.copysign(risk_weights[0], risk_weights[1] * -1)
        else:
            idx, cn, cp = 1, None, bpv  # relative to the back leg e.g. rec 2s10s => rec 10s, pay2s flattener
            risk_weights[1] = math.copysign(risk_weights[1], cp)
            risk_weights[0] = math.copysign(risk_weights[0], risk_weights[1] * -1)

        return (
            self._build_spreadable(
                [leg0, leg1],
                risk_weights=risk_weights,
                constrained_leg_index=idx,
                constrained_notional=cn,
                constrained_bpv=cp,
            ),
            risk_weights,
        )

    def _build_fly(
        self,
        *,
        front_tenor: Optional[str] = None,
        front_effective_date: Optional[datetime] = None,
        front_maturity_date: Optional[datetime] = None,
        belly_tenor: Optional[str] = None,
        belly_effective_date: Optional[datetime] = None,
        belly_maturity_date: Optional[datetime] = None,
        back_tenor: Optional[str] = None,
        back_effective_date: Optional[datetime] = None,
        back_maturity_date: Optional[datetime] = None,
        front_fixed_rate: Optional[float] = -0.0,
        mid_fixed_rate: Optional[float] = -0.0,
        back_fixed_rate: Optional[float] = -0.0,
        front_notional: Optional[float] = None,
        belly_notional: Optional[float] = None,
        back_notional: Optional[float] = None,
        bpv: Optional[float] = None,
        risk_weights: Optional[List[float]] = [1.0, 2.0, 1.0],
        **_,
    ) -> Tuple[List[ql.VanillaSwap], List[float]]:
        assert (
            sum(x is not None for x in (front_notional, belly_notional, back_notional, bpv)) == 1
        ), "Exactly one of front_notional, belly_notional, back_notional or bpv must be provided"
        assert len(risk_weights) == 3, "FLY NEEDS 3 RISK WEIGHTS"

        if front_notional is not None:
            idx, cn, cp = 0, front_notional, None
        elif belly_notional is not None:
            idx, cn, cp = 1, belly_notional, None
        elif back_notional is not None:
            idx, cn, cp = 2, back_notional, None
        else:
            idx, cn, cp = 1, None, bpv

        risk_weights[idx] = math.copysign(risk_weights[idx], cn if cn is not None else cp)
        for i in range(3):
            if i != idx:
                risk_weights[i] = math.copysign(risk_weights[i], -risk_weights[idx])

        leg0 = dict(
            tenor=front_tenor,
            effective_date=front_effective_date,
            maturity_date=front_maturity_date,
            fixed_rate=front_fixed_rate,
        )
        leg1 = dict(
            tenor=belly_tenor,
            effective_date=belly_effective_date,
            maturity_date=belly_maturity_date,
            fixed_rate=mid_fixed_rate,
        )
        leg2 = dict(
            tenor=back_tenor,
            effective_date=back_effective_date,
            maturity_date=back_maturity_date,
            fixed_rate=back_fixed_rate,
        )

        return (
            self._build_spreadable(
                leg_specs=[leg0, leg1, leg2],
                risk_weights=risk_weights,
                constrained_leg_index=idx,
                constrained_notional=cn,
                constrained_bpv=cp,
            ),
            risk_weights,
        )
