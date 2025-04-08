from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import QuantLib as ql
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.Products.CurveBuilding.ql_curve_params import CME_SWAP_CURVE_QL_PARAMS
from core.utils.ql_utils import get_bdates_between
from core.Products.CurveBuilding.Swaptions.ql_cube_building_utils import build_ql_interpolated_vol_cube, build_ql_sabr_vol_cube
from core.Products.CurveBuilding.Swaptions.types import SCube, SABRParams


class SwaptionCubeBuilder:
    _scube_timeseries: Dict[datetime, SCube] = None
    _sabr_params_timeseries: Dict[datetime, SABRParams] = None
    _ql_cube_cache: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube] = None

    def __init__(
        self,
        swaption_cube_timeseries_dict: Dict[datetime, SCube],
        sabr_params_timeseries_dict: Optional[Dict[datetime, SABRParams]] = None,
    ):
        self._ql_cube_cache = {}
        self._scube_timeseries = swaption_cube_timeseries_dict
        self._sabr_params_timeseries = sabr_params_timeseries_dict

    def set_timeseries(
        self,
        swaption_cube_timeseries_dict: Dict[datetime, SCube],
        sabr_params_timeseries_dict: Optional[Dict[datetime, SABRParams]] = None,
    ):
        self._scube_timeseries = swaption_cube_timeseries_dict
        self._sabr_params_timeseries = sabr_params_timeseries_dict

    def to_vanilla_pydt(self, dt: datetime):
        return datetime(dt.year, dt.month, dt.day)

    def return_scube(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
    ) -> Dict[datetime, SCube]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        between_bdates = (
            get_bdates_between(
                start_date=self.to_vanilla_pydt(start_date),
                end_date=self.to_vanilla_pydt(end_date),
                calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
            )
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._scube_timeseries.keys()), set(between_bdates)))

        scubes_to_return: Dict[datetime, SCube] = {}
        for dt in valid_group_dates:
            if dt in self._scube_timeseries:
                scubes_to_return[dt] = self._scube_timeseries[dt]

        return scubes_to_return

    def return_sabr_params(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
    ) -> Dict[datetime, SCube]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        between_bdates = (
            get_bdates_between(
                start_date=self.to_vanilla_pydt(start_date),
                end_date=self.to_vanilla_pydt(end_date),
                calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
            )
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._sabr_params_timeseries.keys()), set(between_bdates)))

        sabr_params_to_return: Dict[datetime, SABRParams] = {}
        for dt in valid_group_dates:
            if dt in self._sabr_params_timeseries:
                sabr_params_to_return[dt] = self._sabr_params_timeseries[dt]

        return sabr_params_to_return

    def parallel_ql_vol_cube_builder(
        self,
        curve: str,
        ql_discount_curves: Dict[datetime, ql.DiscountCurve],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        use_sabr: Optional[bool] = False,
        precalibrate_cube: Optional[bool] = False,
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"

        between_bdates = (
            get_bdates_between(
                start_date=self.to_vanilla_pydt(start_date),
                end_date=self.to_vanilla_pydt(end_date),
                calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
            )
            if (start_date and end_date)
            else [self.to_vanilla_pydt(bday) for bday in bdates]
        )
        valid_group_dates = list(set.intersection(set(self._scube_timeseries.keys()), set(ql_discount_curves.keys()), set(between_bdates)))

        cached_cubes: Dict[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube] = {}
        dates_to_compute = []
        for dt in valid_group_dates:
            if dt in self._ql_cube_cache:
                cached_cubes[dt] = self._ql_cube_cache[dt]
            else:
                dates_to_compute.append(dt)

        computed_cubes: Dict[datetime, ql.DiscountCurve] = {}
        if dates_to_compute:
            max_workers = n_jobs if n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        build_single_vol_cube,
                        dt,
                        curve,
                        self._scube_timeseries[dt],
                        ql_discount_curves[dt],
                        use_sabr,
                        self._sabr_params_timeseries[dt] if self._sabr_params_timeseries is not None and dt in self._sabr_params_timeseries else None,
                        precalibrate_cube,
                        enable_extrapolation,
                    ): dt
                    for dt in dates_to_compute
                }
                if show_tqdm:
                    with tqdm.tqdm(total=len(futures), desc=tqdm_message or "BUILDING QL VOL CUBES...", unit="cube", leave=True) as pbar:
                        for future in as_completed(futures):
                            dt, vol_cube = future.result()
                            if vol_cube is not None:
                                computed_cubes[dt] = vol_cube
                            pbar.update(1)
                else:
                    for future in as_completed(futures):
                        dt, vol_cube = future.result()
                        if vol_cube is not None:
                            computed_cubes[dt] = vol_cube

            self._ql_cube_cache.update(computed_cubes)

        return {**cached_cubes, **computed_cubes}


def build_single_vol_cube(
    as_of_date: datetime,
    curve: str,
    vol_cube: SCube | Dict[int, pd.DataFrame],
    ql_discount_curve: ql.DiscountCurve,
    use_sabr: Optional[bool] = False,
    sabr_params_dict: Optional[Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]]] = None,
    precalibrate_cube: Optional[bool] = False,
    enable_extrapolation: Optional[bool] = True,
) -> Tuple[datetime, ql.InterpolatedSwaptionVolatilityCube | ql.SabrSwaptionVolatilityCube]:
    if CME_SWAP_CURVE_QL_PARAMS[curve]["is_ois"]:
        ql_swap_index_wrapper = ql.OvernightIndexedSwapIndex
    else:
        ql_swap_index_wrapper = ql.SwapIndex

    ql_swap_index = ql_swap_index_wrapper(
        curve,
        CME_SWAP_CURVE_QL_PARAMS[curve]["period"],
        CME_SWAP_CURVE_QL_PARAMS[curve]["settlementDays"],
        CME_SWAP_CURVE_QL_PARAMS[curve]["currency"],
        CME_SWAP_CURVE_QL_PARAMS[curve]["swapIndex"](ql.YieldTermStructureHandle(ql_discount_curve)),
    )

    if sabr_params_dict is not None and use_sabr:
        ql_vol_cube = build_ql_sabr_vol_cube(
            vol_cube=vol_cube,
            sabr_params_dict=sabr_params_dict,
            ql_swap_index=ql_swap_index,
            ql_calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
            ql_day_counter=CME_SWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
            ql_bday_convention=CME_SWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
            pre_calibrate=precalibrate_cube,
            enable_extrapolation=enable_extrapolation,
        )
    else:
        ql_vol_cube = build_ql_interpolated_vol_cube(
            vol_cube=vol_cube,
            ql_swap_index=ql_swap_index,
            ql_calendar=CME_SWAP_CURVE_QL_PARAMS[curve]["calendar"],
            ql_day_counter=CME_SWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
            ql_bday_convention=CME_SWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
            pre_calibrate=precalibrate_cube,
            enable_extrapolation=enable_extrapolation,
        )

    return as_of_date, ql_vol_cube
