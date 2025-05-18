from datetime import datetime
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from core.utils.ql_loader import ql
import tqdm
from joblib import Parallel, delayed

from core.CurveBuilding.IRSwaps.ql_curve_building_utils import build_piecewise_ql_discount_curve, build_ql_discount_curve, get_nodes_dict
from core.CurveBuilding.IRSwaps.CME_IRSWAP_CURVE_QL_PARAMS import CME_IRSWAP_CURVE_QL_PARAMS
from core.utils.ql_utils import datetime_to_ql_date


def make_hashable(x):
    if isinstance(x, (tuple, list)):
        return tuple(make_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
    elif isinstance(x, set):
        return tuple(sorted(make_hashable(e) for e in x))
    else:
        return x


class IRSwapCurveBootstrapper:
    _timeseries_df: pd.DataFrame = None

    def __init__(self, timeseries_df: pd.DataFrame):
        self._timeseries_df = timeseries_df
        if self._timeseries_df is not None:
            assert isinstance(self._timeseries_df.index, pd.DatetimeIndex), "self._timeseries_df index must be a DatetimeIndex"

    def set_timeseries_df(self, timeseries_df: pd.DataFrame):
        self._timeseries_df = timeseries_df
        if self._timeseries_df is not None:
            assert isinstance(self._timeseries_df.index, pd.DatetimeIndex), "self._timeseries_df index must be a DatetimeIndex"

    def parallel_ql_irswap_curve_bootstrapper(
        self,
        curve: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bdates: Optional[List[datetime]] = None,
        ql_interpolation_algo: Optional[
            List[
                Literal[
                    "log_linear",
                    "mono_log_cubic",
                    "natural_cubic",
                    "kruger_log",
                    "natural_log_cubic",
                    "log_mixed_linear",
                    "log_parabolic_cubic",
                    "mono_log_parabolic_cubic",
                ]
            ]
        ] = "pdf_log_linear",
        enable_extrapolation: Optional[bool] = True,
        show_tqdm: Optional[bool] = True,
        tqdm_message: Optional[str] = None,
        n_jobs: Optional[int] = -1,
    ) -> Dict[datetime, ql.YieldTermStructure]:
        assert (start_date and end_date) or bdates, "MUST PASS IN ('start_date' and 'end_date') or 'bdates'"
        assert self._timeseries_df is not None, "'timeseries_df' is required for swap curve bootstrapping"

        if start_date and end_date:
            timeseries_df: pd.DataFrame = self._timeseries_df[
                (self._timeseries_df.index.date >= start_date.date()) & (self._timeseries_df.index.date <= end_date.date())
            ]
        else:
            mask = np.isin(self._timeseries_df.index.date, [bd.date() for bd in bdates])
            timeseries_df = self._timeseries_df.loc[mask]

        data_to_bootstrap = [(date, row.to_dict()) for date, row in timeseries_df.iterrows()]
        data_to_bootstrap_iter = tqdm.tqdm(data_to_bootstrap, desc=tqdm_message or "BOOTSTRAPPING CURVES...") if show_tqdm else data_to_bootstrap
        curves = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_single_irswap_curve)(
                as_of_date=date,
                curve=curve,
                market_data=market_data,
                ql_interpolation_algo=f"pdf_{ql_interpolation_algo}",
                enable_extrapolation=enable_extrapolation,
            )
            for date, market_data in data_to_bootstrap_iter
        )
        parallel_results = {date: curve_obj for (date, _), curve_obj in zip(data_to_bootstrap, curves) if curve_obj is not None}

        ql_curve_dict = {}
        for curve_date, discount_curve_nodes_dict in parallel_results.items():
            if not discount_curve_nodes_dict or len(discount_curve_nodes_dict) == 0:
                continue

            curve_date_norm = pd.to_datetime(curve_date).normalize()
            datetime_series = pd.Series(pd.to_datetime(list(discount_curve_nodes_dict.keys())).normalize())
            type_series = pd.Series(list(discount_curve_nodes_dict.values()))

            if not curve_date_norm in datetime_series.values:
                datetime_series = pd.concat([pd.Series([curve_date]), datetime_series])
                type_series = pd.concat([pd.Series([1]), type_series])

            ql_curve = build_ql_discount_curve(
                datetime_series,
                type_series,
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                f"df_{ql_interpolation_algo}",
            )
            if enable_extrapolation:
                ql_curve.enableExtrapolation()

            ql_curve_dict[curve_date] = ql_curve

        return ql_curve_dict


def bootstrap_single_irswap_curve(
    as_of_date: datetime,
    curve: str,
    market_data: Dict[str, float],
    ql_interpolation_algo: Optional[
        List[
            Literal[
                "pdf_log_linear",
                "pdf_mono_log_cubic",
                "pdf_natural_cubic",
                "pdf_kruger_log",
                "pdf_natural_log_cubic",
                "pdf_log_mixed_linear",
                "pdf_log_parabolic_cubic",
                "pdf_spline_cubic_discount",
                "pdf_mono_log_parabolic_cubic",
            ]
        ]
    ] = "log_linear",
    enable_extrapolation: Optional[bool] = False,
) -> Dict[datetime, float]:
    try:
        ql.Settings.instance().evaluationDate = datetime_to_ql_date(as_of_date)

        CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"]
        dummy_curve = ql.FlatForward(datetime_to_ql_date(as_of_date), 0.0, CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"])
        dummy_handle = ql.YieldTermStructureHandle(dummy_curve)

        if CME_IRSWAP_CURVE_QL_PARAMS[curve]["is_ois"]:
            ql_floating_index = CME_IRSWAP_CURVE_QL_PARAMS[curve]["swapIndex"](dummy_handle)
            rate_helpers = [
                ql.OISRateHelper(
                    CME_IRSWAP_CURVE_QL_PARAMS[curve]["settlementDays"], ql.Period(term), ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)), ql_floating_index
                )
                for term, rate in market_data.items()
                if rate != None and not np.isnan(rate)
            ]
        else:
            ql_floating_index = ql.IborIndex(
                curve,
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["period"],
                2,
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["currency"],
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                False,
                dummy_handle,
            )
            rate_helpers = [
                ql.SwapRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)),
                    ql.Period(term),
                    CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
                    CME_IRSWAP_CURVE_QL_PARAMS[curve]["frequency"],
                    CME_IRSWAP_CURVE_QL_PARAMS[curve]["businessConvention"],
                    CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
                    ql_floating_index,
                )
                for term, rate in market_data.items()
            ]

        ql_discount_curve = build_piecewise_ql_discount_curve(
            swap_rate_helpers=rate_helpers,
            ql_dc=CME_IRSWAP_CURVE_QL_PARAMS[curve]["dayCounter"],
            ql_cal=CME_IRSWAP_CURVE_QL_PARAMS[curve]["calendar"],
            settlement_day=CME_IRSWAP_CURVE_QL_PARAMS[curve]["settlementDays"],
            interpolation_algo=ql_interpolation_algo,
        )
        if enable_extrapolation:
            ql_discount_curve.enableExtrapolation()

        return get_nodes_dict(ql_discount_curve)

    except Exception as e:
        print(f"ERROR DURING {as_of_date} {curve} SWAP CURVE BOOTSTRAP: ", e)
        return None
