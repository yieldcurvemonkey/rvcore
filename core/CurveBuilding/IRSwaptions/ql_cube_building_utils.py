import pandas as pd
import QuantLib as ql
from typing import Optional, List, Literal, Dict, Optional, Literal

from core.utils.ql_utils import period_to_string, period_to_months, parse_tenor_string


def build_ql_interpolated_vol_cube(
    vol_cube: Dict[int | float, pd.DataFrame],
    ql_swap_index: ql.SwapIndex,
    pre_calibrate: Optional[bool] = False,
) -> ql.InterpolatedSwaptionVolatilityCube:
    atm_df = vol_cube[0]
    expiries = [ql.Period(t) for t in atm_df.index]
    tails = [ql.Period(t) for t in atm_df.columns]

    atm_np = atm_df.to_numpy(dtype=float) / 10_000.0
    atm_mat = ql.Matrix([row.tolist() for row in atm_np])
    atm_vol_structure = ql.SwaptionVolatilityMatrix(
        ql_swap_index.fixingCalendar(),
        ql_swap_index.fixedLegConvention(),
        expiries,
        tails,
        atm_mat,
        ql_swap_index.dayCounter(),
        False,
        ql.Normal,
    )

    strike_keys = sorted(vol_cube.keys(), key=float)
    strike_spreads = [float(k) / 10_000.0 for k in strike_keys]
    diff_surfaces = {k: (vol_cube[k].to_numpy(dtype=float) - atm_df.to_numpy(dtype=float)) / 10_000.0 for k in strike_keys}
    rows, n_opt, n_swap = [], len(expiries), len(tails)
    for i in range(n_opt):
        for j in range(n_swap):
            row = [ql.QuoteHandle(ql.SimpleQuote(diff_surfaces[k][i, j])) for k in strike_keys]
            rows.append(row)

    ql_cube = ql.InterpolatedSwaptionVolatilityCube(
        ql.SwaptionVolatilityStructureHandle(atm_vol_structure),
        expiries,
        tails,
        strike_spreads,
        rows,
        ql_swap_index,
        ql_swap_index,
        False,
    )

    if pre_calibrate:
        try:
            _ = ql_cube.volatility(0.25, 10, 0.04, True)
        except Exception:
            pass

    ql_cube.enableExtrapolation()
    return ql_cube


def _build_sabr_guess(
    sabr_params_dict: Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]], spread_opt_tenors: List[ql.Period], spread_swap_tenors: List[ql.Period]
):
    guess_list = []
    for opt in spread_opt_tenors:
        opt_str = period_to_string(opt)
        for swap in spread_swap_tenors:
            swap_str = period_to_string(swap)
            key_candidate = opt_str + "x" + swap_str
            if key_candidate in sabr_params_dict:
                params = sabr_params_dict[key_candidate]
            else:
                candidates = []
                for key in sabr_params_dict:
                    if key.endswith("x" + swap_str):
                        candidate_opt_str = key.split("x")[0]
                        candidate_opt = parse_tenor_string(candidate_opt_str)
                        diff = abs(period_to_months(candidate_opt) - period_to_months(opt))
                        candidates.append((diff, key))
                if candidates:
                    best_key = sorted(candidates, key=lambda x: x[0])[0][1]
                    params = sabr_params_dict[best_key]
                else:
                    params = {"alpha": 0.2, "beta": 0.5, "nu": 0.4, "rho": 0.0}

            guess_tuple = (
                ql.QuoteHandle(ql.SimpleQuote(params["alpha"])),
                ql.QuoteHandle(ql.SimpleQuote(params["beta"])),
                ql.QuoteHandle(ql.SimpleQuote(params["nu"])),
                ql.QuoteHandle(ql.SimpleQuote(params["rho"])),
            )
            guess_list.append(guess_tuple)

    return guess_list


def build_ql_sabr_vol_cube(
    vol_cube: Dict[int, pd.DataFrame],
    sabr_params_dict: Dict[str, Dict[Literal["alpha", "beta", "nu", "rho"], float]],
    ql_swap_index: ql.SwapIndex,
    pre_calibrate: Optional[bool] = False,
) -> ql.SabrSwaptionVolatilityCube:
    ql.Settings.instance().evaluationDate = ql_swap_index.forwardingTermStructure().referenceDate()
    atm_vol_grid = vol_cube[0]

    spread_opt_tenors = [ql.Period(e) for e in atm_vol_grid.index]
    spread_swap_tenors = [ql.Period(t) for t in atm_vol_grid.columns]

    atm_swaption_vol_matrix = ql.SwaptionVolatilityMatrix(
        ql_swap_index.fixingCalendar(),
        ql_swap_index.fixedLegConvention(),
        spread_opt_tenors,
        spread_swap_tenors,
        ql.Matrix([[vol / 10_000 for vol in row] for row in atm_vol_grid.values]),
        ql_swap_index.dayCounter(),
        False,
        ql.Normal,
    )

    sorted_keys = sorted(vol_cube.keys(), key=lambda x: float(x))
    strike_spreads = [float(k) / 10_000 for k in sorted_keys]

    vol_spreads = []
    for option_tenor in atm_vol_grid.index:
        for swap_tenor in atm_vol_grid.columns:
            vol_spread_row = [
                ql.QuoteHandle(ql.SimpleQuote((vol_cube[k].loc[option_tenor, swap_tenor] - atm_vol_grid.loc[option_tenor, swap_tenor]) / 10_000))
                for k in sorted_keys
            ]
            vol_spreads.append(vol_spread_row)

    guess = _build_sabr_guess(sabr_params_dict, spread_opt_tenors, spread_swap_tenors)
    is_parameter_fixed = (False, True, False, False)

    # https://quant.stackexchange.com/questions/75537/sabrswaptionvolcube-class-in-quantilib-python
    ql_sabr_vol_cube = ql.SabrSwaptionVolatilityCube(
        ql.SwaptionVolatilityStructureHandle(atm_swaption_vol_matrix),
        spread_opt_tenors,
        spread_swap_tenors,
        strike_spreads,
        vol_spreads,
        ql_swap_index,
        ql_swap_index,
        False,
        guess,
        is_parameter_fixed,
        True,
        None,
        0.25,
    )

    if pre_calibrate:
        pre_calibration_expiry = 0.25
        pre_calibration_tails = 10
        pre_calibration_strike = 0.04
        try:
            _ = ql_sabr_vol_cube.volatility(pre_calibration_expiry, pre_calibration_tails, pre_calibration_strike, True)
        except Exception:
            pass

    ql_sabr_vol_cube.enableExtrapolation()
    return ql_sabr_vol_cube
