from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import QuantLib as ql
from scipy.interpolate import interp1d

from core.utils.ql_utils import datetime_to_ql_date


class SyntheticBond:
    def __init__(self, cf_date, amount=1.0):
        self._cashflows = [SyntheticCashflow(cf_date, amount)]

    def cashflows(self):
        return self._cashflows


class SyntheticCashflow:
    def __init__(self, ql_date, amount):
        self._ql_date = ql_date
        self._amount = amount

    def date(self):
        return self._ql_date

    def amount(self):
        return self._amount


def make_synthetic_repo_bond(gc_rate_annual: float, ttm: float, settle_date: ql.Date) -> tuple:

    def offset_date(settle_date: ql.Date, ttm: float) -> ql.Date:
        days = int(round(ttm * 365))
        return settle_date + ql.Period(days, ql.Days)

    cf_date = offset_date(settle_date, ttm)
    synthetic_price = np.exp(-gc_rate_annual * ttm)
    synthetic_bond = SyntheticBond(cf_date, amount=1.0)
    synthetic_moddur = ttm

    return (synthetic_bond, synthetic_price, synthetic_moddur)


def build_piecewise_df_function(gc_repo_dict: Dict[float, float], short_end_extrapolate: bool = True) -> Callable[[float], float]:
    """
    Returns a function that, for t < 0.5, linearly interpolates discount factors
    from the GC repo dictionary. For t >= 0.5, the function is left unimplemented
    here (it will be handled by a B-spline in the main code), but we do fill in
    the discount factor at 0.5 as a boundary condition for continuity.

    gc_repo_dict should be a dict of {time_in_years: annual_repo_rate}, e.g.:
        {
          0.0833: 0.05,  # 1m
          0.25:   0.052, # 3m
          0.5:    0.055  # 6m
        }
    We convert these to discount factors using DF(t) = exp(-r * t).
    """
    # Sort GC points by time (t)
    times = sorted(gc_repo_dict.keys())
    # Convert annualized repo rates r(t) to discount factors DF(t)
    df_vals = [np.exp(-gc_repo_dict[t] * t) for t in times]

    # Build a 1D interpolator for the short end
    # 'fill_value="extrapolate"' allows t < min(times) or t > max(times)
    # to be extrapolated linearly. Alternatively, you can do 'fill_value=(df_vals[0], df_vals[-1])'
    short_end_interp = interp1d(times, df_vals, kind="linear", fill_value="extrapolate" if short_end_extrapolate else (df_vals[0], df_vals[-1]))

    def piecewise_df_short_end(t: float) -> float:
        """Return discount factor based on GC repo if t < 0.5, else None (placeholder)."""
        if t <= 0.5:
            return float(short_end_interp(t))
        else:
            # For t >= 0.5, we do not define it here; the user code will handle
            # with a B-spline. This is just a placeholder.
            return None

    return piecewise_df_short_end


def check_arbitrage_free_discount_curve(times: np.ndarray, dfs: np.ndarray) -> dict:
    """
    Checks whether a discount factor curve is arbitrage-free.

    Parameters:
    - times: np.ndarray of increasing time-to-maturity values
    - dfs: np.ndarray of discount factors corresponding to each time

    Returns:
    - dict: {
        "monotonic": True/False,
        "positive": True/False,
        "forward_non_negative": True/False,
        "all_conditions_pass": True/False,
        "violations": list of string descriptions
    }
    """
    times = np.asarray(times, dtype=float)
    dfs = np.asarray(dfs, dtype=float)

    violations = []

    # Check monotonicity
    is_monotonic = np.all(np.diff(dfs) <= 0)
    if not is_monotonic:
        violations.append("Discount factors are not non-increasing")

    # Check positivity
    is_positive = np.all((dfs > 0) & (dfs <= 1.0001))
    if not is_positive:
        violations.append("Discount factors are not all in (0, 1]")

    # Check forward rates: f(t1,t2) = ln(D(t1)/D(t2)) / (t2 - t1)
    fwd_rates = np.log(dfs[:-1] / dfs[1:]) / np.diff(times)
    fwd_non_negative = np.all(fwd_rates >= 0)
    if not fwd_non_negative:
        violations.append("Implied forward rates are negative")

    all_ok = is_monotonic and is_positive and fwd_non_negative

    return {"monotonic": is_monotonic, "positive": is_positive, "forward_non_negative": fwd_non_negative, "all_conditions_pass": all_ok, "violations": violations}


def df_scipy_spline_to_par_semibond_scipy_spline(
    scipy_spline_func: Callable[[float], float] | interp1d,
    min_t: float,
    max_t: float,
    step: float = 0.5,
) -> Callable[[float], float]:

    t_grid = np.arange(min_t, max_t + step, step)
    par_rates = []

    for t in t_grid:
        n_periods = int(round(t * 2))
        if n_periods <= 0:
            par_rate = np.nan
        else:
            coupon_dates = np.arange(0.5, t + 1e-8, 0.5)
            df_sum = sum(scipy_spline_func(tau) for tau in coupon_dates)
            df_at_t = scipy_spline_func(t)
            par_rate = (2 * (1 - df_at_t) / df_sum) * 100

        par_rates.append(par_rate)

    par_spline = interp1d(t_grid, par_rates, kind="linear", fill_value="extrapolate")
    return par_spline


def ql_discount_curve_to_par_scipy_spline_analytical(
    ql_discount_curve, min_t: float, max_t: float, step: Optional[float] = 0.5, frequency: Optional[int] = 2, short_rate: Optional[int] = 0
) -> interp1d:

    def par_yield_analytical(ql_discount_curve, tenor_in_years, frequency=2, short_rate: Optional[int] = 0) -> float:
        dt = 1.0 / frequency
        n = int(round(tenor_in_years * frequency))
        times = [(i + 1) * dt for i in range(n)]
        D = [ql_discount_curve.discount(t) for t in times]
        if not D:
            return short_rate / 100

        DF_last = D[-1]
        sum_pv_coupons = sum(dt * df for df in D)
        par_c = (1.0 - DF_last) / sum_pv_coupons
        return par_c

    t_grid = np.arange(min_t, max_t + step, step)
    par_coupons = [par_yield_analytical(ql_discount_curve=ql_discount_curve, tenor_in_years=t, frequency=frequency, short_rate=short_rate) * 100 for t in t_grid]
    par_spline = interp1d(t_grid, par_coupons, kind="cubic", fill_value="extrapolate")
    return par_spline
