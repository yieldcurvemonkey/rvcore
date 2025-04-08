from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from core.Products.CurveBuilding.Cash.ParCurveModel.ParCurveModelBase import ParCurveModelBase
from core.Products.CurveBuilding.Cash.ParCurveModel.utils import build_piecewise_df_function, check_arbitrage_free_discount_curve
from core.Products.CurveBuilding.ql_curve_building_utils import build_piecewise_curve_from_discount_factor_scipy_spline
from core.utils.ql_utils import datetime_to_ql_date


class JPM2024(ParCurveModelBase):
    _gc_repo_dict: Dict[float, float] = None
    _continuity_weight: int = 1000

    _knot_positions: List[int] = None
    _min_free_float_mm: int | float = None
    _min_ttm_years: int | float = None
    _custom_filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    _issue_date_col: str = None
    _maturity_date_col: str = None
    _coupon_col: str = None
    _price_col: str = None
    _ytm_col: str = None
    _ttm_col: str = None
    _free_float_col: str = None
    _rank_col: str = None
    _security_type_col: str = None
    _original_security_term_col: str = None
    _bill_col: str = None

    _ytm_col: str = None
    _ql_cal: ql.Calendar = None
    _ql_day_count: ql.DayCounter = None
    _ql_bdc = None
    _ql_freq = None
    _ql_compounded = None
    _settlement_t_plus: int = None

    _MAX_LINSPACE = 1000
    _MAX_MATURITIES_YEARS = 30

    def __init__(
        self,
        cusip_set_df: pd.DataFrame,
        gc_repo_dict: Dict[float, float],
        continuity_weight: Optional[int] = 1000,
        knot_positions: Optional[List[int]] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0, 15.0, 20.0, 25.0, 30],
        min_free_float_mm: Optional[int | float] = 5000,
        min_ttm_years: Optional[int | float] = 1,
        custom_filter_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        timestamp_col: Optional[str] = "timestamp",
        issue_date_col: Optional[str] = "issue_date",
        maturity_date_col: Optional[str] = "maturity_date",
        coupon_col: Optional[str] = "coupon",
        price_col: Optional[str] = "eod_price",
        ytm_col: Optional[str] = "eod_ytm",
        ttm_col: Optional[str] = "time_to_maturity",
        free_float_col: Optional[str] = "free_float",
        rank_col: Optional[str] = "rank",
        security_type_col: Optional[str] = "security_type",
        original_security_term_col: Optional[str] = "original_security_term",
        bill_col: Optional[str] = "Bill",
        ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        ql_day_count: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),
        ql_bdc: Optional[Any] = ql.ModifiedFollowing,
        ql_freq: Optional[Any] = ql.Semiannual,
        ql_compounded: Optional[Any] = ql.Compounded,
        settlement_t_plus: Optional[int] = 1,
        debug_verbose: bool = False,
        info_verbose: bool = False,
        warning_verbose: bool = False,
        error_verbose: bool = False,
    ):
        super().__init__(
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
            error_verbose=error_verbose,
        )

        self._cusip_set_df = cusip_set_df.copy()
        self._gc_repo_dict = gc_repo_dict
        self._continuity_weight = continuity_weight

        self._knot_positions = knot_positions
        self._min_free_float_mm = min_free_float_mm
        self._min_ttm_years = min_ttm_years
        self._custom_filter_func = custom_filter_func

        self._issue_date_col = issue_date_col
        self._maturity_date_col = maturity_date_col
        self._coupon_col = coupon_col
        self._timestamp_col = timestamp_col
        self._price_col = price_col
        self._ytm_col = ytm_col
        self._ttm_col = ttm_col
        self._free_float_col = free_float_col
        self._rank_col = rank_col
        self._security_type_col = security_type_col
        self._original_security_term_col = original_security_term_col
        self._bill_col = bill_col
        self._ql_cal = ql_cal
        self._ql_day_count = ql_day_count
        self._ql_bdc = ql_bdc
        self._ql_freq = ql_freq
        self._ql_compounded = ql_compounded
        self._settlement_t_plus = settlement_t_plus

        self._df_short_end_func = build_piecewise_df_function(gc_repo_dict=self._gc_repo_dict)

    def _make_bond_from_row(
        self,
        row: pd.Series,
        settlement_date: ql.Date,
    ):
        issue_d = pd.to_datetime(row[self._issue_date_col])
        mat_d = pd.to_datetime(row[self._maturity_date_col])
        coup = float(row[self._coupon_col]) / 100.0

        schedule = ql.Schedule(
            ql.Date(issue_d.day, issue_d.month, issue_d.year),
            ql.Date(mat_d.day, mat_d.month, mat_d.year),
            ql.Period(self._ql_freq),
            self._ql_cal,
            self._ql_bdc,
            self._ql_bdc,
            ql.DateGeneration.Backward,
            False,
        )
        bond = ql.FixedRateBond(self._settlement_t_plus, 100.0, schedule, [coup], self._ql_day_count)
        obs_price = float(row[self._price_col])

        ts_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, 0.0, self._ql_day_count, self._ql_compounded, self._ql_freq))
        engine = ql.DiscountingBondEngine(ts_handle)
        bond.setPricingEngine(engine)
        mdur = ql.BondFunctions.duration(bond, 0.0, self._ql_day_count, self._ql_compounded, self._ql_freq)

        return bond, obs_price, mdur

    def _build_cubic_bspline_knots(self, knots: List[float], k: Optional[int] = 3) -> np.ndarray:
        t = np.array(knots, dtype=float)
        t0, tN = t[0], t[-1]
        T = np.concatenate(([t0] * k, t, [tN] * k))
        return T

    def _price_residuals(
        self, coeffs: np.ndarray, bonds_data: List[Tuple[ql.Bond, float, float]], bspline: BSpline, settlement_date: ql.Date, day_counter: ql.DayCounter
    ) -> np.ndarray:
        """
        We compute residuals = w_i * (model_price - obs_price), where w_i = 1 / (P_i * duration_i).
        This ensures we are effectively minimizing squared yield errors.

        The discount factor is piecewise:
          d(t) = DF_short_end(t)  if t < 0.5,
                 bspline(t)       if t >= 0.5

        We also add a continuity constraint at t=0.5 to ensure bspline(0.5) matches DF_short_end(0.5).
        """
        new_spline = BSpline.construct_fast(bspline.t, coeffs, bspline.k, extrapolate=True)

        residuals = []
        for i, (bond, obs_price, mdur) in enumerate(bonds_data):
            model_price = 0.0
            valid_bond = True

            for cf in bond.cashflows():
                cf_date = cf.date()
                if cf_date > settlement_date:
                    ttm = day_counter.yearFraction(settlement_date, cf_date)
                    if ttm <= 0.5:  # tol
                        df = self._df_short_end_func(ttm)
                    else:
                        df = new_spline(ttm)

                    if df is None or np.isnan(df) or df <= 0 or df > 1.0001:
                        self._logger.warning(f"Warning: Invalid DF ({df:.4f}) at TTM {ttm:.4f} for bond {i}. Skipping bond in this residual calculation.")
                        valid_bond = False
                        break

                    model_price += cf.amount() * df

            if not valid_bond:
                # print(f"{valid_bond} not valid")
                residuals.append(0.0)
                continue

            w_i = 1.0 / (obs_price * mdur)
            res = w_i * (model_price - obs_price)
            residuals.append(res)
        # return residuals

        big_weight = self._continuity_weight
        repo_constraints = []
        for t_repo, rate_annual in self._gc_repo_dict.items():
            df_target = np.exp(-rate_annual * t_repo)
            repo_residual = big_weight * (new_spline(t_repo) - df_target)
            repo_constraints.append(repo_residual)

        return np.append(residuals, repo_constraints)

    def _filter(self, cusip_set_df: pd.DataFrame) -> pd.DataFrame:
        df = cusip_set_df.copy()
        mask = (
            ~(
                (df[self._original_security_term_col].isin(["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year"]) & df[self._rank_col].isin([0, 1, 2]))
                | ((df[self._original_security_term_col] == "30-Year") & (df[self._rank_col] == 0))
            )
            & (df[self._security_type_col] != self._bill_col)
            & (df[self._free_float_col] > self._min_free_float_mm)
            & (df[self._ttm_col] >= self._min_ttm_years)
        )
        df = df.loc[mask]

        if self._custom_filter_func:
            try:
                df = self._custom_filter_func(df)
            except Exception as e:
                self._logger.error(f"'JPM2024._filter': custom filter function had an error: {e}")

        df = df.dropna(subset=[self._ttm_col, self._ytm_col]).sort_values(by=[self._ttm_col])
        return df

    def fit(self):
        df = self._filter(self._cusip_set_df)
        self._filtered_cusip_set_df = df
        if df.empty:
            raise ValueError(f"'JPM2024.fit': {self._FILTER_TOO_STRONG}")

        k = 3  # cubic
        T = self._build_cubic_bspline_knots(self._knot_positions, k=k)
        n_coeff = len(T) - k - 1

        # initial short end guess for the B-spline portion
        c0 = np.full(n_coeff, 0.98)
        placeholder_coeffs = np.zeros(n_coeff)
        base_spline = BSpline(T, placeholder_coeffs, k, extrapolate=True)

        curve_dt = pd.to_datetime(df[self._timestamp_col].max())
        settlement_date = datetime_to_ql_date(curve_dt)
        ql.Settings.instance().evaluationDate = settlement_date

        bonds_data = []
        for _, row in df.iterrows():
            bond, obs_p, mdur = self._make_bond_from_row(row, settlement_date)
            bonds_data.append((bond, obs_p, mdur))

        def fun_res(c):
            return self._price_residuals(c, bonds_data, base_spline, settlement_date, self._ql_day_count)

        # res = least_squares(fun_res, c0, loss="soft_l1", f_scale=0.25)
        res = least_squares(fun_res, c0)
        c_opt = res.x
        final_spline = BSpline.construct_fast(base_spline.t, c_opt, base_spline.k, extrapolate=True)

        try:
            df_at_0_5 = self._df_short_end_func(0.5)  # Get DF from 6m repo
            df_at_1_0 = float(final_spline(1.0))  # Get DF from fitted spline at t=1.0
        except Exception as e:
            raise ValueError(f"Could not evaluate boundary points for interpolation: {e}")

        if df_at_0_5 is None or np.isnan(df_at_0_5) or df_at_0_5 <= 0 or df_at_0_5 > 1.0001:
            raise ValueError(f"Invalid discount factor calculated at t=0.5: {df_at_0_5}")
        if np.isnan(df_at_1_0) or df_at_1_0 <= 0 or df_at_1_0 > 1.0001:
            print(f"Warning: Clamping potentially invalid spline DF at t=1.0: {df_at_1_0}")
            df_at_1_0 = max(1e-6, min(df_at_1_0, 1.0001))

        def piecewise_discount_factor_linear_interp(ttm: float) -> float:
            if ttm <= 0.5:
                return self._df_short_end_func(ttm)
            return float(final_spline(ttm))

        max_maturity = max(df[self._ttm_col].max(), self._MAX_MATURITIES_YEARS)
        all_times = np.linspace(0.0, max_maturity, self._MAX_LINSPACE)
        discounted = []
        for ttm in all_times:
            # Use the function with linear interpolation now
            dfac = piecewise_discount_factor_linear_interp(ttm)
            discounted.append((ttm, dfac))

        curve_df = pd.DataFrame(discounted, columns=[self._TO_RETURN_TTM_COL, self._TO_RETURN_DF_COL])

        self._optimization_results = res
        self._fitted_curve_df = curve_df
        # Store the NEW function as the model's spline function
        self._scipy_spline_func = piecewise_discount_factor_linear_interp
        self._ql_discount_curve = build_piecewise_curve_from_discount_factor_scipy_spline(
            spline_func=piecewise_discount_factor_linear_interp,  # Pass the new function
            anchor_date=curve_dt,
            ql_calendar=self._ql_cal,
            ql_business_convention=self._ql_bdc,
            ql_day_counter=self._ql_day_count,
            ql_interpolation_algo="log_linear",  # Or try "linear" if issues arise
            enable_extrapolation=True,
        )
        self._arb_free_results = check_arbitrage_free_discount_curve(
            self._fitted_curve_df[self._TO_RETURN_TTM_COL].to_numpy(), self._fitted_curve_df[self._TO_RETURN_DF_COL].to_numpy()
        )


def calculate_jpm_rmse(jpm_par_curve_model: JPM2024) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) in basis points
    for the fitted JPM2024 model against the bonds used in fitting.

    Args:
        jpm_par_curve_model: A fitted instance of the JPM2024 class.

    Returns:
        The RMSE value in basis points, or NaN if calculation is not possible.

    Raises:
        ValueError: If the model has not been fitted yet.
    """
    # --- Check if model is fitted ---
    if jpm_par_curve_model._filtered_cusip_set_df is None or jpm_par_curve_model._scipy_spline_func is None:
        raise ValueError("Model has not been fitted yet. Please call .fit() first.")

    # --- Extract necessary components from the fitted model ---
    df_filtered = jpm_par_curve_model._filtered_cusip_set_df
    # The fitted spline function (for t >= 0.5)
    fitted_spline = jpm_par_curve_model._scipy_spline_func
    # The repo interpolation function (for t < 0.5)
    df_short_end_func = jpm_par_curve_model._df_short_end_func
    day_counter = jpm_par_curve_model._ql_day_count
    timestamp_col = jpm_par_curve_model._timestamp_col  # Get col name from model instance

    # --- Determine settlement date (must be consistent with fit method) ---
    curve_dt = pd.to_datetime(df_filtered[timestamp_col].max())
    # You might need a utility function datetime_to_ql_date(curve_dt) here
    # Using a basic conversion for demonstration:
    try:
        settlement_date = ql.Date(curve_dt.day, curve_dt.month, curve_dt.year)
    except Exception as e:
        raise ValueError(f"Could not convert curve date {curve_dt} to QuantLib date: {e}")

    # Set evaluation date for QuantLib calculations
    eval_date_before = ql.Settings.instance().evaluationDate
    ql.Settings.instance().evaluationDate = settlement_date

    yield_errors_sq_bp = []
    bonds_used_in_rmse = 0

    # --- Iterate through the bonds used for fitting ---
    for idx, row in df_filtered.iterrows():
        try:
            # --- Recreate Bond Object and Calculate Duration ---
            # We need to recalculate this as it wasn't stored.
            # This uses the model's internal method. Ensure it's accessible
            # or reimplement the logic here.
            bond, obs_price, mdur = jpm_par_curve_model._make_bond_from_row(row, settlement_date)

            # Basic check for valid duration
            if mdur is None or mdur <= 1e-6:  # Avoid division by zero/tiny duration
                print(f"Warning: Skipping bond index {idx} due to invalid duration: {mdur}")
                continue

            # --- Calculate Model Price using the Fitted Curve ---
            model_price = 0.0
            valid_cfs = True
            for cf in bond.cashflows():
                cf_date = cf.date()
                # Only consider cashflows after settlement
                if cf_date > settlement_date:
                    ttm = day_counter.yearFraction(settlement_date, cf_date)

                    # Apply the piecewise discount factor logic
                    if ttm <= 0.5:
                        df_val = df_short_end_func(ttm)
                    else:
                        df_val = fitted_spline(ttm)  # Use the fitted spline

                    # Check for invalid discount factor results
                    if df_val is None or np.isnan(df_val) or df_val <= 0:
                        print(f"Warning: Invalid discount factor ({df_val}) for TTM {ttm:.4f} in bond index {idx}. Skipping bond.")
                        valid_cfs = False
                        break  # Stop processing this bond

                    model_price += cf.amount() * df_val

            if not valid_cfs:
                continue  # Skip this bond if any DF was invalid

            # --- Calculate JPM Yield Error Term ---
            # e_i^Y = (model_price - observed_price) / (observed_price * modified_duration)
            e_i_Y = (model_price - obs_price) / (obs_price * mdur)

            # --- Convert to Basis Points and Square ---
            # e_i_bp^Y ≈ -e_i^Y * 10000
            e_i_bp_Y = -e_i_Y * 10000
            yield_errors_sq_bp.append(e_i_bp_Y**2)
            bonds_used_in_rmse += 1

        except Exception as e:
            # Log or print error for the specific bond and continue
            cusip = row.get("cusip", f"Index {idx}")  # Attempt to get CUSIP if column exists
            print(f"Warning: Could not process bond {cusip} for RMSE calculation: {e}")
            continue  # Skip this bond if any error occurs

    # Restore original evaluation date
    ql.Settings.instance().evaluationDate = eval_date_before

    # --- Calculate Final RMSE ---
    if bonds_used_in_rmse == 0:
        print("Warning: No bonds could be processed for RMSE calculation.")
        return np.nan  # Or raise an error

    rmse_bp = np.sqrt(np.mean(yield_errors_sq_bp))

    return rmse_bp


def calculate_individual_yield_errors(
    jpm_par_curve_model: JPM2024, securities_df: pd.DataFrame, identifier_cols: List[str] = ["cusip"]  # Specify columns to identify each bond
) -> pd.DataFrame:
    """
    Calculates the JPM "yield error" in basis points for each security
    in the provided DataFrame using the fitted JPM2024 model.

    This error represents the richness/cheapness of each bond relative
    to the fitted curve. A positive yield error means the bond's market
    yield is higher than the model's implied yield (i.e., cheap), and
    a negative error means the bond's market yield is lower (i.e., rich).

    Args:
        jpm_par_curve_model: A fitted instance of the JPM2024 class.
        securities_df: DataFrame with market data for securities to evaluate.
                       Must contain columns required by _make_bond_from_row
                       (price, coupon, issue_date, maturity_date, timestamp)
                       and columns listed in identifier_cols.
        identifier_cols: List of column names in securities_df to include
                         in the output DataFrame for identification (e.g.,
                         ['cusip', 'maturity_date']).

    Returns:
        A pandas DataFrame with identifier columns and the calculated
        'yield_error_bp' for each security. Bonds causing errors during
        calculation will have NaN as their yield_error_bp.
    """
    # --- Check if model is fitted ---
    if jpm_par_curve_model._scipy_spline_func is None:
        raise ValueError("Model has not been fitted yet. Please call .fit() first.")

    # --- Extract necessary components from the fitted model ---
    fitted_spline = jpm_par_curve_model._scipy_spline_func  # For t >= 0.5
    df_short_end_func = jpm_par_curve_model._df_short_end_func  # For t < 0.5
    day_counter = jpm_par_curve_model._ql_day_count
    timestamp_col = jpm_par_curve_model._timestamp_col  # Get relevant column name

    # --- Determine settlement date from the LATEST timestamp in the INPUT df ---
    # This assumes the input df represents data for a single date snapshot.
    # The curve should have been fitted using data from this same date.
    try:
        curve_dt = pd.to_datetime(securities_df[timestamp_col].max())
        # Basic conversion, replace if you have a datetime_to_ql_date utility
        settlement_date = ql.Date(curve_dt.day, curve_dt.month, curve_dt.year)
    except Exception as e:
        raise ValueError(f"Could not determine settlement date from input DataFrame using column '{timestamp_col}': {e}")

    # Set evaluation date for QuantLib calculations temporarily
    eval_date_before = ql.Settings.instance().evaluationDate
    ql.Settings.instance().evaluationDate = settlement_date

    results = []  # To store results for each bond

    # --- Iterate through the provided securities ---
    for idx, row in securities_df.iterrows():
        if row["time_to_maturity"] < 1:
            continue

        # Prepare dictionary to hold results for this bond
        bond_result = {col: row.get(col) for col in identifier_cols}
        bond_result["yield_error_bp"] = np.nan  # Default to NaN

        try:
            # --- Recreate Bond Object and Calculate Duration ---
            # This uses the model's internal method _make_bond_from_row.
            # Ensure this method is accessible or replicate its logic.
            bond, obs_price, mdur = jpm_par_curve_model._make_bond_from_row(row, settlement_date)

            # Check for valid duration
            if mdur is None or mdur <= 1e-6:
                print(f"Warning: Skipping {bond_result.get('cusip','Index '+str(idx))} due to invalid duration: {mdur}")
                results.append(bond_result)
                continue

            # --- Calculate Model Price using the Fitted Curve ---
            model_price = 0.0
            valid_cfs = True
            for cf in bond.cashflows():
                cf_date = cf.date()
                if cf_date > settlement_date:
                    ttm = day_counter.yearFraction(settlement_date, cf_date)

                    # Apply the piecewise discount factor logic
                    if ttm <= 0.5:
                        df_val = df_short_end_func(ttm)
                    else:
                        df_val = fitted_spline(ttm)  # Use the fitted spline

                    # Check for invalid discount factor results
                    if df_val is None or np.isnan(df_val) or df_val <= 0:
                        print(f"Warning: Invalid DF ({df_val}) for TTM {ttm:.4f} in {bond_result.get('cusip','Index '+str(idx))}. Skipping.")
                        valid_cfs = False
                        break  # Stop processing this bond's cashflows

                    model_price += cf.amount() * df_val

            if not valid_cfs:
                results.append(bond_result)  # Append result with NaN error
                continue  # Skip to next bond

            # --- Calculate JPM Yield Error Term & Convert to Basis Points ---
            # e_i^Y = (model_price - observed_price) / (observed_price * modified_duration)
            w_i = 1 / (obs_price * mdur)
            e_i_Y = (model_price - obs_price) * w_i

            # yield_error_bp ≈ -e_i^Y * 10000
            # Positive error = cheap (market yield > model yield)
            # Negative error = rich (market yield < model yield)
            bond_result["model_price"] = model_price
            bond_result["obs_price"] = obs_price

            bond_result["model_ytm"] = (
                ql.BondFunctions.bondYield(bond, model_price, day_counter, jpm_par_curve_model._ql_compounded, jpm_par_curve_model._ql_freq) * 100
            )
            bond_result["eod_ytm"] = row["eod_ytm"]

            bond_result["mdur"] = mdur

            bond_result["yield_error_bp"] = -e_i_Y * 10000
            bond_result["yield_error_bp_2"] = (bond_result["eod_ytm"] - bond_result["model_ytm"]) * 100

        except Exception as e:
            # Log or print error for the specific bond and keep yield_error_bp as NaN
            print(f"Warning: Could not calculate yield error for {bond_result.get('cusip','Index '+str(idx))}: {e}")
            # yield_error_bp remains NaN as initialized

        # Append the result for this bond (either calculated value or NaN)
        results.append(bond_result)

    # Restore original QuantLib evaluation date
    ql.Settings.instance().evaluationDate = eval_date_before

    # --- Create Output DataFrame ---
    results_df = pd.DataFrame(results)
    return results_df
