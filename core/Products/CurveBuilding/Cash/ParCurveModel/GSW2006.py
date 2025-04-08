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


class GSW2006(ParCurveModelBase):
    # Column names (similar to JPM2024)
    _issue_date_col: str
    _maturity_date_col: str
    _coupon_col: str
    _price_col: str
    _ttm_col: str  # Time to maturity in years needed for filtering
    _rank_col: str  # Needed for OTR filtering
    _security_type_col: str  # Needed for Bill/Callable filtering
    _original_security_term_col: str  # Needed for 20yr bond filtering
    _callable_col: str  # Assumes a boolean column indicating callability
    _timestamp_col: str

    _MAX_LINSPACE = 1000  # For generating final curve points
    _MAX_MATURITIES_YEARS = 30  # For generating final curve points

    # QuantLib settings
    _ql_cal: ql.Calendar
    _ql_day_count: ql.DayCounter
    _ql_bdc: Any
    _ql_freq: Any
    _ql_compounded: Any
    _settlement_t_plus: int

    def __init__(
        self,
        cusip_set_df: pd.DataFrame,
        # GSW doesn't use repo, continuity_weight, or knots
        # Parameters specific to GSW filtering
        filter_maturity_cutoff_years: float = 0.25,  # GSW uses 3 months
        # Column names
        timestamp_col: Optional[str] = "timestamp",
        issue_date_col: Optional[str] = "issue_date",
        maturity_date_col: Optional[str] = "maturity_date",
        coupon_col: Optional[str] = "coupon",
        price_col: Optional[str] = "eod_price",
        ttm_col: Optional[str] = "time_to_maturity",  # Needs TTM calculated
        rank_col: Optional[str] = "rank",  # Assuming 0=OTR, 1=First-off
        security_type_col: Optional[str] = "security_type",  # e.g., 'Note', 'Bond', 'Bill'
        callable_col: Optional[str] = "callable",  # Assumes boolean column exists
        original_security_term_col: Optional[str] = "original_security_term",  # e.g., '20-Year'
        bill_type_str: Optional[str] = "Bill",  # String identifying Treasury Bills
        # QL settings (can reuse JPM defaults)
        ql_cal: Optional[ql.Calendar] = ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        ql_day_count: Optional[ql.DayCounter] = ql.ActualActual(ql.ActualActual.Actual365),  # Common for Treasuries
        ql_bdc: Optional[Any] = ql.ModifiedFollowing,  # Business day convention
        ql_freq: Optional[Any] = ql.Semiannual,
        ql_compounded: Optional[Any] = ql.Compounded,  # Match GSW formula basis
        settlement_t_plus: Optional[int] = 1,
        # Verbosity
        debug_verbose: bool = False,
        info_verbose: bool = False,
        warning_verbose: bool = False,
        error_verbose: bool = False,
    ):
        super().__init__(debug_verbose, info_verbose, warning_verbose, error_verbose)

        self._cusip_set_df = cusip_set_df.copy()

        # Store GSW-specific filter parameters
        self._filter_maturity_cutoff_years = filter_maturity_cutoff_years

        # Store column names
        self._issue_date_col = issue_date_col
        self._maturity_date_col = maturity_date_col
        self._coupon_col = coupon_col
        self._price_col = price_col
        self._ttm_col = ttm_col
        self._rank_col = rank_col
        self._security_type_col = security_type_col
        self._callable_col = callable_col
        self._original_security_term_col = original_security_term_col
        self._bill_type_str = bill_type_str
        self._timestamp_col = timestamp_col

        # Store QL settings
        self._ql_cal = ql_cal
        self._ql_day_count = ql_day_count
        self._ql_bdc = ql_bdc
        self._ql_freq = ql_freq
        self._ql_compounded = ql_compounded
        self._settlement_t_plus = settlement_t_plus

        # Ensure necessary columns exist
        required_cols = [
            timestamp_col,
            issue_date_col,
            maturity_date_col,
            coupon_col,
            price_col,
            ttm_col,
            rank_col,
            security_type_col,
            callable_col,
            original_security_term_col,
        ]
        for col in required_cols:
            if col not in self._cusip_set_df.columns:
                raise ValueError(f"Required column '{col}' not found in input DataFrame.")

    def _filter(self, cusip_set_df: pd.DataFrame) -> pd.DataFrame:
        """Applies filtering rules based on GSW (2006) paper."""
        df = cusip_set_df.copy()
        self._logger.info(f"Starting filter. Initial rows: {len(df)}")

        df[self._timestamp_col] = pd.to_datetime(df[self._timestamp_col])

        maturity_mask = df[self._ttm_col] >= self._filter_maturity_cutoff_years
        df = df[maturity_mask]
        self._logger.info(f"After maturity cutoff ({self._filter_maturity_cutoff_years} yrs): {len(df)} rows")

        bill_mask = df[self._security_type_col] != self._bill_type_str
        df = df[bill_mask]
        self._logger.info(f"After bill filter: {len(df)} rows")

        df = df.dropna(subset=[self._price_col, self._coupon_col, self._ttm_col])
        self._logger.info(f"After dropping NA price/coupon/ttm: {len(df)} rows")
        df = df.sort_values(by=[self._ttm_col])

        return df

    # --- Nelson-Siegel-Svensson (NSS) Functions ---
    def _nss_yield_function(self, n: float, params: List[float]) -> float:
        """Calculates zero-coupon yield using NSS formula (Eq 22 GSW)."""
        b0, b1, b2, b3, t1, t2 = params
        # Handle potential division by zero if n is very small or zero
        if abs(n * t1) < 1e-8:
            term1 = 0.0
        else:
            term1 = (1.0 - np.exp(-n / t1)) / (n / t1)

        if abs(n * t1) < 1e-8:
            term2 = 0.0
        else:
            term2 = term1 - np.exp(-n / t1)

        if abs(n * t2) < 1e-8:
            term3 = 0.0
        else:
            term3 = (1.0 - np.exp(-n / t2)) / (n / t2)

        if abs(n * t2) < 1e-8:
            term4 = 0.0
        else:
            term4 = term3 - np.exp(-n / t2)

        return b0 + b1 * term1 + b2 * term2 + b3 * term4

    def _nss_discount_factor(self, n: float, params: List[float]) -> float:
        """Calculates discount factor d(n) = exp(-n*y(n))."""
        if n <= 1e-8:  # Handle t=0 case
            return 1.0
        yield_n = self._nss_yield_function(n, params)
        return np.exp(-n * yield_n)

    # --- Can likely reuse JPM's _make_bond_from_row ---
    # Copying here for completeness, ensure consistency with QL settings
    def _make_bond_from_row(
        self,
        row: pd.Series,
        settlement_date: ql.Date,
    ) -> Tuple[ql.FixedRateBond, float, float]:
        """Creates QuantLib bond object and gets observed price & duration."""
        try:
            # Convert dates, ensuring they are timezone naive if needed
            issue_d_dt = pd.to_datetime(row[self._issue_date_col]).tz_localize(None)
            mat_d_dt = pd.to_datetime(row[self._maturity_date_col]).tz_localize(None)
            coup = float(row[self._coupon_col]) / 100.0
            obs_price = float(row[self._price_col])

            issue_d = ql.Date(issue_d_dt.day, issue_d_dt.month, issue_d_dt.year)
            mat_d = ql.Date(mat_d_dt.day, mat_d_dt.month, mat_d_dt.year)

            # Check if maturity date is after settlement date
            if mat_d <= settlement_date:
                raise ValueError(f"Maturity date {mat_d} must be after settlement date {settlement_date}")

            schedule = ql.Schedule(
                issue_d,
                mat_d,
                ql.Period(self._ql_freq),
                self._ql_cal,
                self._ql_bdc,  # Accrual BDC
                self._ql_bdc,  # Payment BDC
                ql.DateGeneration.Backward,
                False,  # EndOfMonth
            )

            bond = ql.FixedRateBond(
                self._settlement_t_plus, 100.0, schedule, [coup], self._ql_day_count, self._ql_bdc, 100.0, issue_d  # Payment BDC for FixedRateBond
            )

            # Calculate Modified Duration - needs a dummy curve, rate doesn't matter for duration itself
            flat_rate = 0.03  # Arbitrary flat rate for duration calc
            ts_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, flat_rate, self._ql_day_count, self._ql_compounded, self._ql_freq))
            engine = ql.DiscountingBondEngine(ts_handle)
            bond.setPricingEngine(engine)

            # Use Macaulay duration to match GSW paper context (D = Macaulay)
            # Then calculate Modified Duration D_mod = D / (1 + y/f)
            # GSW uses y=0 in the duration call, which simplifies things but might not be standard QL practice
            # Let's calculate Macaulay first
            macaulay_dur = ql.BondFunctions.duration(bond, ql.InterestRate(0.0, self._ql_day_count, self._ql_compounded, self._ql_freq), ql.Duration.Macaulay)

            # If Macaulay duration is needed directly for weighting:
            mdur = macaulay_dur  # Use Macaulay duration directly for GSW weighting

        except Exception as e:
            self._logger.error(f"Error creating bond/calculating duration for row {row.get('cusip', 'N/A')}: {e}")
            return None, np.nan, np.nan  # Return indication of failure

        return bond, obs_price, mdur

    def _price_residuals(
        self, params: np.ndarray, bonds_data: List[Tuple[ql.Bond, float, float]], settlement_date: ql.Date, day_counter: ql.DayCounter
    ) -> np.ndarray:
        """
        Calculates weighted price residuals for GSW model.
        Residual = (Model Price - Observed Price) / Duration
        (Signs flipped compared to JPM to match standard residual definition for least_squares)
        """
        num_bonds = len(bonds_data)
        residuals = np.zeros(num_bonds)
        processed_count = 0

        # Unpack parameters (ensure order matches optimization vector)
        b0, b1, b2, b3, t1, t2 = params

        # Basic parameter sanity check (especially taus > 0)
        if t1 <= 1e-6 or t2 <= 1e-6:
            self._logger.debug("Tau parameter(s) non-positive, returning large penalty.")
            return np.full(num_bonds, 1e6)  # Penalize invalid tau parameters

        for i, (bond, obs_price, duration) in enumerate(bonds_data):
            # Use Macaulay duration for weight as per GSW paper context
            weight = 1.0 / duration if duration > 1e-6 else 0.0  # Avoid division by zero

            if weight == 0.0:
                residuals[i] = 0.0  # Assign zero residual if duration is invalid
                continue

            model_price = 0.0
            valid_bond = True
            try:
                for cf in bond.cashflows():
                    cf_date = cf.date()
                    if cf_date > settlement_date:
                        ttm = day_counter.yearFraction(settlement_date, cf_date)
                        df = self._nss_discount_factor(ttm, params)

                        if df is None or np.isnan(df) or df <= 0 or df > 1.0001:
                            # self._logger.warning(f"Invalid DF ({df}) for bond {i}, TTM {ttm}. Skipping bond.")
                            valid_bond = False
                            break
                        model_price += cf.amount() * df

                if valid_bond:
                    # Calculate residual: (Model - Actual) / Duration
                    # Note: least_squares minimizes sum of SQUARES of these
                    residuals[i] = (model_price - obs_price) * weight
                    processed_count += 1
                else:
                    residuals[i] = 0.0  # Assign zero residual if DF was invalid (handled by robust loss)

            except Exception as e:
                # self._logger.warning(f"Error pricing bond {i} in residual calculation: {e}")
                residuals[i] = 0.0  # Assign zero residual on error

        # Optional: check processed_count
        # if processed_count < num_bonds * 0.5: print("Low bond processing rate...")

        return residuals

    def fit(self):
        """Fits the GSW Nelson-Siegel-Svensson model."""
        self._logger.info("Starting GSW2006 fit process...")
        df = self._filter(self._cusip_set_df)
        self._filtered_cusip_set_df = df
        if df.empty:
            self._logger.error(f"Filtering resulted in empty DataFrame. {self._FILTER_TOO_STRONG}")
            raise ValueError(f"'GSW2006.fit': {self._FILTER_TOO_STRONG}")
        self._logger.info(f"Filtered DataFrame shape: {df.shape}")

        # Determine settlement date from filtered data
        try:
            curve_dt = pd.to_datetime(df[self._timestamp_col].max())
            # settlement_date = datetime_to_ql_date(curve_dt) # Assumes utility exists
            settlement_date = ql.Date(curve_dt.day, curve_dt.month, curve_dt.year)
            self._logger.info(f"Using settlement date: {settlement_date}")
        except Exception as e:
            self._logger.error(f"Could not determine settlement date from '{self._timestamp_col}': {e}")
            raise ValueError("Could not determine settlement date.")

        # Prepare bond data (bond object, observed price, duration)
        eval_date_before = ql.Settings.instance().evaluationDate
        ql.Settings.instance().evaluationDate = settlement_date
        bonds_data = []
        skipped_count = 0
        for idx, row in df.iterrows():
            bond, obs_p, mdur = self._make_bond_from_row(row, settlement_date)
            # Ensure bond creation was successful and duration is valid
            if bond is not None and mdur is not None and mdur > 1e-6:
                bonds_data.append((bond, obs_p, mdur))
            else:
                skipped_count += 1
        ql.Settings.instance().evaluationDate = eval_date_before

        if not bonds_data:
            self._logger.error("No valid bonds found after QuantLib object creation and duration check.")
            raise ValueError("'GSW2006.fit': No valid bonds to fit.")
        self._logger.info(f"Prepared {len(bonds_data)} bonds for fitting (skipped {skipped_count}).")

        # Define initial guess (p0) and bounds for NSS parameters [b0, b1, b2, b3, t1, t2]
        # These require careful consideration and might need adjustment based on data period
        # Example guesses/bounds:
        initial_beta0 = 0.03  # Approx long rate
        initial_beta1 = 0.02  # Approx (Short - Long)
        initial_beta2 = 0.0  # Curvature terms often start near zero
        initial_beta3 = 0.0
        initial_tau1 = 1.5  # Typical location for first hump (years)
        initial_tau2 = 7.0  # Typical location for second hump (years)
        p0 = [initial_beta0, initial_beta1, initial_beta2, initial_beta3, initial_tau1, initial_tau2]

        # Bounds: betas can be +/-; taus must be > 0
        # Set a small positive lower bound for taus to avoid division by zero
        lower_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, 1e-4, 1e-4]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]  # Upper bounds for taus could be added (e.g., 50)
        bounds = (lower_bounds, upper_bounds)

        # Wrapper for the residual function
        def fun_res(p):
            # Ensure QL date is set for duration calculation if needed inside _price_residuals implicitly
            current_eval_date = ql.Settings.instance().evaluationDate
            if current_eval_date != settlement_date:
                ql.Settings.instance().evaluationDate = settlement_date
            return self._price_residuals(p, bonds_data, settlement_date, self._ql_day_count)

        self._logger.info(f"Starting least_squares optimization with initial guess: {p0}")
        # Perform optimization
        res = least_squares(
            fun_res,
            p0,
            bounds=bounds,
            method="trf",  # Trust Region Reflective often good with bounds
            # loss='soft_l1', # Robust loss can still be helpful
            # f_scale=0.1,
            verbose=2 if self._debug_verbose else 0,  # Show optimizer steps if debugging
        )

        if not res.success:
            self._logger.error(f"Optimization failed: {res.message}")
            # Optional: raise error or just proceed with potentially bad params
            # raise RuntimeError(f"GSW curve fitting failed: {res.message}")

        self._logger.info(f"Optimization finished. Success: {res.success}, Status: {res.status}, Message: {res.message}")
        self._logger.info(f"Optimal Parameters: {res.x}")

        p_opt = res.x
        self._optimization_results = res
        self._fitted_parameters = p_opt  # Store the fitted NSS parameters

        # --- Create fitted curve representation ---
        # Define the final fitted discount function using optimal params
        def final_discount_func(ttm: float) -> float:
            # Add bounds check for robustness
            df_val = self._nss_discount_factor(ttm, p_opt)
            if np.isnan(df_val) or df_val <= 0 or df_val > 1.0001:
                self._logger.debug(f"Clamping DF={df_val} at TTM={ttm}")
                return max(1e-6, min(df_val, 1.0001))
            return df_val

        # Generate points for the fitted curve DataFrame
        # Determine max maturity from the data used in fit
        max_maturity_in_data = df[self._ttm_col].max()
        max_maturity_plot = max(max_maturity_in_data, self._MAX_MATURITIES_YEARS)  # Use max from data or default
        all_times = np.linspace(0.0, max_maturity_plot, self._MAX_LINSPACE)
        discounted = [(ttm, final_discount_func(ttm)) for ttm in all_times]
        curve_df = pd.DataFrame(discounted, columns=[self._TO_RETURN_TTM_COL, self._TO_RETURN_DF_COL])

        self._fitted_curve_df = curve_df
        # For GSW, the core "function" is defined by the parameters, but store lambda for consistency if needed
        self._scipy_spline_func = final_discount_func  # Store the lambda

        self._ql_discount_curve = build_piecewise_curve_from_discount_factor_scipy_spline(
            spline_func=final_discount_func,
            anchor_date=curve_dt,
            ql_calendar=self._ql_cal,
            ql_business_convention=self._ql_bdc,
            ql_day_counter=self._ql_day_count,
            ql_interpolation_algo="log_linear",
            enable_extrapolation=True,
        )

        # --- Final Checks ---
        if self._fitted_curve_df is not None:
            self._arb_free_results = check_arbitrage_free_discount_curve(
                self._fitted_curve_df[self._TO_RETURN_TTM_COL].to_numpy(), self._fitted_curve_df[self._TO_RETURN_DF_COL].to_numpy()
            )
            self._logger.info(f"Arbitrage check results: {self._arb_free_results}")
        else:
            self._arb_free_results = {"error": "Fitted curve DataFrame not generated"}
            self._logger.warning("Fitted curve DataFrame not generated, skipping arbitrage check.")

        self._logger.info("GSW2006 fit process complete.")
