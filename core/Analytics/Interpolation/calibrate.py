# adapted from https://github.com/luphord/nelson_siegel_svensson/blob/master/nelson_siegel_svensson/calibrate.py
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import OptimizeResult, minimize

from core.Analytics.Interpolation.BjorkChristensen import BjorkChristensenCurve
from core.Analytics.Interpolation.BjorkChristensenAugmented import BjorkChristensenAugmentedCurve
from core.Analytics.Interpolation.DieboldLi import DieboldLiCurve
from core.Analytics.Interpolation.MLESM import MerrillLynchExponentialSplineModel
from core.Analytics.Interpolation.NelsonSiegel import NelsonSiegelCurve
from core.Analytics.Interpolation.NelsonSiegelSvensson import NelsonSiegelSvenssonCurve
from core.Analytics.Interpolation.SmithWilson import SmithWilsonCurve, find_ufr_ytm


def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, "Mismatching shapes of time and values"


def betas_ns_ols(tau: float, t: np.ndarray, y: np.ndarray) -> Tuple[NelsonSiegelCurve, Any]:
    _assert_same_shape(t, y)
    curve = NelsonSiegelCurve(0, 0, 0, tau)
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return NelsonSiegelCurve(beta[0], beta[1], beta[2], tau), lstsq_res


def errorfn_ns_ols(tau: float, t: np.ndarray, y: np.ndarray) -> float:
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_ns_ols(tau, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_ns_ols(t: np.ndarray, y: np.ndarray, tau0: float = 2.0) -> Tuple[NelsonSiegelCurve, Any]:
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=tau0, args=(t, y))
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res


def empirical_factors(y_3m: float, y_2y: float, y_10y: float) -> Tuple[float, float, float]:
    return y_10y, y_10y - y_3m, 2 * y_2y - y_3m - y_10y


def betas_nss_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    _assert_same_shape(t, y)
    curve = NelsonSiegelSvenssonCurve(0, 0, 0, 0, tau[0], tau[1])
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return (
        NelsonSiegelSvenssonCurve(beta[0], beta[1], beta[2], beta[3], tau[0], tau[1]),
        lstsq_res,
    )


def errorfn_nss_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray) -> float:
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_nss_ols(tau, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_nss_ols(t: np.ndarray, y: np.ndarray, tau0: Tuple[float, float] = (2.0, 5.0)) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_nss_ols, x0=np.array(tau0), args=(t, y))
    curve, lstsq_res = betas_nss_ols(opt_res.x, t, y)
    return curve, opt_res, lstsq_res


def betas_nss_weighted_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    """
    Modified OLS fitting to include weights for the weighted Svensson model.
    """
    _assert_same_shape(t, y, weights)  # Ensure that all inputs have the same shape
    curve = NelsonSiegelSvenssonCurve(0, 0, 0, 0, tau[0], tau[1])
    factors = curve.factor_matrix(t)

    # Apply the weights to the factors and yields
    W = np.diag(weights)
    weighted_factors = W @ factors
    weighted_y = W @ y

    # Solve the weighted OLS problem
    lstsq_res = lstsq(weighted_factors, weighted_y, rcond=None)
    beta = lstsq_res[0]

    return (
        NelsonSiegelSvenssonCurve(beta[0], beta[1], beta[2], beta[3], tau[0], tau[1]),
        lstsq_res,
    )


def weighted_errorfn_nss_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """
    Error function for the weighted Svensson model to be used in optimization.
    """
    curve, lstsq_res = betas_nss_weighted_ols(tau, t, y, weights)
    return np.sum(weights * (curve(t) - y) ** 2)


def calibrate_nss_weighted_ols(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray, tau0: Tuple[float, float] = (2.0, 5.0)
) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    """
    Calibration function for the weighted Svensson model using OLS.
    """
    _assert_same_shape(t, y, weights)
    opt_res = minimize(weighted_errorfn_nss_ols, x0=np.array(tau0), args=(t, y, weights))
    curve, lstsq_res = betas_nss_weighted_ols(opt_res.x, t, y, weights)
    return curve, opt_res, lstsq_res


def errorfn_bc_ols(tau: float, t: np.ndarray, y: np.ndarray) -> float:
    curve, _ = betas_bc_ols(tau, t, y)
    estimated_yields = curve(t)
    return np.sum((estimated_yields - y) ** 2)


def betas_bc_ols(tau: float, t: np.ndarray, y: np.ndarray) -> Tuple[BjorkChristensenCurve, np.linalg.LinAlgError]:
    curve = BjorkChristensenCurve(0, 0, 0, 0, tau)
    F = curve.factor_matrix(t)
    betas, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
    curve.beta0, curve.beta1, curve.beta2, curve.beta3 = betas
    return curve, betas


def calibrate_bc_ols(t: np.ndarray, y: np.ndarray, tau0: float = 1.0) -> Tuple[BjorkChristensenCurve, Any]:
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_bc_ols, x0=np.array([tau0]), args=(t, y))
    curve, lstsq_res = betas_bc_ols(opt_res.x[0], t, y)
    return curve, opt_res


def calibrate_bc_augmented_ols(maturities: npt.NDArray[np.float64], yields: npt.NDArray[np.float64]) -> Tuple[BjorkChristensenAugmentedCurve, Any]:

    def objective(params: npt.NDArray[np.float64]) -> float:
        beta0, beta1, beta2, beta3, beta4, tau = params
        curve = BjorkChristensenAugmentedCurve(beta0, beta1, beta2, beta3, beta4, tau)
        model_yields = curve(np.array(maturities))
        return np.sum((model_yields - np.array(yields)) ** 2)

    initial_params = [0.01, 0.01, 0.01, 0.01, 0.01, 1.0]
    bounds = [
        (0, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (0, None),
    ]

    result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds)
    fitted_params = result.x
    curve = BjorkChristensenAugmentedCurve(*fitted_params)
    return curve, result


def calibrate_diebold_li_ols(maturities: npt.NDArray[np.float64], yields: npt.NDArray[np.float64]) -> Tuple[DieboldLiCurve, Any]:
    initial_guess = [np.mean(yields), -0.02, 0.02, 0.1]

    def objective(params: npt.NDArray[np.float64]) -> float:
        curve = DieboldLiCurve(params[0], params[1], params[2], params[3])
        return np.sum((curve(np.array(maturities)) - np.array(yields)) ** 2)

    result = minimize(objective, initial_guess, method="BFGS")

    optimized_params = result.x
    curve = DieboldLiCurve(*optimized_params)
    return curve, result


def calibrate_mles_ols(
    maturities: npt.NDArray[np.float64],
    yields: npt.NDArray[np.float64],
    N: int = 8,
    regularization: float = 1e-4,
    overnight_rate: Optional[float] = None,
) -> Tuple[MerrillLynchExponentialSplineModel, Any]:
    if maturities[0] != 0:
        short_rate = overnight_rate or yields[0]
        maturities = np.insert(maturities, 0, 1 / 365)
        yields = np.insert(yields, 0, short_rate)

    """Fit the MLES model to the given yields using OLS."""
    initial_guess = [0.1] + [1.0] * N

    def objective(params: npt.NDArray[np.float64]) -> float:
        alpha = params[0]
        lambda_hat = np.array(params[1:])
        curve = MerrillLynchExponentialSplineModel(alpha, N, lambda_hat)
        curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))
        theoretical_yields = curve.theoretical_yields(np.array(maturities))
        regularization_term = regularization * np.sum(np.diff(lambda_hat) ** 2)

        return np.sum((theoretical_yields - np.array(yields)) ** 2) + regularization_term

    result = minimize(objective, initial_guess, method="BFGS")
    optimized_params = result.x
    optimized_alpha = optimized_params[0]
    optimized_lambda_hat = np.array(optimized_params[1:])
    curve = MerrillLynchExponentialSplineModel(optimized_alpha, N, optimized_lambda_hat)
    curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))

    return curve, result


def calibrate_smith_wilson_ols(
    maturities: npt.NDArray[np.float64],
    yields: npt.NDArray[np.float64],
    ufr: Optional[float] = None,
    alpha_initial: float = 0.1,
    overnight_rate: Optional[float] = None,
) -> Tuple[MerrillLynchExponentialSplineModel, Any]:
    if maturities[0] != 0:
        short_rate = overnight_rate or yields[0]
        maturities = np.insert(maturities, 0, 1 / 365)
        yields = np.insert(yields, 0, short_rate)

    if not ufr:
        ufr = find_ufr_ytm(maturities=maturities, ytms=yields)

    def objective(alpha: float) -> float:
        curve = SmithWilsonCurve(ufr, alpha)
        curve.fit(yields, maturities)
        fitted_yields = curve(maturities)
        return np.sum((fitted_yields - yields) ** 2)

    result = minimize(objective, x0=alpha_initial, bounds=[(0.01, 1.0)], method="L-BFGS-B")

    optimal_alpha = result.x[0]
    calibrated_curve = SmithWilsonCurve(ufr, optimal_alpha)
    calibrated_curve.fit(yields, maturities)

    return calibrated_curve, result


# def calibrate_pca_yield_curve(
#     ytms: npt.NDArray[np.float64], historical_df: pd.DataFrame, n_components: int = 3, use_changes: bool = False
# ) -> Tuple[PCACurve, Any]:
#     if use_changes:
#         historical_df = historical_df.diff().dropna()
#     pca_model = PCACurve(n_components=n_components)
#     pca_model.fit(ytms)
#     return pca_model, pca_model.explained_variance
