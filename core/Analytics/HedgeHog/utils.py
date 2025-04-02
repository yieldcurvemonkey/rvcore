import numpy as np


def linear_solve_for_risk_weighted_notionals(risk_weights: np.ndarray, bpvs: np.ndarray, constrained_leg_index: int, constrained_leg_notional: float):
    risk_weights = np.array(risk_weights, dtype=float)
    bpvs = np.array(bpvs, dtype=float)
    R = constrained_leg_notional * bpvs[constrained_leg_index] / risk_weights[constrained_leg_index]
    notionals = (risk_weights * R) / bpvs

    return notionals
