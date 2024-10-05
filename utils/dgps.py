import numpy as np
import pandas as pd
from utils.db import create_duckdb_database

def generate_treatment_effect(effect_type, T, T0, max_effect=1):
    if effect_type == "constant":
        return np.concatenate([np.zeros(T0), np.full(T - T0, max_effect)])
    elif effect_type == "linear":
        return np.concatenate([np.zeros(T0), np.linspace(0, max_effect, T - T0)])
    elif effect_type == "concave":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * 0.5 * np.log(2 * np.arange(1, T - T0 + 1) / (T - T0) + 1),
            ]
        )
    elif effect_type == "positive_then_negative":
        half_point = (T - T0) // 2
        return np.concatenate(
            [
                np.zeros(T0),
                np.linspace(0, max_effect, half_point),
                np.linspace(max_effect, -max_effect, T - T0 - half_point),
            ]
        )
    elif effect_type == "exponential":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * (1 - np.exp(-np.linspace(0, 5, T - T0))),
            ]
        )
    elif effect_type == "sinusoidal":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * np.sin(np.linspace(0, 2 * np.pi, T - T0)),
            ]
        )
    elif effect_type == "random_walk":
        return np.concatenate(
            [
                np.zeros(T0),
                max_effect * np.cumsum(np.random.randn(T - T0)),
            ]
        )
    else:
        raise ValueError("Unknown effect type")


def sim_panel_advanced(
    base_effect,
    N=1_000_000,
    T=35,
    T0=15,
    sigma_list=[5, 2, 0.01, 2],
    hetfx=False,
    num_treated=None,
    rho=0.7,
    seed=42,
    debug=False,
):
    np.random.seed(seed)
    sigma_unit, sigma_time, sigma_tt, sigma_e = sigma_list
    # Generate data
    unit_ids = np.repeat(np.arange(N), T)
    time_ids = np.tile(np.arange(T), N)
    # Generate unit-specific intercepts and time trends
    unit_fe = np.random.normal(0, sigma_unit, N)
    time_fe = np.random.normal(0, sigma_time, T)
    unit_tt = np.random.normal(0, sigma_tt, N)
    # Generate treatment indicator
    if num_treated is None:
        W = np.random.binomial(1, 0.5, N)
    else:
        treated_units = np.random.choice(N, num_treated, replace=False)
        W = np.zeros(N)
        W[treated_units] = 1
    W = np.repeat(W, T)
    W = W * (time_ids >= T0)
    # Generate treatment effect
    if hetfx:
        unit_effects = np.random.uniform(0.5, 1.5, N)
    else:
        unit_effects = np.ones(N)
    treatment_effect = np.outer(unit_effects, base_effect)
    # Generate serially correlated residuals
    residuals = np.zeros((N, T))
    residuals[:, 0] = np.random.normal(0, sigma_e, N)
    epsilon = np.random.normal(0, 1, (N, T - 1))
    factor = sigma_e * np.sqrt(1 - rho**2)
    for t in range(1, T):
        residuals[:, t] = rho * residuals[:, t - 1] + factor * epsilon[:, t - 1]
    # Generate outcome
    Y = (
        np.repeat(unit_fe, T)
        + np.repeat(unit_tt, T) * time_ids
        + treatment_effect.flatten() * W
        + np.tile(time_fe, N)
        + residuals.flatten()
    )

    # Create DataFrame
    df = pd.DataFrame({"unit": unit_ids, "time": time_ids, "Y": Y, "W": W})
    if debug:
        return Y, W, treatment_effect, df
    return df

def generate_benchmark_data(N, T, T0): 

    effect_type = "concave"
    max_effect = 1
    effect_vector = generate_treatment_effect(effect_type, T, T0, max_effect)

    # simulate data
    df = sim_panel_advanced(base_effect = effect_vector, N = N, T = T, T0 = T0)
    # store data in duckdb database
    create_duckdb_database(df = df, db_name = "benchmarks.db")

    return df