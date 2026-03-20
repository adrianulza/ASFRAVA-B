from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_EPS = 1.0e-6


def _resolve_im_column(edps, selected_ims: str) -> str:
    if selected_ims in edps.columns:
        return selected_ims
    if selected_ims == "PGA" and "PGA" in edps.columns:
        return "PGA"
    raise ValueError(
        f"Selected IM column '{selected_ims}' was not found in the EDP dataset. "
        f"Available columns: {', '.join(edps.columns)}"
    )


def _safe_log(values):
    arr = np.asarray(values, dtype=float)
    return np.log(np.clip(arr, _EPS, None))


def fit_fragility_models(
    edps,
    damage_states,
    regression_selection,
    selected_ims,
    link_function_selection,
    regulation,
    gmrs_folderpath,
    min_scale,
    max_scale,
    step_size,
    increment,
):
    if regression_selection == "MSA":
        models, probabilities, scatter_data = fit_MLE_MSA(
            edps, damage_states, selected_ims, gmrs_folderpath, min_scale, max_scale, step_size, increment
        )
        IM_range = np.linspace(min_scale, max_scale, step_size)
        return models, IM_range, probabilities, scatter_data

    if regression_selection == "GLM":
        models, IM_range, probabilities = fit_glm_model(
            edps, damage_states, link_function_selection, selected_ims, min_scale, max_scale, step_size
        )
        return models, IM_range, probabilities, None

    if regression_selection == "LogregML":
        models, IM_range, probabilities = fit_logistic_regression(
            edps, damage_states, selected_ims, regulation, min_scale, max_scale, step_size
        )
        return models, IM_range, probabilities, None

    raise ValueError(f"Invalid regression selection '{regression_selection}'")


def fit_MLE_MSA(edps, damage_states, selected_ims, gmrs_folderpath, min_scale, max_scale, step_size, increment):
    im_col = _resolve_im_column(edps, selected_ims)
    grouped = (
        edps.assign(_IM=edps[im_col].round(8))
        .groupby("_IM", as_index=False)
        .agg(total=("GMR", "size"), **{ds: (ds, "sum") for ds in damage_states})
        .sort_values("_IM")
    )

    x_curve = np.linspace(min_scale, max_scale, step_size)
    probabilities = {}
    scatter_data = {ds: {"x": [], "y": []} for ds in damage_states}
    models = {}

    x_fit = grouped["_IM"].to_numpy(dtype=float)
    x_fit_safe = np.clip(x_fit, _EPS, None)

    for ds in damage_states:
        y = (grouped[ds].to_numpy(dtype=float) / grouped["total"].to_numpy(dtype=float)).clip(0.0, 1.0)
        model, prob = calculate_fragility_curve(x_fit_safe, y, x_curve)
        models[ds] = model
        probabilities[ds] = prob
        scatter_data[ds]["x"] = x_fit
        scatter_data[ds]["y"] = y

    return models, probabilities, scatter_data


def neg_loglik(theta, x, y, epsilon=1e-10):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    prob = norm.cdf((_safe_log(x) - mu) / sigma)
    prob = np.clip(prob, epsilon, 1.0 - epsilon)
    ll = y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)
    return -np.sum(ll)


def calculate_fragility_curve(x, y, IM_range):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_safe = np.clip(x, _EPS, None)
    theta_start = np.array([np.mean(np.log(x_safe)), np.log(0.6)], dtype=float)

    res = minimize(neg_loglik, theta_start, args=(x_safe, y), method="Nelder-Mead")
    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    model = {"mu": float(mu_hat), "sigma": sigma_hat}
    probabilities = norm.cdf((_safe_log(IM_range) - mu_hat) / sigma_hat)
    return model, probabilities


def fit_glm_model(edps, damage_states, link_function, selected_ims, min_scale, max_scale, step_size):
    probabilities = {}
    models = {}
    im_col = _resolve_im_column(edps, selected_ims)
    IM_range = np.linspace(min_scale, max_scale, step_size)
    log_IM_range = _safe_log(IM_range)

    link = sm.genmod.families.links.Logit if link_function == "Logit" else sm.genmod.families.links.Probit

    for ds in damage_states:
        log_selected_ims = _safe_log(edps[im_col])
        model = sm.GLM(
            edps[ds],
            sm.add_constant(log_selected_ims),
            family=sm.families.Binomial(link=link()),
        ).fit()
        models[ds] = model
        intercept, coef = model.params
        if link_function == "Probit":
            probabilities[ds] = norm.cdf(intercept + coef * log_IM_range)
        else:
            probabilities[ds] = 1.0 / (1.0 + np.exp(-(intercept + coef * log_IM_range)))

    return models, IM_range, probabilities


def fit_logistic_regression(edps, damage_states, selected_ims, regulation, min_scale, max_scale, step_size):
    probabilities = {}
    models = {}
    im_col = _resolve_im_column(edps, selected_ims)
    IM_range = np.linspace(min_scale, max_scale, step_size)

    C_values = {"High Regulation": 1, "Medium Regulation": 10, "No Regulation": 1e5}
    C_value = C_values.get(regulation, 1e5)

    X_raw = _safe_log(edps[im_col].values).reshape(-1, 1)
    X_pred_raw = _safe_log(IM_range).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_pred_scaled = scaler.transform(X_pred_raw)

    for ds in damage_states:
        y = edps[ds]
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            random_state=100,
            max_iter=1000,
            C=C_value,
        )
        model.fit(X_scaled, y)
        probabilities[ds] = model.predict_proba(X_pred_scaled)[:, 1]
        models[ds] = model

    return models, IM_range, probabilities
