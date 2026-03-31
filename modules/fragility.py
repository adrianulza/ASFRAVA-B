from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_EPS = 1.0e-6


# ==================================================================
# Utility helpers
# ==================================================================

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


def _group_stripes(edps, im_col, damage_states):
    """Group EDP rows by IM stripe and return raw counts.

    Returns
    -------
    x_all : ndarray
        All IM stripe levels (including near-zero anchor if present).
    n_total_all : ndarray
        Total record count at each stripe.
    n_exceeded_all : dict[str, ndarray]
        Exceedance count per damage state at each stripe.
    """
    grouped = (
        edps.assign(_IM=edps[im_col].round(8))
        .groupby("_IM", as_index=False)
        .agg(total=("GMR", "size"), **{ds: (ds, "sum") for ds in damage_states})
        .sort_values("_IM")
    )
    x_all          = grouped["_IM"].to_numpy(dtype=float)
    n_total_all    = grouped["total"].to_numpy(dtype=float)
    n_exceeded_all = {ds: grouped[ds].to_numpy(dtype=float) for ds in damage_states}
    return x_all, n_total_all, n_exceeded_all


# ==================================================================
# MSA-MLE  (Baker 2015, Eq. 11) — independent per damage state
# ==================================================================

def neg_loglik(theta, x, n_total, n_exceeded, epsilon=1e-10):
    """Negative binomial log-likelihood for one lognormal fragility curve.

    Implements Baker (2015) Eq. 11:
        ll = sum_j [ z_j * ln(p_j) + (n_j - z_j) * ln(1 - p_j) ]

    Parameters
    ----------
    theta : array-like, shape (2,)
        [mu, log_sigma] where mu = ln(median capacity),
        sigma = exp(log_sigma) = log-standard deviation (beta).
    x : ndarray
        IM level at each stripe j (must be positive).
    n_total : ndarray
        Total ground motions at each stripe j  (n_j).
    n_exceeded : ndarray
        Exceedances at each stripe j  (z_j).
    epsilon : float
        Numerical clip to keep log arguments away from 0 and 1.

    Notes
    -----
    Using proportions y = z_j / n_j instead of raw counts divides
    every term by n_j, giving equal weight to all stripes regardless
    of record count.  Baker (2015) identifies this as a biased
    alternative (his Eq. 12) and recommends against it.
    """
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    prob  = norm.cdf((_safe_log(x) - mu) / sigma)
    prob  = np.clip(prob, epsilon, 1.0 - epsilon)
    ll    = n_exceeded * np.log(prob) + (n_total - n_exceeded) * np.log(1.0 - prob)
    return -np.sum(ll)


def calculate_fragility_curve(x, n_total, n_exceeded, IM_range):
    """Fit one lognormal fragility curve via MLE (Baker 2015, Eq. 11).

    Parameters
    ----------
    x : ndarray
        IM level at each stripe (positive).
    n_total : ndarray
        Total record count per stripe  (n_j).
    n_exceeded : ndarray
        Exceedance count per stripe  (z_j).
    IM_range : ndarray
        IM values at which to evaluate the fitted curve.

    Returns
    -------
    model : dict
        Keys: ``mu`` (ln-median), ``sigma`` (log-std / beta),
        ``median`` (median in IM units).
    probabilities : ndarray
    """
    x          = np.asarray(x,          dtype=float)
    n_total    = np.asarray(n_total,    dtype=float)
    n_exceeded = np.asarray(n_exceeded, dtype=float)
    x_safe     = np.clip(x, _EPS, None)

    theta_start = np.array([np.mean(np.log(x_safe)), np.log(0.6)], dtype=float)

    res = minimize(
        neg_loglik,
        theta_start,
        args=(x_safe, n_total, n_exceeded),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10_000},
    )

    mu_hat, log_sigma_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    model = {
        "mu":     float(mu_hat),
        "sigma":  sigma_hat,
        "median": float(np.exp(mu_hat)),
    }
    probabilities = norm.cdf((_safe_log(IM_range) - mu_hat) / sigma_hat)
    return model, probabilities


def fit_MLE_MSA(edps, damage_states, selected_ims, gmrs_folderpath,
                min_scale, max_scale, step_size, increment):
    """Fit fragility curves using MSA-MLE independently per damage state.

    Groups EDP results by IM stripe, counts exceedances, and maximises
    the binomial log-likelihood (Baker 2015, Eq. 11) separately for each
    damage state.  Because dispersion (beta) is estimated independently
    per damage state, crossing curves are possible.  Use fit_JMLE_MSA
    to enforce a shared dispersion and guarantee non-crossing.

    Notes
    -----
    The near-zero anchor point (IM ~ 0) is excluded from the MLE fit
    because log(IM ~ 0) is numerically extreme.  It is retained in
    scatter_data for the EDP plot.
    """
    im_col = _resolve_im_column(edps, selected_ims)
    x_all, n_total_all, n_exceeded_all = _group_stripes(edps, im_col, damage_states)

    fit_mask   = x_all > _EPS * 1e3
    x_fit      = np.clip(x_all[fit_mask], _EPS, None)
    x_curve    = np.linspace(min_scale, max_scale, step_size)

    probabilities = {}
    scatter_data  = {ds: {"x": [], "y": []} for ds in damage_states}
    models        = {}

    for ds in damage_states:
        n_tot_fit = n_total_all[fit_mask]
        n_exc_fit = n_exceeded_all[ds][fit_mask]

        model, prob = calculate_fragility_curve(x_fit, n_tot_fit, n_exc_fit, x_curve)
        models[ds]        = model
        probabilities[ds] = prob

        scatter_data[ds]["x"] = x_all
        scatter_data[ds]["y"] = (n_exceeded_all[ds] / n_total_all).clip(0.0, 1.0)

    return models, probabilities, scatter_data


# ==================================================================
# J-MLE — Joint MLE with common dispersion
# ==================================================================

def _neg_loglik_joint(params, x, n_total, n_exceeded_list, epsilon=1e-10):
    """Negative joint log-likelihood for J-MLE.

    Parameters
    ----------
    params : ndarray, shape (K+1,)
        [mu_1, delta_1, ..., delta_{K-1}, log_sigma]
        mu_{k+1} = mu_k + exp(delta_k) enforces strict ordering.
    x : ndarray
        IM levels at fitting stripes (positive, anchor excluded).
    n_total : ndarray
        Total record count per stripe  (n_j).
    n_exceeded_list : list[ndarray]
        Exceedance counts per stripe for each damage state, in order.
    epsilon : float
        Numerical clip bound.
    """
    K         = len(n_exceeded_list)
    sigma     = np.exp(params[-1])

    mu_1 = params[0]
    mus  = [mu_1]
    for k in range(1, K):
        mus.append(mus[-1] + np.exp(params[k]))

    total_ll = 0.0
    for k in range(K):
        prob = norm.cdf((_safe_log(x) - mus[k]) / sigma)
        prob = np.clip(prob, epsilon, 1.0 - epsilon)
        n_exc = n_exceeded_list[k]
        total_ll += np.sum(
            n_exc * np.log(prob) + (n_total - n_exc) * np.log(1.0 - prob)
        )

    return -total_ll


def fit_JMLE_MSA(edps, damage_states, selected_ims, gmrs_folderpath,
                 min_scale, max_scale, step_size, increment):
    """Fit fragility curves using Joint MLE with common dispersion (J-MLE).

    Solves one joint optimisation over
        (mu_1, delta_1, ..., delta_{K-1}, log_sigma)
    where:
        mu_{k+1} = mu_k + exp(delta_k)  =>  mu_1 < mu_2 < ... < mu_K
        sigma     = exp(log_sigma)       =>  shared across all K damage states

    Because all K fragility curves share the same sigma and have strictly
    ordered medians, they are parallel lognormal CDFs that cannot intersect
    at any IM level.

    The log-spacing reparameterisation converts the constrained ordering
    problem into an unconstrained one, allowing Nelder-Mead to be used
    directly without a constrained solver (Chen 2024, arXiv:2405.15513).

    Parameters
    ----------
    edps : DataFrame
        EDP dataset as produced by any ASFRAVA-B analysis module.
    damage_states : list[str]
        Damage state column names in ascending severity order,
        e.g. ["ds1", "ds2", "ds3"].
    selected_ims : str
        Name of the IM column in edps.
    gmrs_folderpath : str
        Path to ground-motion records folder (not used directly here
        but kept for API consistency with fit_MLE_MSA).
    min_scale, max_scale : float
        IM range for fragility curve output.
    step_size : int
        Number of points in the output IM range.
    increment : float
        IM increment (kept for API consistency).

    Returns
    -------
    models : dict[str, dict]
        Per damage state dictionary with keys:
            ``mu``          - ln(median capacity)
            ``sigma``       - log-standard deviation = beta_common
            ``median``      - median capacity in IM units
            ``beta_common`` - explicit label; identical to sigma
        beta_common is the same value for every damage state.
    probabilities : dict[str, ndarray]
        Fragility curves evaluated at IM_range.  Guaranteed non-crossing.
    scatter_data : dict[str, dict]
        Observed exceedance fractions per stripe for plotting.
    """
    im_col = _resolve_im_column(edps, selected_ims)
    x_all, n_total_all, n_exceeded_all = _group_stripes(edps, im_col, damage_states)

    fit_mask = x_all > _EPS * 1e3
    x_fit    = np.clip(x_all[fit_mask], _EPS, None)
    n_tot    = n_total_all[fit_mask]

    K              = len(damage_states)
    n_exceeded_fit = [n_exceeded_all[ds][fit_mask] for ds in damage_states]
    x_curve        = np.linspace(min_scale, max_scale, step_size)

    ind_mus    = []
    ind_sigmas = []

    for k in range(K):
        theta0 = np.array([np.mean(np.log(x_fit)), np.log(0.6)], dtype=float)
        r = minimize(
            neg_loglik,
            theta0,
            args=(x_fit, n_tot, n_exceeded_fit[k]),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5_000},
        )
        ind_mus.append(float(r.x[0]))
        ind_sigmas.append(float(np.exp(r.x[1])))

    # Sort individual mu estimates; protect against reversed/equal values
    ind_mus_sorted = sorted(ind_mus)
    log_sigma_init = np.log(float(np.mean(ind_sigmas)))

    # Build initial params: [mu_1, delta_1, ..., delta_{K-1}, log_sigma]
    params_init = [ind_mus_sorted[0]]
    for k in range(K - 1):
        spacing = ind_mus_sorted[k + 1] - ind_mus_sorted[k]
        spacing = max(spacing, 1e-4)
        params_init.append(np.log(spacing))
    params_init.append(log_sigma_init)
    params_init = np.array(params_init, dtype=float)

    # ------------------------------------------------------------------
    # Joint optimisation
    # ------------------------------------------------------------------
    res = minimize(
        _neg_loglik_joint,
        params_init,
        args=(x_fit, n_tot, n_exceeded_fit),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 50_000},
    )

    # ------------------------------------------------------------------
    # Recover final parameters
    # ------------------------------------------------------------------
    opt         = res.x
    beta_common = float(np.exp(opt[-1]))

    mus_hat = [float(opt[0])]
    for k in range(1, K):
        mus_hat.append(mus_hat[-1] + float(np.exp(opt[k])))

    # ------------------------------------------------------------------
    # Assemble output dictionaries
    # ------------------------------------------------------------------
    models        = {}
    probabilities = {}
    scatter_data  = {ds: {"x": [], "y": []} for ds in damage_states}

    for k, ds in enumerate(damage_states):
        mu_k = mus_hat[k]
        models[ds] = {
            "mu":          mu_k,
            "sigma":       beta_common,     
            "median":      float(np.exp(mu_k)),
            "beta_common": beta_common,     
        }
        probabilities[ds] = norm.cdf((_safe_log(x_curve) - mu_k) / beta_common)

        scatter_data[ds]["x"] = x_all
        scatter_data[ds]["y"] = (n_exceeded_all[ds] / n_total_all).clip(0.0, 1.0)

    return models, probabilities, scatter_data


# ==================================================================
# GLM  (individual binary records)
# ==================================================================

def fit_glm_model(edps, damage_states, link_function, selected_ims,
                  min_scale, max_scale, step_size):
    """Fit fragility curves using a Generalised Linear Model (GLM).

    Uses individual record binary outcomes with a Binomial family and
    a Probit or Logit link, fitted independently per damage state.
    """
    probabilities = {}
    models        = {}
    im_col        = _resolve_im_column(edps, selected_ims)
    IM_range      = np.linspace(min_scale, max_scale, step_size)
    log_IM_range  = _safe_log(IM_range)

    link_cls = (
        sm.genmod.families.links.Logit
        if link_function == "Logit"
        else sm.genmod.families.links.Probit
    )

    for ds in damage_states:
        log_selected_ims = _safe_log(edps[im_col])
        model = sm.GLM(
            edps[ds],
            sm.add_constant(log_selected_ims),
            family=sm.families.Binomial(link=link_cls()),
        ).fit()
        models[ds] = model

        predict_df        = sm.add_constant(log_IM_range)
        probabilities[ds] = model.predict(predict_df)

    return models, IM_range, probabilities


# ==================================================================
# Logistic Regression with L2 regularisation  (sklearn)
# ==================================================================

def fit_logistic_regression(edps, damage_states, selected_ims, regulation,
                             min_scale, max_scale, step_size):
    """Fit fragility curves using regularised logistic regression (sklearn).

    Regularisation strength via ``regulation``:
        "High Regulation"   -> C = 1     (strong regularisation)
        "Medium Regulation" -> C = 10
        "No Regulation"     -> C = 1e5   (effectively unregularised)

    Features are standardised before fitting; the same scaler is applied
    consistently to the prediction grid.
    """
    probabilities  = {}
    models         = {}
    im_col         = _resolve_im_column(edps, selected_ims)
    IM_range       = np.linspace(min_scale, max_scale, step_size)

    C_values = {
        "High Regulation":   1,
        "Medium Regulation": 10,
        "No Regulation":     1e5,
    }
    C_value = C_values.get(regulation, 1e5)

    X_raw      = _safe_log(edps[im_col].values).reshape(-1, 1)
    X_pred_raw = _safe_log(IM_range).reshape(-1, 1)

    scaler        = StandardScaler()
    X_scaled      = scaler.fit_transform(X_raw)
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
        models[ds]        = model

    return models, IM_range, probabilities


# ==================================================================
# Public dispatcher
# ==================================================================

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
    """Dispatch to the appropriate fragility fitting method.

    Parameters
    ----------
    edps : DataFrame
        EDP dataset with IM column, binary damage-state indicator
        columns, GMR filename column, and Status column.
    damage_states : list[str]
        Column names for binary damage state indicators in ascending
        severity order, e.g. ["ds1", "ds2", "ds3"].
    regression_selection : {"MSA", "J-MLE", "GLM", "LogregML"}
        Fitting method:
        - "MSA"      : Independent MSA-MLE per damage state
                       (Baker 2015).  May produce crossing curves.
        - "J-MLE"    : Joint MLE with common dispersion.
                       Guaranteed non-crossing (Porter et al. 2007;
                       Lallemant et al. 2015; Nguyen & Lallemant 2022).
        - "GLM"      : GLM on individual binary records.
        - "LogregML" : L2-regularised logistic regression (sklearn).
    selected_ims : str
        Name of the IM column in edps.
    link_function_selection : {"Probit", "Logit"}
        Link function for GLM only; ignored for other methods.
    regulation : str
        Regularisation level for LogregML only; ignored for others.
    gmrs_folderpath : str
        Path to ground-motion records folder (used by MSA methods).
    min_scale, max_scale : float
        IM range for the fragility curve output.
    step_size : int
        Number of points in the output IM range.
    increment : float
        IM increment used during analysis (used by MSA stripe grouping).

    Returns
    -------
    models : dict
        Fitted model objects / parameter dicts keyed by damage state.
    IM_range : ndarray
        IM values at which fragility curves are evaluated.
    probabilities : dict
        Fragility curve arrays keyed by damage state.
    scatter_data : dict or None
        Observed exceedance fractions per stripe for MSA/J-MLE plotting;
        None for GLM and LogregML.
    """
    if regression_selection == "MSA":
        models, probabilities, scatter_data = fit_MLE_MSA(
            edps, damage_states, selected_ims,
            gmrs_folderpath, min_scale, max_scale, step_size, increment,
        )
        IM_range = np.linspace(min_scale, max_scale, step_size)
        return models, IM_range, probabilities, scatter_data

    if regression_selection == "J-MLE":
        models, probabilities, scatter_data = fit_JMLE_MSA(
            edps, damage_states, selected_ims,
            gmrs_folderpath, min_scale, max_scale, step_size, increment,
        )
        IM_range = np.linspace(min_scale, max_scale, step_size)
        return models, IM_range, probabilities, scatter_data

    if regression_selection == "GLM":
        models, IM_range, probabilities = fit_glm_model(
            edps, damage_states, link_function_selection,
            selected_ims, min_scale, max_scale, step_size,
        )
        return models, IM_range, probabilities, None

    if regression_selection == "LogregML":
        models, IM_range, probabilities = fit_logistic_regression(
            edps, damage_states, selected_ims,
            regulation, min_scale, max_scale, step_size,
        )
        return models, IM_range, probabilities, None

    raise ValueError(f"Invalid regression selection '{regression_selection}'")
