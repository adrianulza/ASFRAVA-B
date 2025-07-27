import os

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
    print(f"Running {regression_selection} regression with selected IM: {selected_ims}")

    if regression_selection == "MSA":
        models, probabilities, scatter_data = fit_MLE_MSA(
            edps, damage_states, selected_ims, gmrs_folderpath, min_scale, max_scale, step_size, increment
        )
        IM_range = np.linspace(min_scale, max_scale, step_size)
        print(f"IM_range: {IM_range}")
        return models, IM_range, probabilities, scatter_data

    elif regression_selection == "GLM":
        models, IM_range, probabilities = fit_glm_model(
            edps, damage_states, link_function_selection, selected_ims, min_scale, max_scale, step_size
        )
        print(f"IM_range: {IM_range}")
        print(f"Probabilities: {probabilities}")
        return models, IM_range, probabilities, None

    elif regression_selection == "LogregML":
        models, IM_range, probabilities = fit_logistic_regression(
            edps, damage_states, selected_ims, regulation, min_scale, max_scale, step_size
        )
        print(f"IM_range: {IM_range}")
        print(f"Probabilities: {probabilities}")
        return models, IM_range, probabilities, None

    else:
        raise ValueError(f"Invalid regression selection '{regression_selection}'")


def fit_MLE_MSA(edps, damage_states, selected_ims, gmrs_folderpath, min_scale, max_scale, step_size, increment):

    IM_range = np.linspace(min_scale, max_scale, step_size)
    x = np.arange(min_scale, max_scale + increment, increment)

    gmrs_list = os.listdir(gmrs_folderpath)
    probabilities = {}
    scatter_data = {ds: {"x": [], "y": []} for ds in damage_states}
    models = {}

    # Group the data by PGA and sum the damage states
    collapse_counts = edps.groupby("PGA")[damage_states].sum().reset_index()

    for ds in damage_states:
        # y_values = np.array(edps[ds])
        num_gmrs = np.full_like(x, len(gmrs_list))

        # Interpolate the num_collapse values to match the x range
        num_collapse = np.interp(x, collapse_counts["PGA"], collapse_counts[ds])

        y = num_collapse / num_gmrs
        model, prob = calculate_fragility_curve(x, y, IM_range)
        probabilities[ds] = prob

        # Prepare scatter data
        scatter_data[ds]["x"] = x
        scatter_data[ds]["y"] = y

    return models, probabilities, scatter_data


def neg_loglik(theta, x, y, epsilon=1e-10):
    mu, sigma = theta
    prob = norm.cdf((np.log(x) - mu) / sigma)
    prob = np.clip(prob, epsilon, 1 - epsilon)  # Avoid log(0) by clipping
    ll = y * np.log(prob) + (1 - y) * np.log(1 - prob)
    return -np.sum(ll)


def calculate_fragility_curve(x, y, IM_range):
    theta_start = np.array([0, 1])
    res = minimize(neg_loglik, theta_start, args=(x, y), method="Nelder-Mead")
    mu_hat, sigma_hat = res.x
    model = {"mu": mu_hat, "sigma": sigma_hat}
    probabilities = norm.cdf((np.log(IM_range) - mu_hat) / sigma_hat)
    return model, probabilities


def fit_glm_model(edps, damage_states, link_function, selected_ims, min_scale, max_scale, step_size):
    probabilities = {}
    models = {}
    IM_range = np.linspace(min_scale, max_scale, step_size)
    log_IM_range = np.log(IM_range + 1e-6)

    link = sm.genmod.families.links.Logit if link_function == "Logit" else sm.genmod.families.links.Probit

    for ds in damage_states:
        log_selected_ims = np.log(edps[selected_ims] + 1e-6)

        model = sm.GLM(edps[ds], sm.add_constant(log_selected_ims), family=sm.families.Binomial(link=link())).fit()
        models[ds] = model

        intercept, coef = model.params

        # Compare Probit vs Logit
        print(f"Damage State: {ds}, Link Function: {link_function}, Intercept: {intercept}, Coef: {coef}")

        if link_function == "Probit":
            probabilities[ds] = norm.cdf(intercept + coef * log_IM_range)
        else:
            probabilities[ds] = 1 / (1 + np.exp(-(intercept + coef * log_IM_range)))

    return models, IM_range, probabilities


def fit_logistic_regression(edps, damage_states, selected_ims, regulation, min_scale, max_scale, step_size):
    probabilities = {}
    models = {}

    # Separate scalers for different feature sets
    scaler_IM = StandardScaler()
    scaler_X = StandardScaler()

    # Define intensity measure range
    IM_range = np.linspace(min_scale, max_scale, step_size)
    log_IM_range = np.log(IM_range + 1e-6).reshape(-1, 1)
    IM_range_scaled = scaler_IM.fit_transform(log_IM_range)

    # Define C values based on regulation
    C_values = {"High Regulation": 1, "Medium Regulation": 10, "No Regulation": 1e5}
    C_value = C_values.get(regulation, 1e5)

    for ds in damage_states:
        # Log-transform selected intensity measures
        log_selected_ims = np.log(edps[selected_ims].values + 1e-6).reshape(-1, 1)

        # Scale features separately
        X_scaled = scaler_X.fit_transform(log_selected_ims)
        y = edps[ds]

        # Initialize Logistic Regression model
        model = LogisticRegression(penalty="l2", solver="lbfgs", random_state=100, max_iter=1000, C=C_value)
        model.fit(X_scaled, y)

        # Predict probabilities on the scaled intensity measure range
        probabilities[ds] = model.predict_proba(IM_range_scaled)[:, 1]
        models[ds] = model

    return models, IM_range, probabilities
