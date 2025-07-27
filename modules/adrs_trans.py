import numpy as np
import pandas as pd


def adrs_transformation(building_params: pd.DataFrame):
    if building_params.empty:
        raise ValueError

    norm_disp_factor = building_params["Mode(unitless)"] / building_params["Mode(unitless)"].iloc[0]
    weight = building_params["Mass(ton)"] * 9.80665  # Convert mass to weight

    mxmode1 = np.sum(weight * norm_disp_factor)
    mxmode2 = np.sum(weight * norm_disp_factor**2)
    mxmode3 = mxmode1**2

    Sd_coef = mxmode1 / mxmode2
    Sa_coef = mxmode3 / mxmode2

    return Sd_coef, Sa_coef


def adrs_capacity(ideal_dt, ideal_vb, Sd_coef, Sa_coef):
    ideal_dt = np.array(ideal_dt, dtype=float)
    ideal_vb = np.array(ideal_vb, dtype=float)

    capacity_Sd = ideal_dt / Sd_coef
    capacity_Sa = ideal_vb / Sa_coef

    return capacity_Sd, capacity_Sa
