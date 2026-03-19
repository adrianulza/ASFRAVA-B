import logging
import os
import sys
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import modules.idealization as IP

# Load packages from another folders
import modules.intersection as intrs
from utils.config import settings

# ------------------------------------------------------------------
#           Sa(T) PROTOTYPE SETTINGS
# ------------------------------------------------------------------
# NOTE:
# During this prototype the GUI still labels the driving IM as "PGA".
# In this file, however, min_scale/max_scale/increment and the output
# column named "PGA" are interpreted as target Sa(T) values in g.
SA_SCALING_PERIOD = 1.0   # seconds (hard-coded prototype assumption)
SA_SCALING_DAMPING = 0.05 # 5% damping
MIN_IM_TOL = 1.0e-12


def create_idealized_curve(point1, point2, point3):
    x_values = [point1[0], point2[0], point3[0]]
    y_values = [point1[1], point2[1], point3[1]]
    return pd.DataFrame({"Sd": x_values, "Sa": y_values})


def transform_to_adrs(idealized_curve, Sd_coef, Sa_coef):
    idealized_curve["Sd"] = idealized_curve["Sd"] / Sd_coef
    idealized_curve["Sa"] = idealized_curve["Sa"] / Sa_coef
    return idealized_curve


def compute_pseudo_spectral_acceleration(
    acc_g: np.ndarray,
    dt: float,
    period: float = SA_SCALING_PERIOD,
    damping_ratio: float = SA_SCALING_DAMPING,
) -> float:
    """
    Compute 5%-damped pseudo-spectral acceleration Sa(T) in g for a ground
    motion record already expressed in g.

    The elastic SDOF equation is solved in relative coordinates using the
    average-acceleration Newmark method (gamma = 0.5, beta = 0.25):

        u¨ + 2*xi*omega*u˙ + omega^2*u = -ag(t)

    With ag(t) in g, the returned Sa(T) = omega^2 * max|u| is also in g.
    """
    acc_g = np.asarray(acc_g, dtype=float)

    if acc_g.ndim != 1 or acc_g.size < 2:
        raise ValueError("Ground motion record must be a 1D array with at least two samples.")
    if not np.all(np.isfinite(acc_g)):
        raise ValueError("Ground motion record contains NaN or infinite values.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be greater than zero.")
    if period <= 0.0:
        raise ValueError("Spectral period must be greater than zero.")
    if damping_ratio < 0.0:
        raise ValueError("Damping ratio must be non-negative.")

    gamma = 0.5
    beta = 0.25

    omega = 2.0 * np.pi / period
    mass = 1.0
    stiffness = (omega**2) * mass
    damping = 2.0 * damping_ratio * omega * mass

    npts = acc_g.size
    disp = np.zeros(npts)
    vel = np.zeros(npts)
    acc_rel = np.zeros(npts)

    # Unit-mass base-excitation load vector in relative coordinates.
    load = -acc_g
    acc_rel[0] = (load[0] - damping * vel[0] - stiffness * disp[0]) / mass

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = gamma * dt

    k_eff = stiffness + a0 * mass + a1 * damping

    for i in range(npts - 1):
        p_eff = (
            load[i + 1]
            + mass * (a0 * disp[i] + a2 * vel[i] + a3 * acc_rel[i])
            + damping * (a1 * disp[i] + a4 * vel[i] + a5 * acc_rel[i])
        )

        disp[i + 1] = p_eff / k_eff
        acc_rel[i + 1] = a0 * (disp[i + 1] - disp[i]) - a2 * vel[i] - a3 * acc_rel[i]
        vel[i + 1] = vel[i] + a6 * acc_rel[i] + a7 * acc_rel[i + 1]

    return float((omega**2) * np.max(np.abs(disp)))


# ------------------------------------------------------------------
#           PERFORM ANALYSIS TO GET EDPs (OPENSEESPY)
# ------------------------------------------------------------------


def analyze(
    capacity_filepath: str,
    building_params_filepath: str,
    gmrs_folderpath: str,
    min_scale: float,
    max_scale: float,
    increment: float,
    idealization_choice: str,
    output_folder_path: str,
    fast_mode: bool = False,
    progress_callback=None,
):

    # Read input files
    data_pushover = pd.read_csv(capacity_filepath, delimiter=settings.CSV_SEP)
    building_params = pd.read_csv(building_params_filepath, delimiter=settings.CSV_SEP)
    list_gmrs_data = [f for f in os.listdir(gmrs_folderpath) if f.lower().endswith((".txt", ".csv"))]

    if not list_gmrs_data:
        raise FileNotFoundError(f"No *.txt or *.csv ground-motion files found in “{gmrs_folderpath}”")

    # Idealization of the capacity curve
    if idealization_choice == "EPP":
        point1, point2, point3 = IP.EPP(data_pushover, 0.001)
    elif idealization_choice == "SH":
        point1, point2, point3 = IP.SH(data_pushover, 0.001)
    else:
        raise ValueError(f"Unknown idealization choice: {idealization_choice}")

    idealized_curve = create_idealized_curve(point1, point2, point3)

    deformation_y = point2[0]
    strength_y = point2[1]
    deformation_u = point3[0]
    strength_u = point3[1]

    # Calculate coefficients for ADRS transformation
    building_params["normalization_factor"] = building_params.iloc[:, 2] / building_params.iloc[0, 2]
    building_params["Mass(N)"] = building_params.iloc[:, 1] * 9.80665
    mxmode = (building_params["Mass(N)"] * building_params["normalization_factor"]).sum()
    mxmode2 = (building_params["Mass(N)"] * building_params["normalization_factor"] ** 2).sum()
    mxmode3 = mxmode**2
    Sd_coef = mxmode / mxmode2
    Sa_coef = mxmode3 / mxmode2

    int_idealized_adrs = transform_to_adrs(idealized_curve, Sd_coef, Sa_coef)

    ds1_threshold = int_idealized_adrs["Sd"].iloc[1] * 0.75
    ds3_threshold = int_idealized_adrs["Sd"].iloc[2]
    ds2_threshold = (ds1_threshold + ds3_threshold) / 2

    EPP_mod_factor = 1.0

    point1_adrs = int_idealized_adrs["Sd"].iloc[0], int_idealized_adrs["Sa"].iloc[0]
    point2_adrs = int_idealized_adrs["Sd"].iloc[1], int_idealized_adrs["Sa"].iloc[1]
    point3_adrs = int_idealized_adrs["Sd"].iloc[2] * EPP_mod_factor, int_idealized_adrs["Sa"].iloc[2]

    idealized_adrs_curve = create_idealized_curve(point1_adrs, point2_adrs, point3_adrs)

    fail_state_displ = ds3_threshold * 1.01
    fail_state_acc = int_idealized_adrs["Sa"].iloc[2]

    # Analysis loop over GMRs and target Sa(T) levels (prototype IDA)
    sdv_point = []
    pga_point = []  # NOTE: kept for downstream compatibility; stores target Sa(T) in g.
    sa_point = []
    state_list = []
    gmr_list = []

    g = 9.81  # Gravity constant (m/s^2) for OpenSees input scaling
    xDamp = 0.05
    Tinit = 0.000001
    Tmax = 4
    Tstep = 0.02
    Tn = np.arange(Tstep, Tmax + Tstep, Tstep)
    Tn = np.insert(Tn, 0, Tinit)

    plot_data = []
    for record in list_gmrs_data:
        try:
            fp = os.path.join(gmrs_folderpath, record)
            GMRS = pd.read_csv(fp, delimiter=settings.CSV_SEP, header=0, engine="python")

            if len(GMRS) < 2:
                raise ValueError(f"Ground motion {record} must contain at least two time samples.")

            dt = abs(GMRS.iloc[1, 0] - GMRS.iloc[0, 0])
            nPts = len(GMRS)

            GMRS_acc_raw = GMRS.iloc[:, 1].to_numpy(dtype=float)
            peak_abs_acc = np.max(np.abs(GMRS_acc_raw))

            if peak_abs_acc <= 0.0:
                raise ValueError(f"Ground motion {record} has zero peak acceleration; cannot scale.")

            # Prototype scaling IM: Sa(T = 1.0 s, 5% damping), computed from the
            # unscaled record in g and reported in g.
            record_sa_t = compute_pseudo_spectral_acceleration(
                GMRS_acc_raw,
                dt,
                period=SA_SCALING_PERIOD,
                damping_ratio=SA_SCALING_DAMPING,
            )

            if record_sa_t <= MIN_IM_TOL:
                raise ValueError(
                    f"Ground motion {record} has near-zero Sa(T={SA_SCALING_PERIOD:.2f}s); cannot scale."
                )

            logger.info(
                "Record %s: unscaled Sa(T=%.2fs, xi=%.2f)=%.6f g",
                record,
                SA_SCALING_PERIOD,
                SA_SCALING_DAMPING,
                record_sa_t,
            )

            for scale in np.arange(min_scale, max_scale + increment, increment):
                target_sa = round(scale, 2)
                scale_factor = 0.0 if target_sa == 0.0 else target_sa / record_sa_t
                GMRS_acc_scaled = GMRS_acc_raw * scale_factor

                Acc = []
                Disp = []

                for T in Tn:
                    # OpenSees Model
                    ops.wipe()
                    ops.model("Basic", "-ndm", 1, "-ndf", 1)

                    Ki = strength_y / deformation_y
                    Mass = (T**2 * Ki) / (2 * np.pi) ** 2
                    omega = np.sqrt(Ki / Mass)

                    # Nodes and elements
                    ops.node(1, 0)
                    ops.node(2, 0)
                    ops.fix(1, 1)  # Fixed base node
                    ops.mass(2, Mass)  # Mass at the top node

                    # Material definition with hardening
                    matTag = 1
                    Ki = strength_y / deformation_y  # Initial elastic stiffness
                    Hkin = (strength_u - strength_y) / (deformation_u - deformation_y)  # Kinematic hardening (kN/m)

                    ops.uniaxialMaterial("Hardening", matTag, Ki, strength_y, 0, Hkin)
                    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1, "-doRayleigh", 1)

                    # Ground motion record (GMRS_acc_scaled stays in g; OpenSees
                    # converts it to m/s^2 through -factor g exactly as before.)
                    ops.timeSeries("Path", 2, "-values", *GMRS_acc_scaled, "-dt", dt, "-factor", g)
                    ops.pattern("UniformExcitation", 1, 1, "-accel", 2)

                    # Damping
                    betaKcomm = 2.0 * xDamp / omega
                    ops.rayleigh(0.0, 0.0, 0.0, betaKcomm)

                    # Analysis settings
                    ops.wipeAnalysis()
                    ops.system("BandGeneral")
                    ops.constraints("Plain")
                    ops.test("NormDispIncr", 1.0e-6, 200)
                    ops.algorithm("Newton")
                    ops.numberer("RCM")

                    gamma = 0.5
                    beta = 0.25
                    ops.integrator("Newmark", gamma, beta)
                    ops.analysis("Transient")

                    dtAnalysis = dt
                    TmaxAnalysis = dt * nPts
                    tFinal = TmaxAnalysis
                    ok = 0
                    u1 = []

                    # Run analysis with enhanced adaptive time stepping
                    while ok == 0 and ops.getTime() < tFinal:
                        ok = ops.analyze(1, dtAnalysis)
                        if ok != 0:
                            ops.test("NormDispIncr", 1.0e-6, 200)
                            ops.algorithm("Newton")
                            ok = ops.analyze(1, dt)

                            if ok == 0:
                                ops.test("NormDispIncr", 1.0e-6, 10)
                                ops.algorithm("Newton")

                        u1.append(ops.nodeDisp(2, 1))

                    # Store results
                    Acc.append(np.max(np.abs(u1)) * (omega**2) / g)
                    Disp.append(np.max(np.abs(u1)))

                Acc = np.array(Acc)
                Disp = np.array(Disp)

                rs = pd.DataFrame({"Sd": Disp, "Sa": Acc})  # Always define rs with actual data

                if target_sa == 0.0:
                    # When target Sa(T) = 0, explicitly set all relevant values to 0.
                    sdv_point.append(0)
                    pga_point.append(0)  # NOTE: still the compatibility column name.
                    sa_point.append(0)
                    state = "intersected"
                else:
                    intersection_point = intrs.find_intersection(rs, idealized_adrs_curve, record, target_sa)

                    if intersection_point is None:
                        state = "not intersected"
                    else:
                        state = "intersected"

                    # Store data and plot information
                    sdv_point.append(fail_state_displ if state == "not intersected" else intersection_point[0])
                    pga_point.append(target_sa)  # Prototype: store target Sa(T=1.0s) in "PGA" column.
                    sa_point.append(fail_state_acc if state == "not intersected" else intersection_point[1])

                state_list.append(state)
                gmr_list.append(record)

                plot_data.append(
                    {
                        "rs": rs,
                        "idealized_adrs_curve": idealized_adrs_curve,
                        "intersection_point": (
                            intersection_point if target_sa != 0.0 else (0, 0)
                        ),
                        "record": record,
                        "scale": target_sa,
                    }
                )

                # Fast mode handling
                # Skip higher target Sa(T) levels if the first intersection is not found.
                if fast_mode and state == "not intersected":
                    remaining_scales = np.arange(target_sa + increment, max_scale + increment, increment)

                    for sc in remaining_scales:
                        sdv_point.append(fail_state_displ)
                        pga_point.append(sc)  # Prototype: still stores target Sa(T=1.0s).
                        sa_point.append(fail_state_acc)
                        state_list.append("not intersected")
                        gmr_list.append(record)

                        plot_data.append(
                            {
                                "rs": rs,
                                "idealized_adrs_curve": idealized_adrs_curve,
                                "intersection_point": None,
                                "record": record,
                                "scale": sc,
                            }
                        )

                        if progress_callback:
                            progress_callback()

                    logger.info(
                        "[FAST-MODE] %s: first miss at target Sa(T=%.2fs) %.2f g; skipped %d higher levels.",
                        record,
                        SA_SCALING_PERIOD,
                        target_sa,
                        len(remaining_scales),
                    )
                    break

                if progress_callback:
                    progress_callback()

                logger.info(
                    "Record: %s, Target Sa(T=%.2fs): %.2f g, Scale factor: %.6f, State: %s",
                    record,
                    SA_SCALING_PERIOD,
                    target_sa,
                    scale_factor,
                    state,
                )

        except Exception as e:
            logger.error("Analysis error with record %s: %s", record, e, exc_info=True)

    # Write EDPs to CSV
    assert (
        len(sdv_point) == len(pga_point) == len(sa_point) == len(state_list) == len(gmr_list)
    ), "List lengths are not equal"

    edps_df = pd.DataFrame(
        {
            "Sd": sdv_point,
            # NOTE: kept as "PGA" only for GUI / plotting / fragility compatibility.
            # Values stored here are actually target Sa(T=1.0s) in g during this prototype.
            "PGA": pga_point,
            "SA": sa_point,
            "Status": state_list,
            "GMR": gmr_list,
        }
    ).round(4)

    edps_df["ds1"] = (edps_df["Sd"] >= ds1_threshold).astype(int)
    edps_df["ds2"] = (edps_df["Sd"] >= ds2_threshold).astype(int)
    edps_df["ds3"] = (edps_df["Sd"] >= ds3_threshold).astype(int)

    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_folder_path) / f"EDPs_data_{Path(capacity_filepath).name}"
    edps_df.to_csv(out_file, index=False, sep=settings.CSV_SEP)
    logger.info("EDPs written -> %s", out_file)

    return (
        point1,
        point2,
        point3,
        data_pushover,
        ds1_threshold,
        ds2_threshold,
        ds3_threshold,
        plot_data,
        int_idealized_adrs,
    )
