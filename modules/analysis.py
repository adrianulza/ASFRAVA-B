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


def create_idealized_curve(point1, point2, point3):
    x_values = [point1[0], point2[0], point3[0]]
    y_values = [point1[1], point2[1], point3[1]]
    return pd.DataFrame({"Sd": x_values, "Sa": y_values})


def transform_to_adrs(idealized_curve, Sd_coef, Sa_coef):
    idealized_curve["Sd"] = idealized_curve["Sd"] / Sd_coef
    idealized_curve["Sa"] = idealized_curve["Sa"] / Sa_coef
    return idealized_curve


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

    # Analysis loop over GMRs and scales (IDA)
    sdv_point = []
    pga_point = []
    sa_point = []
    state_list = []
    gmr_list = []

    g = 9.81  # Gravity constant (m/s^2)
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
            dt = abs(GMRS.iloc[1, 0] - GMRS.iloc[0, 0])
            nPts = len(GMRS)
            GMRS_acc = GMRS.iloc[:, 1] * (1 / max(GMRS.iloc[:, 1]))  # Normalizing PGA of GMRS

            intersected = True  # Track intersection status

            for scale in np.arange(min_scale, max_scale + increment, increment):
                scale = round(scale, 2)
                GMRS_acc_scaled = GMRS_acc * scale

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
                    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

                    # Ground motion record
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

                    dtAnalysis = dt * 0.25
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
                    Acc.append(max(u1) * (omega**2) / g)
                    Disp.append(max(u1))

                Acc = np.array(Acc)
                Disp = np.array(Disp)

                rs = pd.DataFrame({"Sd": Disp, "Sa": Acc})  # Always define rs with actual data

                # *** Modification Start: Handling PGA = 0 ***
                if scale == 0:
                    # When PGA = 0, explicitly set all relevant values to 0
                    sdv_point.append(0)  # Set Sd to 0
                    pga_point.append(0)  # Set PGA to 0
                    sa_point.append(0)  # Set SA to 0
                    state = "intersected"  # Set Status to 'intersected'
                # *** Modification End ***

                else:
                    intersection_point = intrs.find_intersection(rs, idealized_adrs_curve, record, scale)

                    if intersection_point is None:
                        state = "not intersected"
                    else:
                        state = "intersected"

                    # Store data and plot information
                    sdv_point.append(fail_state_displ if state == "not intersected" else intersection_point[0])
                    pga_point.append(max(GMRS_acc_scaled))
                    sa_point.append(fail_state_acc if state == "not intersected" else intersection_point[1])

                state_list.append(state)  # Move this outside the if-else block to ensure it's always assigned
                gmr_list.append(record)

                plot_data.append(
                    {
                        "rs": rs,  # Ensure rs is stored regardless of intersection
                        "idealized_adrs_curve": idealized_adrs_curve,
                        "intersection_point": (
                            intersection_point if scale != 0 else (0, 0)
                        ),  # Modified: Set intersection_point to (0, 0) for PGA = 0
                        "record": record,
                        "scale": scale,
                    }
                )

                # Fast mode handling
                """ Skip higher scales if the first intersection is not found """
                if fast_mode and state == "not intersected":
                    # Prefill the rest of the scales as ‘not intersected’
                    remaining_scales = np.arange(scale + increment, max_scale + increment, increment)

                    for sc in remaining_scales:
                        sdv_point.append(fail_state_displ)
                        pga_point.append(sc)
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
                        "[FAST-MODE] %s: first miss at scale %.2f; skipped %d higher scales.",
                        record,
                        scale,
                        len(remaining_scales),
                    )
                    break
                
                if progress_callback:
                    progress_callback()
                
                logger.info("Record: %s, Scale: %.2f, State: %s", record, scale, state)

        except Exception as e:
            logger.error("Analysis error with record %s: %s", record, e, exc_info=True)

    # Write EDPs to CSV
    assert (
        len(sdv_point) == len(pga_point) == len(sa_point) == len(state_list) == len(gmr_list)
    ), "List lengths are not equal"

    edps_df = pd.DataFrame(
        {
            "Sd": sdv_point,
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
