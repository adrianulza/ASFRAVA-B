from __future__ import annotations

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
import modules.intersection as intrs
from utils.config import settings


def create_idealized_curve(point1, point2, point3):
    return pd.DataFrame({"Sd": [point1[0], point2[0], point3[0]], "Sa": [point1[1], point2[1], point3[1]]})


def transform_to_adrs(idealized_curve, Sd_coef, Sa_coef):
    idealized_curve["Sd"] = idealized_curve["Sd"] / Sd_coef
    idealized_curve["Sa"] = idealized_curve["Sa"] / Sa_coef
    return idealized_curve


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
    im_column: str = "PGA",
    base_period=None,
):
    data_pushover = pd.read_csv(capacity_filepath, delimiter=settings.CSV_SEP)
    building_params = pd.read_csv(building_params_filepath, delimiter=settings.CSV_SEP)

    list_gmrs_data = [f for f in os.listdir(gmrs_folderpath) if f.lower().endswith((".txt", ".csv"))]
    if not list_gmrs_data:
        raise FileNotFoundError(f"No *.txt or *.csv ground-motion files found in '{gmrs_folderpath}'")

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
    ds2_threshold = (ds1_threshold + ds3_threshold) / 2.0

    point1_adrs = (int_idealized_adrs["Sd"].iloc[0], int_idealized_adrs["Sa"].iloc[0])
    point2_adrs = (int_idealized_adrs["Sd"].iloc[1], int_idealized_adrs["Sa"].iloc[1])
    point3_adrs = (int_idealized_adrs["Sd"].iloc[2], int_idealized_adrs["Sa"].iloc[2])
    idealized_adrs_curve = create_idealized_curve(point1_adrs, point2_adrs, point3_adrs)

    fail_state_displ = ds3_threshold * 1.01
    fail_state_acc = int_idealized_adrs["Sa"].iloc[2]

    sdv_point = []
    im_point = []
    sa_point = []
    state_list = []
    gmr_list = []

    g = 9.81
    xDamp = 0.05
    Tinit = 0.000001
    Tmax = 4.0
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
            GMRS_acc_raw = GMRS.iloc[:, 1].to_numpy(dtype=float)

            peak_abs_acc = np.max(np.abs(GMRS_acc_raw))
            if peak_abs_acc <= 0.0:
                raise ValueError(f"Ground motion {record} has zero peak acceleration; cannot normalize.")

            GMRS_acc = GMRS_acc_raw / peak_abs_acc

            for scale in np.arange(min_scale, max_scale + increment, increment):
                scale = round(float(scale), 8)
                GMRS_acc_scaled = GMRS_acc * scale

                Acc = []
                Disp = []

                for T in Tn:
                    ops.wipe()
                    ops.model("Basic", "-ndm", 1, "-ndf", 1)

                    Ki = strength_y / deformation_y
                    Mass = (T**2 * Ki) / (2 * np.pi) ** 2
                    omega = np.sqrt(Ki / Mass)

                    ops.node(1, 0)
                    ops.node(2, 0)
                    ops.fix(1, 1)
                    ops.mass(2, Mass)

                    matTag = 1
                    Hkin = (strength_u - strength_y) / (deformation_u - deformation_y)
                    ops.uniaxialMaterial("Hardening", matTag, Ki, strength_y, 0, Hkin)
                    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1, "-doRayleigh", 1)

                    ops.timeSeries("Path", 2, "-values", *GMRS_acc_scaled, "-dt", dt, "-factor", g)
                    ops.pattern("UniformExcitation", 1, 1, "-accel", 2)

                    betaKcomm = 2.0 * xDamp / omega
                    ops.rayleigh(0.0, 0.0, 0.0, betaKcomm)

                    ops.wipeAnalysis()
                    ops.system("BandGeneral")
                    ops.constraints("Plain")
                    ops.test("NormDispIncr", 1.0e-6, 200)
                    ops.algorithm("Newton")
                    ops.numberer("RCM")
                    ops.integrator("Newmark", 0.5, 0.25)
                    ops.analysis("Transient")

                    tFinal = dt * nPts
                    ok = 0
                    u1 = []

                    while ok == 0 and ops.getTime() < tFinal:
                        ok = ops.analyze(1, dt)
                        if ok != 0:
                            ops.test("NormDispIncr", 1.0e-6, 200)
                            ops.algorithm("Newton")
                            ok = ops.analyze(1, dt)
                        if ok == 0:
                            ops.test("NormDispIncr", 1.0e-6, 10)
                            ops.algorithm("Newton")
                            u1.append(ops.nodeDisp(2, 1))

                    max_disp = np.max(np.abs(u1)) if u1 else 0.0
                    Acc.append(max_disp * (omega**2) / g)
                    Disp.append(max_disp)

                Acc = np.asarray(Acc)
                Disp = np.asarray(Disp)
                rs = pd.DataFrame({"Sd": Disp, "Sa": Acc})

                if np.isclose(scale, 0.0):
                    state = "intersected"
                    intersection_point = (0.0, 0.0)
                else:
                    intersection_point = intrs.find_intersection(rs, idealized_adrs_curve, record, scale)
                    state = "not intersected" if intersection_point is None else "intersected"

                sdv_point.append(fail_state_displ if state == "not intersected" else intersection_point[0])
                im_point.append(float(np.max(np.abs(GMRS_acc_scaled))))
                sa_point.append(fail_state_acc if state == "not intersected" else intersection_point[1])
                state_list.append(state)
                gmr_list.append(record)

                plot_data.append(
                    {
                        "rs": rs,
                        "idealized_adrs_curve": idealized_adrs_curve,
                        "intersection_point": intersection_point,
                        "record": record,
                        "scale": scale,
                    }
                )

                if fast_mode and state == "not intersected":
                    remaining_scales = np.arange(scale + increment, max_scale + increment, increment)
                    for sc in remaining_scales:
                        sc = round(float(sc), 8)
                        sdv_point.append(fail_state_displ)
                        im_point.append(sc)
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
                        "[FAST-MODE] %s: first miss at scale %.4f; skipped %d higher scales.",
                        record,
                        scale,
                        len(remaining_scales),
                    )
                    break

                if progress_callback:
                    progress_callback()

                logger.info("Record: %s, Scale: %.4f, State: %s", record, scale, state)

        except Exception as exc:
            logger.error("Analysis error with record %s: %s", record, exc, exc_info=True)

    assert len(sdv_point) == len(im_point) == len(sa_point) == len(state_list) == len(gmr_list), "List lengths mismatch"

    edps_df = pd.DataFrame(
        {
            "Sd": sdv_point,
            im_column: im_point,
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
