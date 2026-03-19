import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import modules.idealization as IP
import modules.intersection as intrs
from utils.config import settings

# ------------------------------------------------------------------
# AvgSa prototype settings
# ------------------------------------------------------------------
# NOTE
# ----
# 1) Ground motions are read in g.
# 2) AvgSa is computed and reported in g.
# 3) OpenSees still receives the motion in g and converts it to m/s^2 through
#    the existing "-factor g" pattern, exactly like the current PGA workflow.
# 4) GUI labels and downstream CSV/fragility code are intentionally left
#    unchanged for this prototype. Therefore, the output CSV column named
#    "PGA" temporarily stores AvgSa values [g].
# 5) The period band is hard-coded for the prototype as agreed:
#       T* = 1.0 s
#       AvgSa over 10 equally spaced periods from 0.2T* to 1.5T*
# 6) A near-zero positive minimum scale (for example 0.00001) is treated as a
#    dedicated anchor level when it is very small compared with the increment.
#    This keeps the main ladder at clean engineering values (0.1, 0.2, ...)
#    while still allowing a positive IM value for later log-statistical work.

AVGSA_REFERENCE_PERIOD = 1.0
AVGSA_MIN_PERIOD_RATIO = 0.2
AVGSA_MAX_PERIOD_RATIO = 1.5
AVGSA_NUM_PERIODS = 10
AVGSA_DAMPING_RATIO = 0.05
AVGSA_NEAR_ZERO_ANCHOR_RATIO = 0.01
GRAVITY = 9.81
CSV_IM_DECIMALS = 8
CSV_RESPONSE_DECIMALS = 4


@dataclass
class ScaleTarget:
    value: float
    kind: str  # "regular", "anchor", or "endpoint"
    count_in_progress: bool = False


def create_idealized_curve(point1, point2, point3):
    x_values = [point1[0], point2[0], point3[0]]
    y_values = [point1[1], point2[1], point3[1]]
    return pd.DataFrame({"Sd": x_values, "Sa": y_values})



def transform_to_adrs(idealized_curve, Sd_coef, Sa_coef):
    idealized_curve["Sd"] = idealized_curve["Sd"] / Sd_coef
    idealized_curve["Sa"] = idealized_curve["Sa"] / Sa_coef
    return idealized_curve


# ------------------------------------------------------------------
# AvgSa helper functions
# ------------------------------------------------------------------
def _tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(v) for v in values))
    return max(1.0e-12, 1.0e-10 * scale)



def _round_target(value: float) -> float:
    return float(np.round(float(value), 12))



def _format_im(value: float) -> str:
    return f"{float(value):.8g}"



def _build_scale_schedule(min_scale: float, max_scale: float, increment: float) -> list[ScaleTarget]:
    """Build the exact AvgSa target list for the prototype.

    Rules
    -----
    1) If min_scale is essentially a near-zero positive anchor compared with the
       increment, keep it as a dedicated first level and start the main ladder
       from one increment.
    2) Otherwise, build the standard arithmetic progression from min_scale.
    3) If the arithmetic progression does not land exactly on max_scale, append
       max_scale explicitly as a final extra endpoint.
    4) Preserve compatibility with the current GUI progress-bar logic by only
       counting the levels that correspond to the legacy formula
           int((max_scale - min_scale) / increment) + 1
       The extra anchor and/or appended endpoint are not counted unless needed
       to satisfy that expected number of progress ticks.
    """
    if min_scale < 0.0:
        raise ValueError("Minimum AvgSa target must be non-negative.")
    if max_scale < min_scale:
        raise ValueError(
            "Maximum AvgSa target must be greater than or equal to minimum target."
        )
    if increment <= 0.0:
        raise ValueError("AvgSa increment must be positive.")

    tol = _tolerance(min_scale, max_scale, increment)
    schedule: list[ScaleTarget] = []

    near_zero_anchor = min_scale > 0.0 and min_scale <= AVGSA_NEAR_ZERO_ANCHOR_RATIO * increment
    if near_zero_anchor:
        schedule.append(ScaleTarget(value=_round_target(min_scale), kind="anchor"))
        start = increment
    else:
        start = min_scale

    k = 0
    while True:
        target = start + k * increment
        if target > max_scale + tol:
            break
        schedule.append(ScaleTarget(value=_round_target(target), kind="regular"))
        k += 1

    if not schedule or not np.isclose(schedule[-1].value, max_scale, rtol=0.0, atol=tol):
        schedule.append(ScaleTarget(value=_round_target(max_scale), kind="endpoint"))

    # Stable de-duplication while preserving order.
    deduped: list[ScaleTarget] = []
    for item in schedule:
        if not deduped:
            deduped.append(item)
            continue
        if np.isclose(item.value, deduped[-1].value, rtol=0.0, atol=tol):
            # If duplicates occur, keep the more meaningful kind priority:
            # regular > endpoint > anchor
            priority = {"anchor": 0, "endpoint": 1, "regular": 2}
            if priority[item.kind] > priority[deduped[-1].kind]:
                deduped[-1] = item
        else:
            deduped.append(item)
    schedule = deduped

    expected_progress_steps = int((max_scale - min_scale) / increment) + 1
    regular_indices = [idx for idx, item in enumerate(schedule) if item.kind == "regular"]
    for idx in regular_indices:
        schedule[idx].count_in_progress = True

    deficit = expected_progress_steps - len(regular_indices)
    if deficit > 0:
        # Prefer to count an appended endpoint before a near-zero anchor when
        # the GUI expects one more step than the regular ladder provides.
        special_priority = [
            idx for idx, item in enumerate(schedule) if item.kind == "endpoint"
        ] + [idx for idx, item in enumerate(schedule) if item.kind == "anchor"]
        for idx in special_priority[:deficit]:
            schedule[idx].count_in_progress = True

    if len(schedule) != expected_progress_steps:
        logger.info(
            "AvgSa schedule has %d target levels, while the current GUI progress logic expects %d. "
            "The extra anchor and/or explicit endpoint are handled inside analysis.py.",
            len(schedule),
            expected_progress_steps,
        )

    return schedule



def _build_avgsa_periods(
    reference_period: float = AVGSA_REFERENCE_PERIOD,
    min_ratio: float = AVGSA_MIN_PERIOD_RATIO,
    max_ratio: float = AVGSA_MAX_PERIOD_RATIO,
    n_points: int = AVGSA_NUM_PERIODS,
) -> np.ndarray:
    """Build the hard-coded AvgSa period set for the prototype.

    The prototype follows the common building-oriented AvgSa convention of using
    10 equally spaced periods across approximately 0.2T1 to 1.5T1.
    """
    if reference_period <= 0.0:
        raise ValueError("Reference period T* must be positive.")
    if min_ratio <= 0.0 or max_ratio <= 0.0:
        raise ValueError("AvgSa period ratios must be positive.")
    if max_ratio <= min_ratio:
        raise ValueError(
            "Upper AvgSa period ratio must be greater than lower period ratio."
        )
    if n_points < 2:
        raise ValueError("At least two periods are required to compute AvgSa.")

    return np.linspace(reference_period * min_ratio, reference_period * max_ratio, num=n_points)



def _extract_uniform_dt(time_values: np.ndarray) -> float:
    """Validate a uniformly sampled time vector and return dt."""
    time_values = np.asarray(time_values, dtype=float)
    if time_values.size < 2:
        raise ValueError("Ground motion file must contain at least two time samples.")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("Ground motion time column contains NaN or infinite values.")

    dt_values = np.diff(time_values)
    if np.any(dt_values <= 0.0):
        raise ValueError("Ground motion time column must be strictly increasing.")

    dt = float(np.mean(dt_values))
    tol = max(1.0e-8, 1.0e-6 * dt)
    if not np.allclose(dt_values, dt, rtol=1.0e-4, atol=tol):
        raise ValueError(
            "Ground motion time step is not uniform enough for response-spectrum evaluation."
        )

    return dt



def _elastic_response_spectrum(
    acc_g: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping_ratio: float = AVGSA_DAMPING_RATIO,
    gravity: float = GRAVITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 5%-damped elastic pseudo-response spectrum.

    The SDOF equation is solved using a continuous-time state-space model and
    scipy.signal.lsim with linear interpolation of the input between samples.
    The input motion is provided in g and converted internally to m/s^2 only for
    solving the linear oscillator equations. The returned pseudo-spectral
    acceleration is converted back to g.
    """
    acc_g = np.asarray(acc_g, dtype=float)
    periods = np.asarray(periods, dtype=float)

    if acc_g.ndim != 1:
        raise ValueError("Ground motion acceleration array must be one-dimensional.")
    if acc_g.size < 2:
        raise ValueError("Ground motion acceleration array must contain at least two samples.")
    if not np.all(np.isfinite(acc_g)):
        raise ValueError("Ground motion acceleration array contains NaN or infinite values.")
    if dt <= 0.0:
        raise ValueError("Time step dt must be positive.")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("All spectrum periods must be finite and positive.")
    if damping_ratio < 0.0:
        raise ValueError("Damping ratio must be non-negative.")

    acc_si = acc_g * gravity
    time_vector = np.arange(acc_si.size, dtype=float) * dt

    sd = np.zeros(periods.size, dtype=float)
    sa = np.zeros(periods.size, dtype=float)

    for idx, period in enumerate(periods):
        omega = 2.0 * np.pi / period

        A = np.array(
            [[0.0, 1.0], [-omega**2, -2.0 * damping_ratio * omega]],
            dtype=float,
        )
        B = np.array([[0.0], [-1.0]], dtype=float)
        C = np.array([[1.0, 0.0]], dtype=float)
        D = np.array([[0.0]], dtype=float)

        sdof_system = signal.StateSpace(A, B, C, D)
        _, rel_disp, _ = signal.lsim(
            sdof_system,
            U=acc_si,
            T=time_vector,
            X0=np.array([0.0, 0.0], dtype=float),
            interp=True,
        )

        rel_disp = np.asarray(rel_disp, dtype=float).reshape(-1)
        sd[idx] = np.max(np.abs(rel_disp))
        sa[idx] = (omega**2) * sd[idx] / gravity

    return sd, sa



def _compute_avgsa_from_spectrum(sa_values: np.ndarray) -> float:
    """Compute AvgSa as the geometric mean of spectral ordinates."""
    sa_values = np.asarray(sa_values, dtype=float)
    if sa_values.size == 0:
        raise ValueError("AvgSa cannot be computed from an empty Sa vector.")
    if not np.all(np.isfinite(sa_values)):
        raise ValueError("Spectrum contains NaN or infinite Sa values.")
    if np.any(sa_values <= 0.0):
        raise ValueError(
            "AvgSa requires strictly positive Sa(T) ordinates. "
            "The response-spectrum calculation produced a non-positive value."
        )

    return float(np.exp(np.mean(np.log(sa_values))))



def _compute_record_avgsa(
    acc_g: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping_ratio: float = AVGSA_DAMPING_RATIO,
) -> float:
    _, sa_values = _elastic_response_spectrum(
        acc_g=acc_g,
        dt=dt,
        periods=periods,
        damping_ratio=damping_ratio,
        gravity=GRAVITY,
    )
    return _compute_avgsa_from_spectrum(sa_values)


# ------------------------------------------------------------------
# PERFORM ANALYSIS TO GET EDPs (OPENSEESPY)
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
    """Run the ASFRAVA-B analysis using AvgSa-based amplitude scaling.

    Prototype behavior
    ------------------
    - The GUI fields min_scale, max_scale, increment are interpreted as target
      AvgSa values in g.
    - Each record is scaled so that the scaled motion matches the requested
      AvgSa target over the hard-coded period band 0.2T* to 1.5T* with T*=1.0 s.
    - The output CSV column still named "PGA" temporarily stores AvgSa values [g]
      for compatibility with the current GUI and downstream modules.

    Important compatibility note
    ----------------------------
    This file implements the exact AvgSa scale schedule for the prototype,
    including the near-zero anchor and an optional explicit max endpoint.
    If you later want MSA fragility fitting to use the exact same IM grid, the
    same schedule logic should also be reused in modules/fragility.py.
    """
    # Read input files
    data_pushover = pd.read_csv(capacity_filepath, delimiter=settings.CSV_SEP)
    building_params = pd.read_csv(building_params_filepath, delimiter=settings.CSV_SEP)

    list_gmrs_data = sorted(
        f for f in os.listdir(gmrs_folderpath) if f.lower().endswith((".txt", ".csv"))
    )
    if not list_gmrs_data:
        raise FileNotFoundError(
            f'No *.txt or *.csv ground-motion files found in "{gmrs_folderpath}"'
        )

    scale_schedule = _build_scale_schedule(min_scale, max_scale, increment)
    avgsa_periods = _build_avgsa_periods()
    logger.info(
        "AvgSa prototype active: T*=%.3f s, %d equally spaced periods from %.3f s to %.3f s.",
        AVGSA_REFERENCE_PERIOD,
        AVGSA_NUM_PERIODS,
        avgsa_periods[0],
        avgsa_periods[-1],
    )
    logger.info(
        "AvgSa targets: %s",
        ", ".join(_format_im(item.value) for item in scale_schedule),
    )

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
    building_params["normalization_factor"] = (
        building_params.iloc[:, 2] / building_params.iloc[0, 2]
    )
    building_params["Mass(N)"] = building_params.iloc[:, 1] * 9.80665

    mxmode = (building_params["Mass(N)"] * building_params["normalization_factor"]).sum()
    mxmode2 = (
        building_params["Mass(N)"] * building_params["normalization_factor"] ** 2
    ).sum()
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
    point3_adrs = (
        int_idealized_adrs["Sd"].iloc[2] * EPP_mod_factor,
        int_idealized_adrs["Sa"].iloc[2],
    )
    idealized_adrs_curve = create_idealized_curve(point1_adrs, point2_adrs, point3_adrs)
    fail_state_displ = ds3_threshold * 1.01
    fail_state_acc = int_idealized_adrs["Sa"].iloc[2]

    # Analysis loop over GMRs and AvgSa targets
    sdv_point = []
    im_point = []  # stored under CSV header "PGA" for temporary compatibility
    sa_point = []
    state_list = []
    gmr_list = []

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
            if GMRS.shape[1] < 2:
                raise ValueError(
                    f"Ground motion {record} must contain at least two columns: time and acceleration."
                )

            time_values = GMRS.iloc[:, 0].to_numpy(dtype=float)
            GMRS_acc_raw = GMRS.iloc[:, 1].to_numpy(dtype=float)
            if not np.all(np.isfinite(GMRS_acc_raw)):
                raise ValueError(
                    f"Ground motion {record} contains NaN or infinite acceleration values."
                )

            dt = _extract_uniform_dt(time_values)
            nPts = len(GMRS_acc_raw)

            peak_abs_acc = np.max(np.abs(GMRS_acc_raw))
            if peak_abs_acc <= 0.0:
                raise ValueError(f"Ground motion {record} has zero peak acceleration; cannot scale.")

            avgsa_raw = _compute_record_avgsa(
                acc_g=GMRS_acc_raw,
                dt=dt,
                periods=avgsa_periods,
                damping_ratio=AVGSA_DAMPING_RATIO,
            )
            if avgsa_raw <= 0.0:
                raise ValueError(f"Ground motion {record} produced non-positive AvgSa; cannot scale.")

            # Normalize once by the record's raw AvgSa and reuse for every target.
            GMRS_acc_unit_avgsa = GMRS_acc_raw / avgsa_raw
            logger.info("Record %s raw AvgSa = %s g", record, _format_im(avgsa_raw))

            for scale_idx, scale_target in enumerate(scale_schedule):
                target_avgsa = float(scale_target.value)

                if np.isclose(target_avgsa, 0.0, rtol=0.0, atol=1.0e-15):
                    zero_rs = pd.DataFrame(
                        {
                            "Sd": np.zeros_like(Tn, dtype=float),
                            "Sa": np.zeros_like(Tn, dtype=float),
                        }
                    )
                    state = "intersected"
                    intersection_point = (0.0, 0.0)

                    sdv_point.append(0.0)
                    im_point.append(0.0)
                    sa_point.append(0.0)
                    state_list.append(state)
                    gmr_list.append(record)
                    plot_data.append(
                        {
                            "rs": zero_rs,
                            "idealized_adrs_curve": idealized_adrs_curve,
                            "intersection_point": intersection_point,
                            "record": record,
                            "scale": target_avgsa,
                        }
                    )

                    if progress_callback and scale_target.count_in_progress:
                        progress_callback()
                    logger.info(
                        "Record: %s, Target AvgSa: %s g, State: %s",
                        record,
                        _format_im(target_avgsa),
                        state,
                    )
                    continue

                scale_factor = target_avgsa / avgsa_raw
                GMRS_acc_scaled = GMRS_acc_unit_avgsa * target_avgsa

                Acc = []
                Disp = []

                for T in Tn:
                    # OpenSees model
                    ops.wipe()
                    ops.model("Basic", "-ndm", 1, "-ndf", 1)

                    Ki = strength_y / deformation_y
                    Mass = (T**2 * Ki) / (2 * np.pi) ** 2
                    omega = np.sqrt(Ki / Mass)

                    # Nodes and elements
                    ops.node(1, 0)
                    ops.node(2, 0)
                    ops.fix(1, 1)
                    ops.mass(2, Mass)

                    # Material definition with hardening
                    matTag = 1
                    Ki = strength_y / deformation_y
                    Hkin = (strength_u - strength_y) / (deformation_u - deformation_y)
                    ops.uniaxialMaterial("Hardening", matTag, Ki, strength_y, 0, Hkin)
                    ops.element(
                        "zeroLength",
                        1,
                        1,
                        2,
                        "-mat",
                        1,
                        "-dir",
                        1,
                        "-doRayleigh",
                        1,
                    )

                    # Ground motion record (still provided in g; OpenSees converts via -factor g)
                    ops.timeSeries(
                        "Path",
                        2,
                        "-values",
                        *GMRS_acc_scaled,
                        "-dt",
                        dt,
                        "-factor",
                        GRAVITY,
                    )
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
                    ops.integrator("Newmark", 0.5, 0.25)
                    ops.analysis("Transient")

                    dtAnalysis = dt
                    TmaxAnalysis = dt * nPts
                    tFinal = TmaxAnalysis
                    ok = 0
                    u1 = []

                    # Run analysis with existing adaptive strategy.
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

                    max_disp = np.max(np.abs(u1)) if u1 else 0.0
                    Acc.append(max_disp * (omega**2) / GRAVITY)
                    Disp.append(max_disp)

                rs = pd.DataFrame({"Sd": np.asarray(Disp), "Sa": np.asarray(Acc)})
                intersection_point = intrs.find_intersection(
                    rs,
                    idealized_adrs_curve,
                    record,
                    target_avgsa,
                )

                if intersection_point is None:
                    state = "not intersected"
                else:
                    state = "intersected"

                sdv_point.append(
                    fail_state_displ if state == "not intersected" else intersection_point[0]
                )
                im_point.append(target_avgsa)
                sa_point.append(
                    fail_state_acc if state == "not intersected" else intersection_point[1]
                )
                state_list.append(state)
                gmr_list.append(record)
                plot_data.append(
                    {
                        "rs": rs,
                        "idealized_adrs_curve": idealized_adrs_curve,
                        "intersection_point": intersection_point,
                        "record": record,
                        "scale": target_avgsa,
                    }
                )

                if fast_mode and state == "not intersected":
                    remaining_targets = scale_schedule[scale_idx + 1 :]
                    for remaining in remaining_targets:
                        sdv_point.append(fail_state_displ)
                        im_point.append(float(remaining.value))
                        sa_point.append(fail_state_acc)
                        state_list.append("not intersected")
                        gmr_list.append(record)
                        plot_data.append(
                            {
                                "rs": rs,
                                "idealized_adrs_curve": idealized_adrs_curve,
                                "intersection_point": None,
                                "record": record,
                                "scale": float(remaining.value),
                            }
                        )
                        if progress_callback and remaining.count_in_progress:
                            progress_callback()

                    logger.info(
                        "[FAST-MODE] %s: first miss at target AvgSa %s g; skipped %d higher targets. "
                        "Scale factor used at miss = %.6f.",
                        record,
                        _format_im(target_avgsa),
                        len(remaining_targets),
                        scale_factor,
                    )
                    break

                if progress_callback and scale_target.count_in_progress:
                    progress_callback()
                logger.info(
                    "Record: %s, Target AvgSa: %s g, Scale factor: %.6f, State: %s",
                    record,
                    _format_im(target_avgsa),
                    scale_factor,
                    state,
                )

        except Exception as e:
            logger.error("Analysis error with record %s: %s", record, e, exc_info=True)

    # Write EDPs to CSV
    assert (
        len(sdv_point) == len(im_point) == len(sa_point) == len(state_list) == len(gmr_list)
    ), "List lengths are not equal"

    edps_df = pd.DataFrame(
        {
            "Sd": sdv_point,
            "PGA": im_point,  # temporary compatibility: this now stores AvgSa [g]
            "SA": sa_point,
            "Status": state_list,
            "GMR": gmr_list,
        }
    )

    # Preserve the near-zero AvgSa anchor in the IM column while keeping the
    # historical CSV compactness for response values.
    edps_df["Sd"] = edps_df["Sd"].round(CSV_RESPONSE_DECIMALS)
    edps_df["PGA"] = edps_df["PGA"].round(CSV_IM_DECIMALS)
    edps_df["SA"] = edps_df["SA"].round(CSV_RESPONSE_DECIMALS)

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
