import numpy as np
import pandas as pd


def area_3p(point1, point2, point3):
    """
    Calculate the area under the curve formed by three points.
    Each point is [x, y].
    """
    x = 0
    y = 1

    area1 = point2[x] * point2[y] / 2.0
    area2 = point2[y] * (point3[x] - point2[x])
    area3 = (point3[y] - point2[y]) * (point3[x] - point2[x]) / 2.0

    return area1 + area2 + area3


def area_under_pushover(data, idx_max):
    """
    Calculate area under the pushover curve up to row index = idx_max.
    data: pandas DataFrame with columns [Dt(m), Vb(kN)] for displacement & base shear.
    """
    x = 0
    y = 1
    area = 0.0
    for i in range(1, idx_max + 1):
        dx = data.iloc[i, x] - data.iloc[i - 1, x]
        dy = data.iloc[i, y] - data.iloc[i - 1, y]
        triangle = dx * dy / 2.0
        square = dx * data.iloc[i - 1, y]
        area += triangle + square
    return area


def EPP(data, test_converge=0.001):
    """
    Elastic-Perfectly Plastic idealization.
    data: DataFrame with columns [Dt(m), Vb(kN)].
    test_converge: The relative area difference threshold for stopping.
    """
    id_max = data.iloc[:, 1].idxmax()  # index of max base shear
    max_point = data.iloc[id_max, :2].tolist()  # [x, y]
    Po_area = area_under_pushover(data, id_max)

    x = 0.0
    area = 0.0

    # Loop until the area under the 3-point idealization
    # is close to the actual pushover area
    while (abs(Po_area - area) / Po_area) >= test_converge:
        x += 0.00001
        point1 = [0, 0]
        point2 = [x, max_point[1]]
        point3 = max_point
        area = area_3p(point1, point2, point3)

    return point1, point2, point3


def SH(data, test_converge=0.001):
    """
    Strain-Hardening idealization.
    data: DataFrame with columns [Dt(m), Vb(kN)].
    """
    id_max = data.iloc[:, 1].idxmax()
    max_point = data.iloc[id_max, :2].tolist()  # [x, y]
    Po_area = area_under_pushover(data, id_max)

    x = 0.0
    area = 0.0

    while (abs(Po_area - area) / Po_area) >= test_converge:
        x += 0.00001
        point1 = [0, 0]
        # We need an intermediate point2 that lies on the original curve
        # at x=some displacement; but then we adjust it by 0.6, etc.
        original_point2 = get_point2_interpolated(data, x)

        # Example: Reduce it by 0.6 factor, or however your logic states:
        E = original_point2[1] / original_point2[0]  # slope
        # scale down the force
        scaled_force = original_point2[1] / 0.6
        # recalc the disp using the same slope
        scaled_disp = scaled_force / E

        point2 = [scaled_disp, scaled_force]
        point3 = max_point

        area = area_3p(point1, point2, point3)

    return point1, point2, point3


def get_point2_interpolated(data, x):
    """
    Given an x (displacement), find the corresponding y by linear interpolation
    along the 'Dt(m)' column. data is a DataFrame with columns [Dt(m), Vb(kN)].
    """
    if (data["Dt(m)"] == x).any():
        # exact match
        idx = data.index[data["Dt(m)"] == x][0]
        return [x, data["Vb(kN)"].iloc[idx]]
    else:
        # find bracket
        bigger_idx = data.index[data["Dt(m)"] > x]
        if bigger_idx.empty:
            # x is bigger than any in the data
            idx = data.index[-2]  # near the end
        else:
            idx = bigger_idx[0] - 1

        x1 = data["Dt(m)"].iloc[idx]
        x2 = data["Dt(m)"].iloc[idx + 1]
        y1 = data["Vb(kN)"].iloc[idx]
        y2 = data["Vb(kN)"].iloc[idx + 1]

        # linear interpolation
        y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        return [x, y]


def create_idealized_curve(point1, point2, point3, num_points=10):
    """
    Return a DataFrame with columns [Sd, Sa] representing a piecewise-linear
    curve between (point1->point2->point3).
    """
    x_values = np.linspace(point1[0], point3[0], num_points)
    # piecewise:
    # - from point1.x to point2.x is the first slope
    # - from point2.x to point3.x is the second slope
    # We'll do a piecewise definition in 'np.piecewise'
    x_p1 = point1[0]
    x_p2 = point2[0]
    x_p3 = point3[0]

    y_values = np.piecewise(
        x_values,
        [x_values <= x_p2, x_values > x_p2],
        [
            lambda x: (point2[1] / (x_p2 - x_p1)) * (x - x_p1),
            lambda x: point2[1] + (point3[1] - point2[1]) / (x_p3 - x_p2) * (x - x_p2),
        ],
    )

    return pd.DataFrame({"Sd": x_values, "Sa": y_values})
