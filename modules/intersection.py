import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


# Function to find the intersection using determinants
def line_intersection_determinant(x1, y1, x2, y2, x3, y3, x4, y4):
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    det = a1 * b2 - a2 * b1

    if det == 0:
        return None  # Lines are parallel or coincident
    else:
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        if (
            min(x1, x2) <= x <= max(x1, x2)
            and min(x3, x4) <= x <= max(x3, x4)
            and min(y1, y2) <= y <= max(y1, y2)
            and min(y3, y4) <= y <= max(y3, y4)
        ):
            return x, y
    return None


# Function to refine segments adaptively
def refine_segments(x, y, refinement_factor=6):
    refined_x = []
    refined_y = []
    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        refined_x.extend(np.linspace(x1, x2, refinement_factor, endpoint=False))
        refined_y.extend(np.linspace(y1, y2, refinement_factor, endpoint=False))
    refined_x.append(x[-1])
    refined_y.append(y[-1])
    return np.array(refined_x), np.array(refined_y)


# Function to find the intersection point between two segments
def intersection_point_on_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Parallel lines
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return x, y
    return None


def interpolate_curve(data):
    x_new = np.linspace(data["Sd"].min(), data["Sd"].max())
    y_new = np.interp(x_new, data["Sd"], data["Sa"])
    return pd.DataFrame({"Sd": x_new, "Sa": y_new})


def find_intersection(rs, idealized_adrs_curve, record, scale):
    # Primary method: by using determinant
    intersections = []
    for i in range(len(rs["Sd"]) - 1):
        x1, y1 = rs["Sd"][i], rs["Sa"][i]
        x2, y2 = rs["Sd"][i + 1], rs["Sa"][i + 1]
        for j in range(len(idealized_adrs_curve["Sd"]) - 1):
            x3, y3 = idealized_adrs_curve["Sd"][j], idealized_adrs_curve["Sa"][j]
            x4, y4 = idealized_adrs_curve["Sd"][j + 1], idealized_adrs_curve["Sa"][j + 1]
            intersection = line_intersection_determinant(x1, y1, x2, y2, x3, y3, x4, y4)
            if intersection:
                intersections.append(intersection)

    # Secondary method: by using piecewise linearization
    if not intersections:
        refined_Sd, refined_Sa = refine_segments(rs["Sd"].values, rs["Sa"].values)
        refined_idealized_Sd, refined_idealized_Sa = refine_segments(
            idealized_adrs_curve["Sd"].values, idealized_adrs_curve["Sa"].values
        )

        for i in range(len(refined_Sd) - 1):
            x1, y1 = refined_Sd[i], refined_Sa[i]
            x2, y2 = refined_Sd[i + 1], refined_Sa[i + 1]
            for j in range(len(refined_idealized_Sd) - 1):
                x3, y3 = refined_idealized_Sd[j], refined_idealized_Sa[j]
                x4, y4 = refined_idealized_Sd[j + 1], refined_idealized_Sa[j + 1]
                intersection = intersection_point_on_segment(x1, y1, x2, y2, x3, y3, x4, y4)
                if intersection:
                    intersections.append(intersection)

    if intersections:
        best_intersection = min(intersections, key=lambda point: point[0])
    else:
        best_intersection = None

    # # Print the chosen intersection point
    # """ activate this for printing in terminal and logging """
    # if best_intersection:
    #     # Format intersection point: convert to float and round to 4 decimals
    #     x, y = map(lambda v: round(float(v), 4), best_intersection)
    #     print(f"Intersection found for Record: {record}, Scale: {scale} at point: ({x}, {y})")
    #     return x, y
    # else:
    #     print(f"No intersection for Record: {record}, Scale: {scale}")
    #     return None

    # Logging the intersection point
    """ activate this for logging-only-mode instead of printing in terminal """
    if best_intersection:
        x, y = map(lambda v: round(float(v), 4), best_intersection)
        logger.info("Intersection found for Record: %s, Scale: %.2f at point: (%.4f, %.4f)", record, scale, x, y)
        return x, y
    else:
        logger.info("No intersection for Record: %s, Scale: %.2f", record, scale)
        return None
