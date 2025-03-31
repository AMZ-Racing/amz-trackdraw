import math
import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev

# ---------------------------- UTILITY FUNCTIONS ---------------------------- #
def create_closed_spline(control_points, num_points=100):
    """
    Compute a smooth, closed B-spline from a list of control points.
    We assume the user-specified control points form the basis;
    splprep with per=True will create a periodic (closed) spline.
    """
    pts = np.array(control_points)
    # add first point to end to close the loop
    pts = np.vstack((pts, pts[0]))
    # if we have only one point, return it
    if len(pts) <= 2:
        return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.column_stack((x_fine, y_fine))


def robust_parallel_offset(ls, distance, side, join_style=2):
    """
    Compute a parallel offset of a LineString using shapely's parallel_offset.
    If the result is a MultiLineString, return the longest LineString.
    """
    offset = ls.parallel_offset(distance, side, join_style=join_style)
    if offset.is_empty:
        return None
    if offset.geom_type == "MultiLineString":
        lines = list(offset)
        lines.sort(key=lambda l: l.length, reverse=True)
        return lines[0]
    return offset


def generate_offset_boundaries(track_points, track_width_meters, px_per_m):
    """
    Compute left and right boundaries as parallel offsets from the centerline.
    This method uses Shapely's parallel_offset for more robust behavior in tight corners.
    """
    track_width_px = track_width_meters * px_per_m
    ls = LineString(track_points)
    left_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'left', join_style=2)
    right_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'right', join_style=2)
    if left_offset is None or right_offset is None:
        return None, None
    left_coords = list(left_offset.coords)
    right_coords = list(right_offset.coords)
    return np.array(left_coords), np.array(right_coords)


def sample_cones(boundary, cone_spacing_meters, px_per_m):
    """Sample points along a boundary so that they are approximately cone_spacing_meters apart."""
    cone_spacing_px = cone_spacing_meters * px_per_m
    pts = boundary
    if len(pts) < 2:
        return pts
    distances = [0]
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        distances.append(distances[-1] + d)
    total_length = distances[-1]
    num_cones = max(2, int(total_length // cone_spacing_px))
    sample_d = np.linspace(0, total_length, num_cones)
    sampled = []
    for sd in sample_d:
        for i in range(1, len(distances)):
            if distances[i] >= sd:
                t = (sd - distances[i-1]) / (distances[i] - distances[i-1])
                x = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
                y = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
                sampled.append((x, y))
                break
    return np.array(sampled)
