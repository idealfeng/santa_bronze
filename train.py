# Santa 2025 - Christmas Tree Packing Challenge (local solver)
# Generates a valid `submission.csv` using public baselines + local search.
#
# Notes
# - The competition provides only `sample_submission.csv`; tree geometry is fixed and defined in the public metric.
# - This script matches the metric's collision logic (shapely + touches allowance) and bounding-square scoring.
#
# Dependencies: pandas, shapely

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from shapely import affinity
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shapely\n"
        "Install with: pip install shapely\n"
        f"Original import error: {e}"
    )


XY_MIN, XY_MAX = -100.0, 100.0

# Match public metric (santa-2025-metric.ipynb)
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e18")
SCALE = float(SCALE_FACTOR)

# Fixed tree bounds (unscaled) for common angles used in grid layouts.
# These are exact for 0/180 since the widest point is base_w/2=0.35 at y=0,
# and the vertical extremes are trunk_bottom=-0.2 and tip_y=0.8.
TREE_BOUNDS: Dict[int, Tuple[float, float, float, float]] = {
    0: (-0.35, -0.2, 0.35, 0.8),
    180: (-0.35, -0.8, 0.35, 0.2),
}

# Unscaled tree polygon vertices, matching ChristmasTree.__post_init__.
TREE_VERTS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
)

_TREE_POLY_SCALED = Polygon([(vx * SCALE, vy * SCALE) for vx, vy in TREE_VERTS])


def _tree_polygon_scaled(x: float, y: float, deg: float) -> Polygon:
    rotated = affinity.rotate(_TREE_POLY_SCALED, float(deg), origin=(0, 0))
    return affinity.translate(rotated, xoff=float(x) * SCALE, yoff=float(y) * SCALE)


def _bounds_for_fixed_angles(placements: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for x, y, deg in placements:
        b = TREE_BOUNDS.get(int(round(deg)) % 360)
        if b is None:
            poly = ChristmasTree(x, y, deg).polygon
            bx0, by0, bx1, by1 = (v / float(SCALE_FACTOR) for v in poly.bounds)
        else:
            bx0, by0, bx1, by1 = b
        min_x = min(min_x, x + bx0)
        max_x = max(max_x, x + bx1)
        min_y = min(min_y, y + by0)
        max_y = max(max_y, y + by1)
    return min_x, min_y, max_x, max_y


def side_length_for(placements: List[Tuple[float, float, float]]) -> float:
    min_x, min_y, max_x, max_y = _bounds_for_fixed_angles(placements)
    return max(max_x - min_x, max_y - min_y)


def _lattice_points(
    *,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: float = 0.35,
    span_x: int = 30,
    span_y: int = 30,
) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    for j in range(-span_y, span_y + 1):
        y_even = j * y_step
        y_odd = j * y_step + y_odd_offset
        for i in range(-span_x, span_x + 1):
            pts.append((i * dx, y_even, 0.0))
            pts.append((i * dx + odd_x_offset, y_odd, 180.0))
    return pts


def build_submission_lattice_greedy(
    n_max: int,
    *,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: float = 0.35,
    span_x: int = 30,
    span_y: int = 30,
) -> pd.DataFrame:
    # Build a nested solution by greedily adding lattice points that minimize the current
    # bounding-square side length. Fast and fully collision-free by construction.
    pts = _lattice_points(
        dx=dx,
        y_step=y_step,
        y_odd_offset=y_odd_offset,
        odd_x_offset=odd_x_offset,
        span_x=span_x,
        span_y=span_y,
    )

    # Precompute each point's bounds contribution.
    contrib = []
    for x, y, deg in pts:
        b = TREE_BOUNDS[int(round(deg)) % 360]
        contrib.append((x + b[0], y + b[1], x + b[2], y + b[3]))

    selected_idx: List[int] = []
    used = [False] * len(pts)
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for n in range(1, n_max + 1):
        best_i = None
        best_side = float("inf")
        best_area = float("inf")
        best_aspect = float("inf")

        for i, (cx0, cy0, cx1, cy1) in enumerate(contrib):
            if used[i]:
                continue
            nx0 = min(min_x, cx0)
            ny0 = min(min_y, cy0)
            nx1 = max(max_x, cx1)
            ny1 = max(max_y, cy1)
            w = nx1 - nx0
            h = ny1 - ny0
            side = w if w >= h else h

            if side < best_side:
                best_side = side
                best_area = w * h
                best_aspect = abs(w - h)
                best_i = i
            elif side == best_side:
                area = w * h
                aspect = abs(w - h)
                if area < best_area or (area == best_area and aspect < best_aspect):
                    best_area = area
                    best_aspect = aspect
                    best_i = i

        assert best_i is not None, "Not enough lattice points; increase span_x/span_y."
        used[best_i] = True
        selected_idx.append(best_i)
        cx0, cy0, cx1, cy1 = contrib[best_i]
        min_x = min(min_x, cx0)
        min_y = min(min_y, cy0)
        max_x = max(max_x, cx1)
        max_y = max(max_y, cy1)

    # Emit groups as first n selected points (nested).
    rows = []
    for n in range(1, n_max + 1):
        for i_t, idx in enumerate(selected_idx[:n]):
            x, y, deg = pts[idx]
            rows.append({"id": f"{n:03d}_{i_t}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def _clamp_xy(x: Decimal, y: Decimal) -> Tuple[Decimal, Decimal]:
    x_f = float(x)
    y_f = float(y)
    if not (XY_MIN <= x_f <= XY_MAX and XY_MIN <= y_f <= XY_MAX):
        x = Decimal(str(min(XY_MAX, max(XY_MIN, x_f))))
        y = Decimal(str(min(XY_MAX, max(XY_MIN, y_f))))
    return x, y


@dataclass
class ChristmasTree:
    center_x: Decimal | str | float = Decimal("0")
    center_y: Decimal | str | float = Decimal("0")
    angle: Decimal | str | float = Decimal("0")
    polygon: Polygon = None  # set in __post_init__

    def __post_init__(self) -> None:
        self.center_x = Decimal(str(self.center_x))
        self.center_y = Decimal(str(self.center_y))
        self.angle = Decimal(str(self.angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at tip
                (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
                # Right side - Top tier
                (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                # Right side - Middle tier
                (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Right side - Bottom tier
                (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Right trunk
                (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (trunk_w / Decimal("2") * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
                # Left trunk
                (-(trunk_w / Decimal("2")) * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
                (-(trunk_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Bottom tier
                (-(base_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Left side - Middle tier
                (-(mid_w / Decimal("4")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (-(mid_w / Decimal("2")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Left side - Top tier
                (-(top_w / Decimal("4")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (-(top_w / Decimal("2")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR),
        )


def generate_weighted_angle() -> float:
    while True:
        angle = random.uniform(0.0, 2.0 * math.pi)
        if random.uniform(0.0, 1.0) < abs(math.sin(2.0 * angle)):
            return angle


def _has_overlap(candidate_poly: Polygon, tree_index: STRtree, placed_polygons: List[Polygon]) -> bool:
    possible_indices = tree_index.query(candidate_poly)
    for i in possible_indices:
        other = placed_polygons[i]
        if candidate_poly.intersects(other) and not candidate_poly.touches(other):
            return True
    return False


def initialize_trees(
    num_trees: int,
    *,
    existing_trees: Optional[Iterable[ChristmasTree]] = None,
    attempts: int = 10,
    start_radius: Decimal = Decimal("20.0"),
    step_in: Decimal = Decimal("0.5"),
    step_out: Decimal = Decimal("0.05"),
) -> Tuple[List[ChristmasTree], Decimal]:
    if num_trees <= 0:
        return [], Decimal("0")

    placed_trees = list(existing_trees) if existing_trees is not None else []
    num_to_add = num_trees - len(placed_trees)
    if num_to_add < 0:
        placed_trees = placed_trees[:num_trees]
        num_to_add = 0

    if num_to_add > 0:
        unplaced = [ChristmasTree(angle=Decimal(str(random.uniform(0.0, 360.0)))) for _ in range(num_to_add)]
        if not placed_trees:
            placed_trees.append(unplaced.pop(0))  # first at origin

        for tree_to_place in unplaced:
            placed_polygons = [t.polygon for t in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = Decimal("0")
            best_py = Decimal("0")
            min_radius = Decimal("Infinity")

            for _ in range(max(1, attempts)):
                a = generate_weighted_angle()
                vx = Decimal(str(math.cos(a)))
                vy = Decimal(str(math.sin(a)))

                radius = start_radius
                collision_found = False

                while radius >= 0:
                    px = radius * vx
                    py = radius * vy
                    px, py = _clamp_xy(px, py)

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * SCALE_FACTOR),
                        yoff=float(py * SCALE_FACTOR),
                    )
                    if _has_overlap(candidate_poly, tree_index, placed_polygons):
                        collision_found = True
                        break
                    radius -= step_in

                if collision_found:
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy
                        px, py = _clamp_xy(px, py)
                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * SCALE_FACTOR),
                            yoff=float(py * SCALE_FACTOR),
                        )
                        if not _has_overlap(candidate_poly, tree_index, placed_polygons):
                            break
                else:
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * SCALE_FACTOR),
                yoff=float(tree_to_place.center_y * SCALE_FACTOR),
            )
            placed_trees.append(tree_to_place)

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds
    side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    side_length = Decimal(str(side_length_scaled)) / SCALE_FACTOR
    return placed_trees, side_length


def _format_s(v: float) -> str:
    return f"s{v:.6f}"


def _format_s_prec(v: float, *, decimals: int = 12) -> str:
    return f"s{v:.{decimals}f}"


def _require_numpy():
    if np is None:  # pragma: no cover
        raise SystemExit("Missing dependency: numpy\nInstall with: pip install numpy")
    return np


def _parse_s_floats(col: pd.Series):
    npx = _require_numpy()
    s = col.astype(str).str.strip()
    if not s.str.startswith("s").all():
        bad = s[~s.str.startswith("s")].head(3).tolist()
        raise ValueError(f"Expected 's' prefix, got e.g. {bad}")
    return s.str[1:].astype(float).to_numpy(dtype=npx.float64, copy=False)


def _tree_pointcloud_xy(x, y, deg):
    npx = _require_numpy()
    base = npx.asarray(TREE_VERTS, dtype=npx.float64)
    vx = base[:, 0][None, :]
    vy = base[:, 1][None, :]

    a = npx.deg2rad(deg).astype(npx.float64, copy=False)
    c = npx.cos(a)[:, None]
    s = npx.sin(a)[:, None]

    X = x[:, None] + c * vx - s * vy
    Y = y[:, None] + s * vx + c * vy
    return X.reshape(-1), Y.reshape(-1)


def _side_from_points(X, Y) -> float:
    return float(max(X.max() - X.min(), Y.max() - Y.min()))


def _side_for_phis(X, Y, phis):
    npx = _require_numpy()
    phis = npx.asarray(phis, dtype=npx.float64)
    c = npx.cos(phis)[:, None]
    s = npx.sin(phis)[:, None]

    Xr = X[None, :] * c - Y[None, :] * s
    Yr = X[None, :] * s + Y[None, :] * c
    w = Xr.max(axis=1) - Xr.min(axis=1)
    h = Yr.max(axis=1) - Yr.min(axis=1)
    return npx.maximum(w, h)


def _best_rotation_phi_rad(X, Y, *, coarse_steps: int = 180, refine_steps: int = 200):
    npx = _require_numpy()
    period = math.pi / 2.0
    phis = npx.linspace(0.0, period, num=coarse_steps, endpoint=False, dtype=npx.float64)
    sides = _side_for_phis(X, Y, phis)
    best_i = int(sides.argmin())
    best_phi = float(phis[best_i])
    best_side = float(sides[best_i])

    window = period / coarse_steps
    phis2 = best_phi + npx.linspace(-window, window, num=refine_steps, dtype=npx.float64)
    phis2 = npx.mod(phis2, period)
    sides2 = _side_for_phis(X, Y, phis2)
    best2_i = int(sides2.argmin())
    best2_phi = float(phis2[best2_i])
    best2_side = float(sides2[best2_i])
    if best2_side < best_side:
        return best2_phi, best2_side
    return best_phi, best_side


def _shift_into_bounds(values, *, low: float = -100.0, high: float = 100.0):
    npx = _require_numpy()
    values = npx.asarray(values, dtype=npx.float64)
    mn = float(values.min())
    mx = float(values.max())
    lo = low - mn
    hi = high - mx
    if lo > hi:
        raise ValueError("Cannot shift values into [-100, 100].")
    if lo > 0:
        shift = lo
    elif hi < 0:
        shift = hi
    else:
        shift = 0.0
    return values + shift, float(shift)


def _rotate_group_df(df_group: pd.DataFrame, phi_rad: float, *, decimals: int = 12) -> pd.DataFrame:
    npx = _require_numpy()
    x = _parse_s_floats(df_group["x"])
    y = _parse_s_floats(df_group["y"])
    deg = _parse_s_floats(df_group["deg"])

    c = math.cos(phi_rad)
    s = math.sin(phi_rad)
    x2 = x * c - y * s
    y2 = x * s + y * c
    deg2 = deg + math.degrees(phi_rad)

    x2, _sx = _shift_into_bounds(x2)
    y2, _sy = _shift_into_bounds(y2)

    out = df_group.copy()
    out["x"] = [_format_s_prec(v, decimals=decimals) for v in x2.tolist()]
    out["y"] = [_format_s_prec(v, decimals=decimals) for v in y2.tolist()]
    out["deg"] = [_format_s_prec(v, decimals=decimals) for v in deg2.tolist()]
    return out


def _group_side_and_best_phi(df_group: pd.DataFrame, *, allow_rotate: bool) -> Tuple[float, float]:
    x = _parse_s_floats(df_group["x"])
    y = _parse_s_floats(df_group["y"])
    deg = _parse_s_floats(df_group["deg"])
    X, Y = _tree_pointcloud_xy(x, y, deg)
    if not allow_rotate:
        return _side_from_points(X, Y), 0.0
    phi, side = _best_rotation_phi_rad(X, Y)
    return side, phi


def _read_submission_groups(path: str) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path, dtype=str)
    df = normalize_submission_df(df, path_hint=path)
    df["_group"] = df["id"].astype(str).str.split("_").str[0]
    groups: Dict[str, pd.DataFrame] = {}
    for g, sub in df.groupby("_group", sort=True):
        groups[str(g)] = sub.sort_values("id").reset_index(drop=True)
    return groups


def ensemble_best_by_group(
    paths: List[str],
    *,
    fix_direction: bool,
    decimals: int = 12,
    verbose: bool = True,
) -> pd.DataFrame:
    if not paths:
        raise ValueError("No input submissions provided.")

    inputs = [(p, _read_submission_groups(p)) for p in paths]
    all_groups = sorted({g for _p, mp in inputs for g in mp.keys()}, key=lambda s: int(s))

    chosen: List[pd.DataFrame] = []
    counts: Dict[str, int] = {p: 0 for p in paths}
    for g in all_groups:
        best_path = None
        best_side = float("inf")
        best_phi = 0.0
        best_df = None

        for path, mp in inputs:
            if g not in mp:
                continue
            df_g = mp[g]
            side, phi = _group_side_and_best_phi(df_g, allow_rotate=fix_direction)
            if side < best_side:
                best_side = side
                best_phi = phi
                best_path = path
                best_df = df_g

        if best_df is None:
            raise ValueError(f"Missing group {g} in all inputs.")

        if fix_direction and best_phi != 0.0:
            best_df = _rotate_group_df(best_df, best_phi, decimals=decimals)

        chosen.append(best_df[["id", "x", "y", "deg"]])
        counts[best_path] += 1

    out = pd.concat(chosen, ignore_index=True).sort_values("id").reset_index(drop=True)
    out = out[["id", "x", "y", "deg"]]

    if verbose:
        used = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        msg = ", ".join([f"{Path(p).name}:{n}" for p, n in used if n])
        print(f"Ensembled groups from: {msg}")
    return out


def _tree_bounds_per_tree(df_group: pd.DataFrame):
    npx = _require_numpy()
    x = _parse_s_floats(df_group["x"])
    y = _parse_s_floats(df_group["y"])
    deg = _parse_s_floats(df_group["deg"])

    base = npx.asarray(TREE_VERTS, dtype=npx.float64)
    vx = base[:, 0][None, :]
    vy = base[:, 1][None, :]

    a = npx.deg2rad(deg).astype(npx.float64, copy=False)
    c = npx.cos(a)[:, None]
    s = npx.sin(a)[:, None]
    X = x[:, None] + c * vx - s * vy
    Y = y[:, None] + s * vx + c * vy
    return X.min(axis=1), X.max(axis=1), Y.min(axis=1), Y.max(axis=1)


def _delete_propagate_groups(groups: Dict[int, pd.DataFrame], *, verbose: bool = True) -> Dict[int, pd.DataFrame]:
    npx = _require_numpy()
    n_max = max(groups.keys())
    improved = 0
    for n in range(n_max, 1, -1):
        if (n - 1) not in groups:
            continue

        main = groups[n].reset_index(drop=True)
        prev = groups[n - 1].reset_index(drop=True)

        prev_side, _ = _group_side_and_best_phi(prev, allow_rotate=False)

        minx, maxx, miny, maxy = _tree_bounds_per_tree(main)
        sorted_minx = npx.sort(minx)
        sorted_maxx = npx.sort(maxx)
        sorted_miny = npx.sort(miny)
        sorted_maxy = npx.sort(maxy)

        g_minx = float(sorted_minx[0])
        g_maxx = float(sorted_maxx[-1])
        g_miny = float(sorted_miny[0])
        g_maxy = float(sorted_maxy[-1])

        s_minx = float(sorted_minx[1])
        s_maxx = float(sorted_maxx[-2])
        s_miny = float(sorted_miny[1])
        s_maxy = float(sorted_maxy[-2])

        tol = 1e-12
        cand_mask = (
            (npx.abs(minx - g_minx) <= tol)
            | (npx.abs(maxx - g_maxx) <= tol)
            | (npx.abs(miny - g_miny) <= tol)
            | (npx.abs(maxy - g_maxy) <= tol)
        )
        cand_idx = npx.nonzero(cand_mask)[0].tolist()

        best_i = None
        best_side = prev_side
        for i in cand_idx:
            nx0 = s_minx if abs(float(minx[i]) - g_minx) <= tol else g_minx
            nx1 = s_maxx if abs(float(maxx[i]) - g_maxx) <= tol else g_maxx
            ny0 = s_miny if abs(float(miny[i]) - g_miny) <= tol else g_miny
            ny1 = s_maxy if abs(float(maxy[i]) - g_maxy) <= tol else g_maxy
            side = max(nx1 - nx0, ny1 - ny0)
            if side < best_side:
                best_side = side
                best_i = int(i)

        if best_i is not None:
            improved += 1
            groups[n - 1] = main.drop(index=best_i).reset_index(drop=True)[["x", "y", "deg"]]
            if verbose:
                print(f"delete-prop n={n:03d}: {prev_side:.9f} -> {best_side:.9f}")

    if verbose:
        print(f"delete-prop improved groups: {improved}")
    return groups


def _group_has_overlap(df_group: pd.DataFrame) -> bool:
    g = df_group.reset_index(drop=True)
    x = _parse_s_floats(g["x"])
    y = _parse_s_floats(g["y"])
    deg = _parse_s_floats(g["deg"])
    polys = [_tree_polygon_scaled(float(x[i]), float(y[i]), float(deg[i])) for i in range(len(g))]
    r_tree = STRtree(polys)
    for i, poly in enumerate(polys):
        for j in r_tree.query(poly):
            if j == i:
                continue
            other = polys[j]
            if poly.intersects(other) and not poly.touches(other):
                return True
    return False


def _prune_group_to_n_min_side(df_group: pd.DataFrame, *, target_n: int) -> pd.DataFrame:
    """
    Select a subset of trees (rows) of size target_n to minimize the bounding square side.

    This is a greedy "remove extreme contributor" heuristic. It's safe w.r.t overlaps because
    removing trees cannot create overlaps.
    """
    _require_numpy()
    target_n = int(target_n)
    g = df_group.reset_index(drop=True)[["x", "y", "deg"]].copy()
    if target_n <= 0:
        return g.iloc[:0].copy()
    if len(g) <= target_n:
        return g

    npx = _require_numpy()
    tol = 1e-12
    while len(g) > target_n:
        minx, maxx, miny, maxy = _tree_bounds_per_tree(g)
        g_minx = float(minx.min())
        g_maxx = float(maxx.max())
        g_miny = float(miny.min())
        g_maxy = float(maxy.max())

        # Candidate removals: any tree that touches a current extreme in X or Y.
        cand = set()
        cand.update(npx.where(npx.abs(minx - g_minx) <= tol)[0].tolist())
        cand.update(npx.where(npx.abs(maxx - g_maxx) <= tol)[0].tolist())
        cand.update(npx.where(npx.abs(miny - g_miny) <= tol)[0].tolist())
        cand.update(npx.where(npx.abs(maxy - g_maxy) <= tol)[0].tolist())
        if not cand:
            cand = set(range(len(g)))

        best_i = None
        best_side = float("inf")
        for i in cand:
            mask = npx.ones(len(g), dtype=bool)
            mask[int(i)] = False
            g2_minx = float(minx[mask].min())
            g2_maxx = float(maxx[mask].max())
            g2_miny = float(miny[mask].min())
            g2_maxy = float(maxy[mask].max())
            side = max(g2_maxx - g2_minx, g2_maxy - g2_miny)
            if side < best_side:
                best_side = side
                best_i = int(i)

        if best_i is None:
            best_i = 0

        g = g.drop(index=best_i).reset_index(drop=True)

    return g[["x", "y", "deg"]]


def _suffix_min_propagate_groups(
    groups: Dict[int, pd.DataFrame],
    *,
    mode: str,
    min_improve: float,
    decimals: int,
    verbose: bool,
) -> Dict[int, pd.DataFrame]:
    """
    If a larger-N configuration fits in a smaller (or equal) square, then the smaller-N is "suboptimal":
    you can reuse the larger layout and simply drop trees.

    This pass enforces: side(n) <= min_{k>=n} side(k) by copying from a suffix best group.

    mode:
      - "prefix": take first n rows from best larger group (matches many incremental submissions).
      - "prune": greedily remove trees to reach n with smaller side.
    """
    _require_numpy()
    if not groups:
        return groups

    mode = str(mode or "prefix").strip().lower()
    if mode not in ("prefix", "prune"):
        raise ValueError(f"Unknown suffix-min mode: {mode}")

    decimals = max(16, int(decimals))
    min_improve = float(min_improve)

    n_max = max(groups.keys())
    sides = {n: _group_side(groups[n]) for n in groups.keys()}

    suffix_best_n: Dict[int, int] = {}
    suffix_best_side: Dict[int, float] = {}
    best_n = n_max
    best_side = float("inf")
    for n in range(n_max, 0, -1):
        if n not in groups:
            continue
        s = float(sides[n])
        if s < best_side - min_improve:
            best_side = s
            best_n = n
        suffix_best_n[n] = int(best_n)
        suffix_best_side[n] = float(best_side)

    improved = 0
    for n in sorted(groups.keys()):
        m = suffix_best_n.get(n)
        if m is None or m == n:
            continue
        if suffix_best_side.get(n, float("inf")) >= float(sides[n]) - min_improve:
            continue

        src = groups[m].reset_index(drop=True)[["x", "y", "deg"]]
        if mode == "prefix":
            cand = src.iloc[:n].copy().reset_index(drop=True)
        else:
            cand = _prune_group_to_n_min_side(src, target_n=n)

        # Removing trees cannot introduce overlaps, so no need to validate here.
        old_side = float(sides[n])
        new_side = float(_group_side(cand))
        if new_side < old_side - min_improve:
            groups[n] = cand.reset_index(drop=True)[["x", "y", "deg"]]
            sides[n] = new_side
            improved += 1
            if verbose:
                print(f"suffix-min n={n:03d}: {old_side:.9f} -> {new_side:.9f} (from n={m:03d}, mode={mode})")

    if verbose:
        print(f"suffix-min improved groups: {improved}")
    return groups


def _insert_propagate_groups(
    groups: Dict[int, pd.DataFrame],
    *,
    nmax: int = 60,
    topk: int = 5,
    min_improve: float = 1e-12,
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Best-effort "reverse propagate": try to improve group n+1 using group n by inserting 1 tree.

    This only makes sense because all trees are identical in this competition, so any (n+1)-tree
    arrangement is valid for group n+1. We attempt a small set of insertions from the current
    (n+1)-tree group into the n-tree group's layout and keep the best valid improvement.
    """

    _require_numpy()
    if not groups:
        return groups

    n_max = max(groups.keys())
    n_stop = min(int(nmax), int(n_max) - 1)
    if n_stop <= 0:
        return groups

    topk = max(1, int(topk))
    improved = 0

    for n in range(1, n_stop + 1):
        if n not in groups or (n + 1) not in groups:
            continue

        base = groups[n].reset_index(drop=True)[["x", "y", "deg"]]
        target = groups[n + 1].reset_index(drop=True)[["x", "y", "deg"]]
        target_side = _group_side(target)

        ranked: List[Tuple[float, int]] = []
        for i in range(len(target)):
            cand = pd.concat([base, target.iloc[[i]]], ignore_index=True)
            ranked.append((_group_side(cand), int(i)))
        ranked.sort(key=lambda t: t[0])

        best_df = None
        best_side = target_side
        for cand_side, i in ranked[:topk]:
            if cand_side >= best_side - float(min_improve):
                break
            cand = pd.concat([base, target.iloc[[i]]], ignore_index=True)
            if _group_has_overlap(cand):
                continue
            best_df = cand
            best_side = cand_side
            break

        if best_df is not None:
            improved += 1
            groups[n + 1] = best_df.reset_index(drop=True)[["x", "y", "deg"]]
            if verbose:
                print(f"insert-prop n={n+1:03d}: {target_side:.9f} -> {best_side:.9f} (from n={n:03d})")

    if verbose:
        print(f"insert-prop improved groups: {improved}")
    return groups


def _submission_df_to_groups(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]
    groups: Dict[int, pd.DataFrame] = {}
    for g, sub in tmp.groupby("_group", sort=True):
        groups[int(g)] = sub[["x", "y", "deg"]].reset_index(drop=True)
    return groups


def _groups_to_submission_df(groups: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for n in sorted(groups.keys()):
        g = groups[n].reset_index(drop=True)
        for i, row in g.iterrows():
            rows.append({"id": f"{n:03d}_{i}", "x": row["x"], "y": row["y"], "deg": row["deg"]})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def apply_fix_direction(df: pd.DataFrame, *, decimals: int = 12) -> pd.DataFrame:
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]
    fixed = []
    for _g, sub in tmp.groupby("_group", sort=True):
        sub = sub.sort_values("id").reset_index(drop=True)
        _side, phi = _group_side_and_best_phi(sub, allow_rotate=True)
        if phi != 0.0:
            sub = _rotate_group_df(sub, phi, decimals=decimals)
        fixed.append(sub[["id", "x", "y", "deg"]])
    return pd.concat(fixed, ignore_index=True).sort_values("id").reset_index(drop=True)


def _group_side(df_group: pd.DataFrame) -> float:
    x = _parse_s_floats(df_group["x"])
    y = _parse_s_floats(df_group["y"])
    deg = _parse_s_floats(df_group["deg"])
    X, Y = _tree_pointcloud_xy(x, y, deg)
    return _side_from_points(X, Y)


def _group_score(df_group: pd.DataFrame) -> float:
    side = _group_side(df_group)
    n = max(1, len(df_group))
    return (side * side) / float(n)


def _wrap_degrees(deg):
    npx = _require_numpy()
    deg = npx.asarray(deg, dtype=npx.float64)
    return (deg + 180.0) % 360.0 - 180.0


def _circular_mean_degrees(a_deg, b_deg):
    npx = _require_numpy()
    a = npx.deg2rad(npx.asarray(a_deg, dtype=npx.float64))
    b = npx.deg2rad(npx.asarray(b_deg, dtype=npx.float64))
    c = npx.cos(a) + npx.cos(b)
    s = npx.sin(a) + npx.sin(b)
    out = npx.arctan2(s, c)
    # If the mean vector is ~0, the average is undefined; keep a_deg in that case.
    bad = (npx.abs(c) < 1e-12) & (npx.abs(s) < 1e-12)
    if npx.any(bad):
        out = out.copy()
        out[bad] = a[bad]
    return _wrap_degrees(npx.rad2deg(out))


def _symmetry180_group(df_group: pd.DataFrame, *, decimals: int) -> pd.DataFrame:
    """
    Enforce 180° rotational symmetry by pairing trees around a chosen center and averaging each pair
    to the nearest symmetric configuration. Works best as a "try-and-keep-if-better" operator.

    Input/Output: a per-group df with columns x,y,deg (string values with 's' prefix).
    """
    npx = _require_numpy()
    g = df_group.reset_index(drop=True)[["x", "y", "deg"]].copy()
    n = len(g)
    if n <= 1 or (n % 2) != 0:
        return g

    x = _parse_s_floats(g["x"])
    y = _parse_s_floats(g["y"])
    deg = _parse_s_floats(g["deg"])

    # Symmetry center: bounding-box center of tree centers (cheap + stable).
    xc = 0.5 * (float(x.min()) + float(x.max()))
    yc = 0.5 * (float(y.min()) + float(y.max()))

    ang = npx.arctan2(y - yc, x - xc)
    order = npx.argsort(ang, kind="mergesort")
    half = n // 2
    a_idx = order[:half]
    b_idx = order[half:]

    # Average each pair (a, b) to satisfy: (xb, yb) = (2c - xa, 2c - ya)
    xa = x[a_idx]
    ya = y[a_idx]
    xb = x[b_idx]
    yb = y[b_idx]

    xa_new = 0.5 * (xa + (2.0 * xc - xb))
    ya_new = 0.5 * (ya + (2.0 * yc - yb))
    xb_new = 2.0 * xc - xa_new
    yb_new = 2.0 * yc - ya_new

    # Average angles so that: deg_b = deg_a + 180 (mod 360)
    da = deg[a_idx]
    db = deg[b_idx]
    da_new = _circular_mean_degrees(da, _wrap_degrees(db - 180.0))
    db_new = _wrap_degrees(da_new + 180.0)

    x2 = x.copy()
    y2 = y.copy()
    d2 = deg.copy()
    x2[a_idx] = xa_new
    y2[a_idx] = ya_new
    d2[a_idx] = da_new
    x2[b_idx] = xb_new
    y2[b_idx] = yb_new
    d2[b_idx] = db_new

    x2, _sx = _shift_into_bounds(x2)
    y2, _sy = _shift_into_bounds(y2)

    out = g.copy()
    out["x"] = [_format_s_prec(v, decimals=decimals) for v in x2.tolist()]
    out["y"] = [_format_s_prec(v, decimals=decimals) for v in y2.tolist()]
    out["deg"] = [_format_s_prec(v, decimals=decimals) for v in d2.tolist()]
    return out[["x", "y", "deg"]]


def _symmetry180_try_improve_groups(
    groups: Dict[int, pd.DataFrame],
    *,
    nmax: int,
    exclude: Optional[set[int]],
    min_improve: float,
    decimals: int,
    verbose: bool,
) -> Dict[int, pd.DataFrame]:
    _require_numpy()
    exclude = exclude or set()
    improved = 0
    for n, g in list(groups.items()):
        if n > int(nmax):
            continue
        if (n % 2) != 0:
            continue
        if n in exclude:
            continue
        old_side = _group_side(g)
        cand = _symmetry180_group(g, decimals=decimals)
        if _group_has_overlap(cand):
            continue
        new_side = _group_side(cand)
        if new_side < old_side - float(min_improve):
            groups[n] = cand.reset_index(drop=True)[["x", "y", "deg"]]
            improved += 1
            if verbose:
                print(f"sym180 n={n:03d}: {old_side:.9f} -> {new_side:.9f}")
    if verbose:
        print(f"sym180 improved groups: {improved}")
    return groups


def _sa_optimize_group(
    df_group: pd.DataFrame,
    *,
    steps: int,
    restarts: int,
    t0: float,
    alpha: float,
    move_xy: float,
    move_deg: float,
    decimals: int,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    if steps <= 0:
        return df_group

    decimals = max(16, int(decimals))
    g0 = df_group[["id", "x", "y", "deg"]].copy().reset_index(drop=True)
    x0 = _parse_s_floats(g0["x"])
    y0 = _parse_s_floats(g0["y"])
    deg0 = _parse_s_floats(g0["deg"])

    best_x = x0.copy()
    best_y = y0.copy()
    best_deg = deg0.copy()
    start_side = _group_side(g0)
    best_side = start_side

    # Pre-build polygons for fast overlap tests. Scaled geometry matches the metric.
    polys0 = [_tree_polygon_scaled(float(x0[i]), float(y0[i]), float(deg0[i])) for i in range(len(g0))]

    for r in range(max(1, int(restarts))):
        rng = random.Random(seed + r * 1_000_003)

        x = best_x.copy()
        y = best_y.copy()
        deg = best_deg.copy()
        polys = list(polys0) if r == 0 else [_tree_polygon_scaled(float(x[i]), float(y[i]), float(deg[i])) for i in range(len(g0))]

        cur_side = _side_from_points(*_tree_pointcloud_xy(x, y, deg))
        temp = max(1e-9, float(t0))
        t0_eff = temp
        accepted = 0
        feasible = 0

        for _step in range(int(steps)):
            i = rng.randrange(len(g0))

            # Occasional big kick helps escape a tiny local basin.
            kick = 5.0 if rng.random() < 0.02 else 1.0
            scale = max(0.15, temp / t0_eff) * kick

            do_move = rng.random() < 0.85
            do_rot = rng.random() < 0.60
            if not (do_move or do_rot):
                do_move = True

            nx = float(x[i])
            ny = float(y[i])
            na = float(deg[i])

            if do_move:
                nx += (rng.random() * 2.0 - 1.0) * move_xy * scale
                ny += (rng.random() * 2.0 - 1.0) * move_xy * scale
                nx = min(XY_MAX, max(XY_MIN, nx))
                ny = min(XY_MAX, max(XY_MIN, ny))
            if do_rot:
                na = (na + (rng.random() * 2.0 - 1.0) * move_deg * scale) % 360.0

            cand_poly = _tree_polygon_scaled(nx, ny, na)
            ok = True
            for j, other in enumerate(polys):
                if j == i:
                    continue
                if cand_poly.intersects(other) and not cand_poly.touches(other):
                    ok = False
                    break
            if not ok:
                temp *= alpha
                continue

            feasible += 1
            ox, oy, oa = float(x[i]), float(y[i]), float(deg[i])

            x[i] = nx
            y[i] = ny
            deg[i] = na
            new_side = _side_from_points(*_tree_pointcloud_xy(x, y, deg))
            delta = new_side - cur_side

            accept = delta <= 0.0 or rng.random() < math.exp(-delta / max(1e-9, temp))
            if accept:
                accepted += 1
                cur_side = new_side
                polys[i] = cand_poly
                if cur_side < best_side:
                    best_side = cur_side
                    best_x = x.copy()
                    best_y = y.copy()
                    best_deg = deg.copy()
            else:
                x[i] = ox
                y[i] = oy
                deg[i] = oa

            temp *= alpha

        if verbose:
            ar = accepted / max(1, steps)
            fr = feasible / max(1, steps)
            print(
                f"  restart {r+1}/{max(1,int(restarts))}: best_side={best_side:.6f} accept={ar:.2%} feasible={fr:.2%}"
            )

    if best_side >= start_side - 1e-12:
        return g0

    out = g0.copy()
    out["x"] = [_format_s_prec(float(v), decimals=decimals) for v in best_x.tolist()]
    out["y"] = [_format_s_prec(float(v), decimals=decimals) for v in best_y.tolist()]
    out["deg"] = [_format_s_prec(float(v) % 360.0, decimals=decimals) for v in best_deg.tolist()]
    return out


def optimize_small_groups_sa(
    df: pd.DataFrame,
    *,
    nmax: int,
    top: int,
    steps: int,
    restarts: int,
    t0: float,
    alpha: float,
    move_xy: float,
    move_deg: float,
    seed: int,
    decimals: int,
    fix_direction: bool,
    verbose: bool,
) -> pd.DataFrame:
    if nmax <= 0:
        return df

    decimals = max(16, int(decimals))
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]
    groups: Dict[int, pd.DataFrame] = {}
    for g, sub in tmp.groupby("_group", sort=True):
        groups[int(g)] = sub.sort_values("id").reset_index(drop=True)[["id", "x", "y", "deg"]]

    candidates = [g for g in groups.keys() if 1 <= g <= nmax]
    if not candidates:
        return df

    if top and top > 0:
        ranked = [(g, _group_score(groups[g])) for g in candidates]
        ranked.sort(key=lambda t: t[1], reverse=True)
        candidates = [g for g, _s in ranked[: min(top, len(ranked))]]

    candidates = sorted(candidates)
    if verbose:
        print(f"SA optimizing groups: {candidates[0]}..{candidates[-1]} (count={len(candidates)})")

    start = time.time()
    for g in candidates:
        gdf = groups[g]
        before_side = _group_side(gdf)
        before_score = (before_side * before_side) / float(len(gdf))
        if verbose:
            print(f"n={g:03d}: start side={before_side:.6f} score={before_score:.6f}")

        opt = _sa_optimize_group(
            gdf,
            steps=steps,
            restarts=restarts,
            t0=t0,
            alpha=alpha,
            move_xy=move_xy,
            move_deg=move_deg,
            decimals=decimals,
            seed=seed + g * 10_000,
            verbose=verbose,
        )
        if fix_direction:
            _side, phi = _group_side_and_best_phi(opt, allow_rotate=True)
            if phi != 0.0:
                opt = _rotate_group_df(opt, phi, decimals=decimals)

        after_side = _group_side(opt)
        after_score = (after_side * after_side) / float(len(opt))
        if verbose:
            dt = (time.time() - start) / 60.0
            print(
                f"n={g:03d}: done side={after_side:.6f} score={after_score:.6f} (delta={after_score-before_score:+.6f}) t={dt:.1f}m"
            )
        groups[g] = opt

    out = pd.concat([groups[g] for g in sorted(groups.keys())], ignore_index=True).sort_values("id").reset_index(drop=True)
    return out[["id", "x", "y", "deg"]]


def build_submission_greedy_prefix(n_max: int, *, seed: int, attempts: int) -> pd.DataFrame:
    random.seed(seed)

    index = [f"{n:03d}_{t}" for n in range(1, n_max + 1) for t in range(n)]

    tree_data: List[Tuple[float, float, float]] = []
    current: List[ChristmasTree] = []
    for n in range(1, n_max + 1):
        current, _side = initialize_trees(n, existing_trees=current, attempts=attempts)
        for tree in current:
            tree_data.append((float(tree.center_x), float(tree.center_y), float(tree.angle)))

    df = pd.DataFrame(index=index, columns=["x", "y", "deg"], data=tree_data).rename_axis("id")
    for col in ["x", "y", "deg"]:
        df[col] = df[col].astype(float).round(6).map(_format_s)
    return df


def _grid_layout(
    n: int,
    *,
    n_even: int,
    n_odd: int,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: Optional[float] = None,
) -> List[Tuple[float, float, float]]:
    if n_odd <= 0 or n_even <= 0:
        return []

    if odd_x_offset is None:
        odd_x_offset = dx / 2.0

    placements: List[Tuple[float, float, float]] = []
    rest = n
    r = 0
    while rest > 0:
        m = min(rest, n_even if (r % 2 == 0) else n_odd)
        rest -= m

        angle = 0.0 if (r % 2 == 0) else 180.0
        x_offset = 0.0 if (r % 2 == 0) else odd_x_offset
        if r % 2 == 0:
            y = (r // 2) * y_step
        else:
            y = y_odd_offset + ((r - 1) // 2) * y_step

        for i in range(m):
            placements.append((dx * i + x_offset, y, angle))
        r += 1

    return placements


def _bounds_for_grid(
    n: int,
    *,
    n_even: int,
    n_odd: int,
    dx: float = 0.7,
    y_step: float = 1.0,
    y_odd_offset: float = 0.8,
    odd_x_offset: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if odd_x_offset is None:
        odd_x_offset = dx / 2.0

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    rest = n
    r = 0
    while rest > 0:
        m = min(rest, n_even if (r % 2 == 0) else n_odd)
        rest -= m

        angle = 0 if (r % 2 == 0) else 180
        x_offset = 0.0 if (r % 2 == 0) else odd_x_offset
        if r % 2 == 0:
            y = (r // 2) * y_step
        else:
            y = y_odd_offset + ((r - 1) // 2) * y_step

        bx0, by0, bx1, by1 = TREE_BOUNDS[angle]
        min_x = min(min_x, x_offset + bx0)
        max_x = max(max_x, x_offset + dx * (m - 1) + bx1)
        min_y = min(min_y, y + by0)
        max_y = max(max_y, y + by1)

        r += 1

    return min_x, min_y, max_x, max_y


def solve_n_grid(n: int, *, span: int = 1) -> List[Tuple[float, float, float]]:
    # Port of public notebook "88.32999 A Well-Aligned Initial Solution".
    best_side2 = float("inf")
    best_pair: Optional[Tuple[int, int]] = None

    for n_even in range(1, n + 1):
        lo = max(1, n_even - span)
        hi = min(n, n_even + span)
        for n_odd in range(lo, hi + 1):
            if n_odd <= 0:
                continue
            min_x, min_y, max_x, max_y = _bounds_for_grid(n, n_even=n_even, n_odd=n_odd)
            side = max(max_x - min_x, max_y - min_y)
            side2 = side * side
            if side2 < best_side2:
                best_side2 = side2
                best_pair = (n_even, n_odd)

    assert best_pair is not None
    return _grid_layout(n, n_even=best_pair[0], n_odd=best_pair[1])


def build_submission_grid(n_max: int, *, span: int = 1) -> pd.DataFrame:
    rows = []
    for n in range(1, n_max + 1):
        placements = solve_n_grid(n, span=span)
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def build_submission_grid_prune(n_max: int, *, span: int = 1) -> pd.DataFrame:
    # Build only n_max using grid, then derive n_max-1..1 by deleting one tree at a time
    # that minimizes the bounding-square side length (cheap and often improves total score).
    current = solve_n_grid(n_max, span=span)
    solutions: Dict[int, List[Tuple[float, float, float]]] = {n_max: current}

    for n in range(n_max - 1, 0, -1):
        prev = solutions[n + 1]
        best_i = 0
        best_side = float("inf")
        for i in range(len(prev)):
            cand = prev[:i] + prev[i + 1 :]
            s = side_length_for(cand)
            if s < best_side:
                best_side = s
                best_i = i
        solutions[n] = prev[:best_i] + prev[best_i + 1 :]

    rows = []
    for n in range(1, n_max + 1):
        placements = solutions[n]
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def build_submission_grid_propagate(n_max: int, *, span: int = 1) -> pd.DataFrame:
    # Start from best-known grid for each n, then try to improve n-1 using n by deletion.
    solutions: Dict[int, List[Tuple[float, float, float]]] = {}
    sides: Dict[int, float] = {}
    for n in range(1, n_max + 1):
        placements = solve_n_grid(n, span=span)
        solutions[n] = placements
        sides[n] = side_length_for(placements)

    for n in range(n_max, 1, -1):
        src = solutions[n]
        best = solutions[n - 1]
        best_side = sides[n - 1]

        for i in range(len(src)):
            cand = src[:i] + src[i + 1 :]
            s = side_length_for(cand)
            if s < best_side:
                best_side = s
                best = cand

        solutions[n - 1] = best
        sides[n - 1] = best_side

    rows = []
    for n in range(1, n_max + 1):
        placements = solutions[n]
        for i, (x, y, deg) in enumerate(placements):
            rows.append({"id": f"{n:03d}_{i}", "x": _format_s(x), "y": _format_s(y), "deg": _format_s(deg)})
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def score_submission(submission: pd.DataFrame) -> float:
    df = submission.copy()
    if "id" in df.columns:
        ids = df["id"].astype(str)
    else:
        ids = df.index.astype(str)
        df = df.reset_index(drop=True)

    data_cols = ["x", "y", "deg"]
    for c in data_cols:
        s = df[c].astype(str)
        if not s.str.startswith("s").all():
            raise ValueError(f"Column {c} contains value(s) without 's' prefix.")
        df[c] = s.str[1:]

    limit = 100.0
    x = df["x"].astype(float)
    y = df["y"].astype(float)
    if (x < -limit).any() or (x > limit).any() or (y < -limit).any() or (y > limit).any():
        raise ValueError("x and/or y values outside the bounds of -100 to 100.")

    groups = ids.str.split("_").str[0]
    df["_group"] = groups.values

    total = Decimal("0.0")
    for group, g in df.groupby("_group", sort=True):
        placed: List[ChristmasTree] = []
        for _, row in g.iterrows():
            placed.append(ChristmasTree(row["x"], row["y"], row["deg"]))

        all_polygons = [t.polygon for t in placed]
        r_tree = STRtree(all_polygons)
        for i, poly in enumerate(all_polygons):
            for j in r_tree.query(poly):
                if j == i:
                    continue
                if poly.intersects(all_polygons[j]) and not poly.touches(all_polygons[j]):
                    raise ValueError(f"Overlapping trees in group {group}")

        bounds = unary_union(all_polygons).bounds
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        group_score = (
            (Decimal(str(side_length_scaled)) ** 2)
            / (SCALE_FACTOR**2)
            / Decimal(str(len(g)))
        )
        total += group_score

    return float(total)


def fast_score_submission(submission: pd.DataFrame) -> float:
    npx = _require_numpy()

    df = submission.copy()
    if "id" in df.columns:
        ids = df["id"].astype(str)
    else:
        ids = df.index.astype(str)
        df = df.reset_index(drop=True)

    for c in ["x", "y", "deg"]:
        df[c] = df[c].astype(str).str.strip()

    x = _parse_s_floats(df["x"])
    y = _parse_s_floats(df["y"])
    deg = _parse_s_floats(df["deg"])

    groups = ids.str.split("_").str[0].to_numpy(dtype=object, copy=False)
    # Sort once so each group's rows are contiguous (fast slicing).
    order = npx.argsort(groups, kind="stable")
    groups = groups[order]
    x = x[order]
    y = y[order]
    deg = deg[order]

    # Find group boundaries.
    change = npx.nonzero(groups[1:] != groups[:-1])[0] + 1
    starts = npx.concatenate([npx.array([0], dtype=npx.int64), change])
    ends = npx.concatenate([change, npx.array([len(groups)], dtype=npx.int64)])

    total = 0.0
    for s, e in zip(starts.tolist(), ends.tolist()):
        n = e - s
        X, Y = _tree_pointcloud_xy(x[s:e], y[s:e], deg[s:e])
        side = _side_from_points(X, Y)
        total += (side * side) / float(n)
    return float(total)


def _iter_submission_csvs(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def normalize_submission_df(df: pd.DataFrame, *, path_hint: str = "<dataframe>") -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    id_col = cols.get("id")
    if id_col is None:
        raise ValueError(f"{path_hint}: missing id column. Found columns: {list(df.columns)}")

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    x_col = pick(["x", "pos_x", "px"])
    y_col = pick(["y", "pos_y", "py"])
    deg_col = pick(["deg", "angle", "rotation", "theta"])
    if x_col is None or y_col is None or deg_col is None:
        raise ValueError(
            f"{path_hint}: cannot find x/y/deg columns. Found columns: {list(df.columns)}"
        )

    out = df[[id_col, x_col, y_col, deg_col]].copy()
    out.columns = ["id", "x", "y", "deg"]

    for c in ["x", "y", "deg"]:
        s = out[c].astype(str).str.strip()
        s = s.where(s.str.startswith("s"), "s" + s)
        out[c] = s

    return out[["id", "x", "y", "deg"]]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-in",
        default=None,
        help="Read a CSV and write a Kaggle-ready submission (id,x,y,deg) to --out, then exit.",
    )
    parser.add_argument(
        "--score-file",
        default=None,
        help="Score an existing submission CSV (id,x,y,deg with 's' prefixes) and exit.",
    )
    parser.add_argument(
        "--score-dir",
        default=None,
        help="Score every .csv under a directory (fast bbox-only scoring, no overlap validation).",
    )
    parser.add_argument("--top", type=int, default=20, help="With --score-dir, show top N results (default 20).")
    parser.add_argument(
        "--validate-top",
        type=int,
        default=5,
        help="With --score-dir, also run full overlap-validating score on top K (default 5).",
    )
    parser.add_argument(
        "--ensemble-in",
        nargs="+",
        default=None,
        help="Merge one or more submissions by picking the best group from any input, then write to --out.",
    )
    parser.add_argument(
        "--ensemble-dir",
        default=None,
        help="Recursively include all .csv under a directory for ensembling (optionally filtered by --ensemble-limit).",
    )
    parser.add_argument(
        "--ensemble-limit",
        type=int,
        default=50,
        help="With --ensemble-dir, only keep the best N submissions by fast score (default 50).",
    )
    parser.add_argument(
        "--fix-direction",
        action="store_true",
        help="Rotate each group globally to minimize its axis-aligned bounding square (works with --ensemble-in).",
    )
    parser.add_argument(
        "--symmetry180",
        action="store_true",
        help=(
            "Try enforcing 180° rotational symmetry on even-n groups and keep only improvements "
            "(best-effort; can be combined with --fix-direction / propagate tricks)."
        ),
    )
    parser.add_argument(
        "--symmetry180-nmax",
        type=int,
        default=200,
        help="With --symmetry180, only attempt groups up to this n (default 200).",
    )
    parser.add_argument(
        "--symmetry180-exclude",
        nargs="*",
        type=int,
        default=[],
        help="With --symmetry180, exclude specific group sizes (e.g. 36 64).",
    )
    parser.add_argument(
        "--symmetry180-min-improve",
        type=float,
        default=1e-12,
        help="With --symmetry180, minimum side improvement to accept (default 1e-12).",
    )
    parser.add_argument(
        "--suffix-min-propagate",
        action="store_true",
        help=(
            "If some larger-N group fits in a smaller square than a smaller-N group, copy from the larger "
            "and drop trees to improve the smaller (enforces suffix-min monotonicity of side)."
        ),
    )
    parser.add_argument(
        "--suffix-min-mode",
        choices=["prefix", "prune"],
        default="prefix",
        help="With --suffix-min-propagate, how to drop trees from the larger group (default prefix).",
    )
    parser.add_argument(
        "--suffix-min-improve",
        type=float,
        default=1e-12,
        help="With --suffix-min-propagate, minimum side improvement to accept (default 1e-12).",
    )
    parser.add_argument(
        "--delete-propagate",
        action="store_true",
        help="Try to improve group n-1 by deleting one tree from group n (descending n), a common public trick.",
    )
    parser.add_argument(
        "--insert-propagate",
        action="store_true",
        help="Try to improve group n+1 using group n by inserting 1 tree (best-effort; small-n only).",
    )
    parser.add_argument(
        "--insert-propagate-nmax",
        type=int,
        default=60,
        help="With --insert-propagate, only attempt groups up to this n (default 60).",
    )
    parser.add_argument(
        "--insert-propagate-topk",
        type=int,
        default=5,
        help="With --insert-propagate, validate up to top-K candidate inserts per group (default 5).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=16,
        help="When writing optimized values, number of decimals to keep (default 16).",
    )
    parser.add_argument("--out", default="submission.csv", help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=["grid", "lattice_greedy", "grid_propagate", "grid_prune", "greedy_prefix"],
        default="grid",
        help="Submission generator: grid (strong baseline) or greedy_prefix (getting-started style).",
    )
    parser.add_argument(
        "--grid-span",
        type=int,
        default=1,
        help="Grid mode: allow odd-row length to differ by up to this value from even-row length (default 1).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--nmax", type=int, default=200, help="Max trees (default 200).")
    parser.add_argument("--attempts", type=int, default=10, help="Attempts per added tree.")
    parser.add_argument("--score", action="store_true", help="Compute and print the metric score locally.")
    parser.add_argument(
        "--opt-small-in",
        default=None,
        help="Run simulated annealing (SA) to optimize small groups of an existing submission, then write to --out.",
    )
    parser.add_argument("--opt-small-nmax", type=int, default=30, help="With --opt-small-in, optimize groups 1..N.")
    parser.add_argument(
        "--opt-small-top",
        type=int,
        default=0,
        help="With --opt-small-in, only optimize the worst top-K groups (by group score). 0 = all in 1..N.",
    )
    parser.add_argument("--opt-small-steps", type=int, default=10_000, help="SA steps per group (default 10000).")
    parser.add_argument("--opt-small-restarts", type=int, default=1, help="SA restarts per group (default 1).")
    parser.add_argument("--opt-small-t0", type=float, default=0.05, help="SA initial temperature (default 0.05).")
    parser.add_argument("--opt-small-alpha", type=float, default=0.9995, help="SA cooling factor (default 0.9995).")
    parser.add_argument(
        "--opt-small-move-xy",
        type=float,
        default=0.05,
        help="Max translation delta for a move (default 0.05).",
    )
    parser.add_argument(
        "--opt-small-move-deg",
        type=float,
        default=8.0,
        help="Max rotation delta (degrees) for a move (default 8.0).",
    )
    parser.add_argument("--opt-small-quiet", action="store_true", help="Reduce SA logging.")
    args = parser.parse_args(argv)

    if not (1 <= args.nmax <= 200):
        raise SystemExit("--nmax must be in [1, 200].")

    if args.clean_in is not None:
        src = pd.read_csv(args.clean_in, dtype=str)
        df = normalize_submission_df(src, path_hint=str(args.clean_in))
        if (
            args.delete_propagate
            or args.insert_propagate
            or args.symmetry180
            or args.suffix_min_propagate
            or args.fix_direction
        ):
            groups = _submission_df_to_groups(df)
            if args.delete_propagate:
                groups = _delete_propagate_groups(groups, verbose=True)
            if args.insert_propagate:
                groups = _insert_propagate_groups(
                    groups,
                    nmax=int(args.insert_propagate_nmax),
                    topk=int(args.insert_propagate_topk),
                    verbose=True,
                )
            if args.symmetry180:
                groups = _symmetry180_try_improve_groups(
                    groups,
                    nmax=int(args.symmetry180_nmax),
                    exclude=set(int(x) for x in (args.symmetry180_exclude or [])),
                    min_improve=float(args.symmetry180_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            if args.suffix_min_propagate:
                groups = _suffix_min_propagate_groups(
                    groups,
                    mode=str(args.suffix_min_mode),
                    min_improve=float(args.suffix_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            df = _groups_to_submission_df(groups)
            if args.fix_direction:
                df = apply_fix_direction(df, decimals=args.decimals)
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(df)})")
        if args.score:
            print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.score_file is not None:
        df = pd.read_csv(args.score_file, dtype=str)
        df = normalize_submission_df(df, path_hint=str(args.score_file))
        print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.score_dir is not None:
        root = Path(args.score_dir)
        paths = _iter_submission_csvs(root)
        if not paths:
            raise SystemExit(f"No .csv files under: {root}")

        results = []
        for p in paths:
            try:
                df = pd.read_csv(p, dtype=str)
                df = normalize_submission_df(df, path_hint=str(p))
                s_fast = fast_score_submission(df)
                results.append((s_fast, str(p)))
            except Exception:
                continue

        if not results:
            raise SystemExit(f"No scorable submissions under: {root}")

        results.sort(key=lambda t: t[0])
        top_n = max(1, int(args.top))
        print(f"Found {len(results)} scorable CSVs. Top {min(top_n, len(results))}:")
        for i, (s_fast, p) in enumerate(results[:top_n], 1):
            print(f"{i:02d}. {s_fast:.12f}  {p}")

        k = max(0, int(args.validate_top))
        if k:
            print(f"\nValidating top {min(k, len(results))} with overlap checks:")
            for i, (s_fast, p) in enumerate(results[: min(k, len(results))], 1):
                df = pd.read_csv(p, dtype=str)
                df = normalize_submission_df(df, path_hint=str(p))
                try:
                    s_full = score_submission(df)
                    print(f"{i:02d}. full={s_full:.12f} (fast={s_fast:.12f})  {p}")
                except Exception as e:
                    print(f"{i:02d}. invalid ({e})  {p}")
        return 0

    if args.opt_small_in is not None:
        src = pd.read_csv(args.opt_small_in, dtype=str)
        df = normalize_submission_df(src, path_hint=str(args.opt_small_in))
        df = optimize_small_groups_sa(
            df,
            nmax=int(args.opt_small_nmax),
            top=int(args.opt_small_top),
            steps=int(args.opt_small_steps),
            restarts=int(args.opt_small_restarts),
            t0=float(args.opt_small_t0),
            alpha=float(args.opt_small_alpha),
            move_xy=float(args.opt_small_move_xy),
            move_deg=float(args.opt_small_move_deg),
            seed=int(args.seed),
            decimals=int(args.decimals),
            fix_direction=bool(args.fix_direction),
            verbose=not bool(args.opt_small_quiet),
        )
        if args.delete_propagate or args.insert_propagate or args.symmetry180 or args.suffix_min_propagate:
            groups = _submission_df_to_groups(df)
            if args.delete_propagate:
                groups = _delete_propagate_groups(groups, verbose=True)
            if args.insert_propagate:
                groups = _insert_propagate_groups(
                    groups,
                    nmax=int(args.insert_propagate_nmax),
                    topk=int(args.insert_propagate_topk),
                    verbose=True,
                )
            if args.symmetry180:
                groups = _symmetry180_try_improve_groups(
                    groups,
                    nmax=int(args.symmetry180_nmax),
                    exclude=set(int(x) for x in (args.symmetry180_exclude or [])),
                    min_improve=float(args.symmetry180_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            if args.suffix_min_propagate:
                groups = _suffix_min_propagate_groups(
                    groups,
                    mode=str(args.suffix_min_mode),
                    min_improve=float(args.suffix_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            df = _groups_to_submission_df(groups)
            if args.fix_direction:
                df = apply_fix_direction(df, decimals=args.decimals)

        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(df)})")
        if args.score:
            print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.ensemble_in is not None or args.ensemble_dir is not None:
        paths: List[str] = []
        if args.ensemble_in is not None:
            paths.extend(args.ensemble_in)
        if args.ensemble_dir is not None:
            root = Path(args.ensemble_dir)
            csvs = _iter_submission_csvs(root)
            scored = []
            for p in csvs:
                try:
                    df0 = pd.read_csv(p, dtype=str)
                    df0 = normalize_submission_df(df0, path_hint=str(p))
                    scored.append((fast_score_submission(df0), str(p)))
                except Exception:
                    continue
            scored.sort(key=lambda t: t[0])
            limit = max(1, int(args.ensemble_limit))
            paths.extend([p for _s, p in scored[:limit]])

        # Dedupe while preserving order
        seen = set()
        uniq = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        paths = uniq

        df = ensemble_best_by_group(
            paths,
            fix_direction=args.fix_direction,
            decimals=args.decimals,
            verbose=True,
        )
        if args.delete_propagate or args.insert_propagate or args.symmetry180 or args.suffix_min_propagate:
            groups = _submission_df_to_groups(df)
            if args.delete_propagate:
                groups = _delete_propagate_groups(groups, verbose=True)
            if args.insert_propagate:
                groups = _insert_propagate_groups(
                    groups,
                    nmax=int(args.insert_propagate_nmax),
                    topk=int(args.insert_propagate_topk),
                    verbose=True,
                )
            if args.symmetry180:
                groups = _symmetry180_try_improve_groups(
                    groups,
                    nmax=int(args.symmetry180_nmax),
                    exclude=set(int(x) for x in (args.symmetry180_exclude or [])),
                    min_improve=float(args.symmetry180_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            if args.suffix_min_propagate:
                groups = _suffix_min_propagate_groups(
                    groups,
                    mode=str(args.suffix_min_mode),
                    min_improve=float(args.suffix_min_improve),
                    decimals=int(args.decimals),
                    verbose=True,
                )
            df = _groups_to_submission_df(groups)
            if args.fix_direction:
                df = apply_fix_direction(df, decimals=args.decimals)
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(df)})")
        if args.score:
            print(f"Score: {score_submission(df):.12f}")
        return 0

    out_path = Path(args.out)
    if args.mode == "grid":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid(args.nmax, span=args.grid_span)
    elif args.mode == "lattice_greedy":
        df = build_submission_lattice_greedy(args.nmax)
    elif args.mode == "grid_propagate":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid_propagate(args.nmax, span=args.grid_span)
    elif args.mode == "grid_prune":
        if args.grid_span < 0:
            raise SystemExit("--grid-span must be >= 0.")
        df = build_submission_grid_prune(args.nmax, span=args.grid_span)
    else:
        df = build_submission_greedy_prefix(args.nmax, seed=args.seed, attempts=args.attempts)
    if (
        args.delete_propagate
        or args.insert_propagate
        or args.symmetry180
        or args.suffix_min_propagate
        or args.fix_direction
    ):
        groups = _submission_df_to_groups(df)
        if args.delete_propagate:
            groups = _delete_propagate_groups(groups, verbose=True)
        if args.insert_propagate:
            groups = _insert_propagate_groups(
                groups,
                nmax=int(args.insert_propagate_nmax),
                topk=int(args.insert_propagate_topk),
                verbose=True,
            )
        if args.symmetry180:
            groups = _symmetry180_try_improve_groups(
                groups,
                nmax=int(args.symmetry180_nmax),
                exclude=set(int(x) for x in (args.symmetry180_exclude or [])),
                min_improve=float(args.symmetry180_min_improve),
                decimals=int(args.decimals),
                verbose=True,
            )
        if args.suffix_min_propagate:
            groups = _suffix_min_propagate_groups(
                groups,
                mode=str(args.suffix_min_mode),
                min_improve=float(args.suffix_min_improve),
                decimals=int(args.decimals),
                verbose=True,
            )
        df = _groups_to_submission_df(groups)
        if args.fix_direction:
            df = apply_fix_direction(df, decimals=args.decimals)
    df.to_csv(out_path)
    print(f"Wrote {out_path} (rows={len(df)})")
    if args.score:
        print(f"Score: {score_submission(df):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
