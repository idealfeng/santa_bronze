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
    for c in ["id", "x", "y", "deg"]:
        if c not in df.columns:
            raise ValueError(f"{path}: missing column {c}. Found columns: {list(df.columns)}")
    df = df[["id", "x", "y", "deg"]].copy()
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
        "--ensemble-in",
        nargs="+",
        default=None,
        help="Merge one or more submissions by picking the best group from any input, then write to --out.",
    )
    parser.add_argument(
        "--fix-direction",
        action="store_true",
        help="Rotate each group globally to minimize its axis-aligned bounding square (works with --ensemble-in).",
    )
    parser.add_argument(
        "--delete-propagate",
        action="store_true",
        help="Try to improve group n-1 by deleting one tree from group n (descending n), a common public trick.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=12,
        help="When writing optimized values, number of decimals to keep (default 12).",
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
    args = parser.parse_args(argv)

    if not (1 <= args.nmax <= 200):
        raise SystemExit("--nmax must be in [1, 200].")

    if args.clean_in is not None:
        src = pd.read_csv(args.clean_in)
        df = src[["id", "x", "y", "deg"]].copy()
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} (rows={len(df)})")
        if args.score:
            print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.score_file is not None:
        df = pd.read_csv(args.score_file)
        df = df[["id", "x", "y", "deg"]].copy()
        print(f"Score: {score_submission(df):.12f}")
        return 0

    if args.ensemble_in is not None:
        df = ensemble_best_by_group(
            args.ensemble_in,
            fix_direction=args.fix_direction,
            decimals=args.decimals,
            verbose=True,
        )
        if args.delete_propagate:
            groups = _submission_df_to_groups(df)
            groups = _delete_propagate_groups(groups, verbose=True)
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
    df.to_csv(out_path)
    print(f"Wrote {out_path} (rows={len(df)})")
    if args.score:
        print(f"Score: {score_submission(df):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
