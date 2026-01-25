from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# When executed as `python tools/extract_from_images.py`, sys.path[0] is `tools/`,
# so add repo root to import the local `train.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import train

try:  # optional but strongly recommended for robust connected components
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ExtractedGroup:
    n: int
    source: str
    df_group: pd.DataFrame  # columns: id,x,y,deg (all strings with 's' prefix)
    group_score: float  # S^2 / n


def _parse_n_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else None


def _quantize_rgb(img_rgb: np.ndarray, q: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize RGB image to reduce color variation.

    Returns:
      key: HxW uint32 packed key
      uniq: unique keys
      counts: counts per unique key
    """
    if q <= 1:
        q = 1
    qimg = (img_rgb // q).astype(np.uint16)
    # pack 5 bits per channel when q=8 (0..31) but works for other q too as long as <= 256.
    key = (qimg[:, :, 0].astype(np.uint32) << 16) | (qimg[:, :, 1].astype(np.uint32) << 8) | qimg[:, :, 2].astype(
        np.uint32
    )
    uniq, counts = np.unique(key.reshape(-1), return_counts=True)
    return key, uniq, counts


def _key_to_rgb(key: int, q: int) -> Tuple[int, int, int]:
    r = (key >> 16) & 0xFF
    g = (key >> 8) & 0xFF
    b = key & 0xFF
    # map back to representative (center of bin)
    return (int(r * q + q // 2), int(g * q + q // 2), int(b * q + q // 2))


def _is_background(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r >= 240 and g >= 240 and b >= 240


def _is_red_ui(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r >= 180 and g <= 90 and b <= 90


def _is_black_ui(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r <= 70 and g <= 70 and b <= 70


def _pick_tree_keys(
    *,
    uniq: np.ndarray,
    counts: np.ndarray,
    q: int,
    n_expected: int,
    min_count: int,
    max_count: int,
) -> List[int]:
    order = np.argsort(-counts)
    picked: List[int] = []
    for idx in order:
        k = int(uniq[idx])
        c = int(counts[idx])
        if c < min_count:
            break
        if c > max_count:
            continue
        rgb = _key_to_rgb(k, q=q)
        if _is_background(rgb) or _is_red_ui(rgb) or _is_black_ui(rgb):
            continue
        picked.append(k)
        if len(picked) >= n_expected:
            break
    return picked


def _pca_axis(points_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA axis for points in (row, col) coordinates.
    Returns (mean_rc, axis_rc_unit)
    """
    pts = points_rc.astype(np.float64, copy=False)
    mean = pts.mean(axis=0)
    x = pts - mean
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return mean, axis


def _endpoint_spread(points_rc: np.ndarray, mean_rc: np.ndarray, axis_rc: np.ndarray, t_edge: float, *, frac: float) -> float:
    x = points_rc.astype(np.float64, copy=False) - mean_rc[None, :]
    t = x @ axis_rc
    t_min = float(np.quantile(t, 0.005))
    t_max = float(np.quantile(t, 0.995))
    rng = max(1e-9, t_max - t_min)
    if t_edge < 0.5 * (t_min + t_max):
        mask = t <= (t_min + frac * rng)
    else:
        mask = t >= (t_max - frac * rng)
    if not np.any(mask):
        mask = np.ones(len(points_rc), dtype=bool)
    perp = np.array([-axis_rc[1], axis_rc[0]], dtype=np.float64)
    u = x[mask] @ perp
    return float(np.std(u))


def _estimate_tree_pose(points_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate (origin_rc_at_y0, tip_rc) in pixel coordinates.

    In train.py geometry the tree's local origin is at the base center (y=0):
      - tip is at y=+0.8
      - trunk bottom is at y=-0.2

    For rendered images we cannot reliably depend on trunk pixels, so we locate y=0 by finding
    the position along the PCA axis where the perpendicular width is maximal (the base is widest).
    """
    mean_rc, axis_rc = _pca_axis(points_rc)
    x = points_rc.astype(np.float64, copy=False) - mean_rc[None, :]
    t = x @ axis_rc
    perp = np.array([-axis_rc[1], axis_rc[0]], dtype=np.float64)
    u = x @ perp

    # Robust ends (avoid anti-alias halo / tiny outliers).
    t_min = float(np.quantile(t, 0.01))
    t_max = float(np.quantile(t, 0.99))
    t_rng = max(1e-9, t_max - t_min)

    nbins = 41
    win = max(2.0, 0.04 * t_rng)
    best_t0 = 0.5 * (t_min + t_max)
    best_w = -1.0
    for t0 in np.linspace(t_min, t_max, nbins):
        m = np.abs(t - float(t0)) <= win
        if int(np.sum(m)) < 40:
            continue
        uu = u[m]
        w = float(np.quantile(uu, 0.98) - np.quantile(uu, 0.02))
        if w > best_w:
            best_w = w
            best_t0 = float(t0)

    origin_rc = mean_rc + best_t0 * axis_rc

    # Tip is the farther endpoint from the origin along the axis (distance ≈ 0.8 vs 0.2).
    t_tip = t_max if (t_max - best_t0) >= (best_t0 - t_min) else t_min
    tip_rc = mean_rc + float(t_tip) * axis_rc
    return origin_rc, tip_rc


def _estimate_tree_endpoints(points_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate (end0_rc, end1_rc, length_px) along the main axis using robust percentiles.
    The segment ideally corresponds to trunk-bottom (-0.2) <-> tip (+0.8), i.e. total height 1.0.
    """
    mean_rc, axis_rc = _pca_axis(points_rc)
    x = points_rc.astype(np.float64, copy=False) - mean_rc[None, :]
    t = x @ axis_rc
    t0 = float(np.quantile(t, 0.01))
    t1 = float(np.quantile(t, 0.99))
    p0 = mean_rc + t0 * axis_rc
    p1 = mean_rc + t1 * axis_rc
    return p0, p1, float(np.linalg.norm(p1 - p0))


def _deg_from_origin_tip(origin_rc: np.ndarray, tip_rc: np.ndarray) -> float:
    """
    Return deg such that a canonical tree (tip along +Y) rotated by deg points tip toward (origin->tip).
    """
    dr = float(tip_rc[0] - origin_rc[0])
    dc = float(tip_rc[1] - origin_rc[1])
    dx = dc
    dy = -dr
    ang = math.degrees(math.atan2(dy, dx)) - 90.0
    ang = (ang + 180.0) % 360.0 - 180.0
    return float(ang)


_TREE_VERTS_XY = np.asarray(train.TREE_VERTS, dtype=np.float64)  # (V,2) in solver coords (x right, y up)


def _edge_sample_points_xy(*, points_per_edge: int = 16) -> np.ndarray:
    verts = _TREE_VERTS_XY
    pts: List[np.ndarray] = []
    for i in range(len(verts)):
        a = verts[i]
        b = verts[(i + 1) % len(verts)]
        for t in np.linspace(0.0, 1.0, max(2, int(points_per_edge)), endpoint=False):
            pts.append(a + float(t) * (b - a))
    return np.stack(pts, axis=0)


_TREE_EDGE_SAMPLES_XY = _edge_sample_points_xy(points_per_edge=16)


def _render_tree_poly_points_rc(origin_rc: np.ndarray, deg: float, *, px_per_unit: float) -> np.ndarray:
    """
    Return polygon vertices in pixel (row,col) coordinates for a tree at origin_rc with rotation deg.
    """
    a = math.radians(float(deg))
    c = math.cos(a)
    s = math.sin(a)
    x = _TREE_VERTS_XY[:, 0]
    y = _TREE_VERTS_XY[:, 1]
    xr = x * c - y * s
    yr = x * s + y * c
    col = float(origin_rc[1]) + xr * float(px_per_unit)
    row = float(origin_rc[0]) - yr * float(px_per_unit)
    return np.stack([row, col], axis=1)


def _pose_refine_template(
    mask_u8: np.ndarray,
    origin_rc: np.ndarray,
    deg: float,
    *,
    px_per_unit: float,
) -> Tuple[np.ndarray, float]:
    """
    Refine (origin_rc,deg) by minimizing distance between the rendered tree boundary and the mask boundary.
    """
    if cv2 is None:
        return origin_rc, float(deg)

    mask_bool = mask_u8.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(ys) < 50:
        return origin_rc, float(deg)

    pad = 18
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(mask_u8.shape[0], int(ys.max()) + 1 + pad)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(mask_u8.shape[1], int(xs.max()) + 1 + pad)
    mask_roi_u8 = mask_u8[r0:r1, c0:c1]
    mask_roi = mask_roi_u8.astype(bool)
    h, w = mask_roi_u8.shape
    if h <= 4 or w <= 4:
        return origin_rc, float(deg)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(mask_roi_u8, cv2.MORPH_GRADIENT, kernel, iterations=1)
    edge_bin = (edge > 0).astype(np.uint8)
    dist = cv2.distanceTransform((1 - edge_bin).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    samples_xy = _TREE_EDGE_SAMPLES_XY  # (M,2) in local (x,y)

    def cost(o_rc: np.ndarray, a_deg: float) -> float:
        a = math.radians(float(a_deg))
        c = math.cos(a)
        s = math.sin(a)
        x = samples_xy[:, 0]
        y = samples_xy[:, 1]
        xr = x * c - y * s
        yr = x * s + y * c
        col = (float(o_rc[1]) - float(c0)) + xr * float(px_per_unit)
        row = (float(o_rc[0]) - float(r0)) - yr * float(px_per_unit)
        rr = np.clip(np.rint(row).astype(np.int32), 0, h - 1)
        cc = np.clip(np.rint(col).astype(np.int32), 0, w - 1)
        # Prefer candidates that also overlap some filled pixels (breaks symmetry on blank distances).
        m = mask_roi[rr, cc]
        d = dist[rr, cc]
        return float(np.mean(d) + 2.0 * (1.0 - float(np.mean(m))))

    def norm_deg(a: float) -> float:
        a = (float(a) + 180.0) % 360.0 - 180.0
        return float(a)

    best_o = np.asarray(origin_rc, dtype=np.float64)
    best_a = norm_deg(float(deg))
    best_c = float("inf")

    # Try both orientations (deg and deg+180) as starting points.
    for base_a in (best_a, norm_deg(best_a + 180.0)):
        o0 = np.asarray(origin_rc, dtype=np.float64)
        a0 = float(base_a)

        # Coarse search.
        for da in np.arange(-8.0, 8.0 + 1e-9, 2.0):
            for dr in range(-6, 7, 2):
                for dc in range(-6, 7, 2):
                    o = o0 + np.array([dr, dc], dtype=np.float64)
                    a = norm_deg(a0 + float(da))
                    v = cost(o, a)
                    if v < best_c:
                        best_c = v
                        best_o = o
                        best_a = a

        # Fine search around coarse optimum.
        o1 = best_o.copy()
        a1 = best_a
        for da in np.arange(-2.0, 2.0 + 1e-9, 0.5):
            for dr in range(-2, 3, 1):
                for dc in range(-2, 3, 1):
                    o = o1 + np.array([dr, dc], dtype=np.float64)
                    a = norm_deg(a1 + float(da))
                    v = cost(o, a)
                    if v < best_c:
                        best_c = v
                        best_o = o
                        best_a = a

    return best_o, float(best_a)


def _count_overlaps_xy_deg(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray) -> int:
    polys = [train._tree_polygon_scaled(float(xs[i]), float(ys[i]), float(degs[i])) for i in range(len(xs))]
    r_tree = train.STRtree(polys)
    overlaps = 0
    for i, poly in enumerate(polys):
        for j in r_tree.query(poly):
            if int(j) <= i:
                continue
            other = polys[int(j)]
            if poly.intersects(other) and not poly.touches(other):
                overlaps += 1
    return overlaps


def _flip_180_local_search(xs: np.ndarray, ys: np.ndarray, degs: np.ndarray, *, iters: int = 6) -> np.ndarray:
    """
    Greedy 180° flip search to reduce overlaps. Keeps centers fixed.
    """
    degs = np.asarray(degs, dtype=np.float64).copy()
    best = _count_overlaps_xy_deg(xs, ys, degs)
    if best == 0:
        return degs

    for _ in range(max(1, int(iters))):
        improved = False
        for i in range(len(degs)):
            cand = degs.copy()
            cand[i] = float(cand[i] + 180.0)
            ov = _count_overlaps_xy_deg(xs, ys, cand)
            if ov < best:
                degs = cand
                best = ov
                improved = True
                if best == 0:
                    return degs
        if not improved:
            break
    return degs


def _solve_min_scale_no_overlap(
    centers_rc: np.ndarray,
    degs_init: np.ndarray,
    *,
    scale0: float,
    expand: float = 1.08,
    max_expand: int = 25,
    binary_steps: int = 14,
) -> Tuple[float, np.ndarray]:
    """
    Find the smallest scale >= 0 that yields a non-overlapping placement, allowing 180° flips.

    Scale converts pixel deltas to coordinate deltas:
      x = +col * scale
      y = -row * scale
    """
    centers_rc = np.asarray(centers_rc, dtype=np.float64)
    degs_init = np.asarray(degs_init, dtype=np.float64)
    mean_rc = centers_rc.mean(axis=0)
    d_rc = centers_rc - mean_rc[None, :]

    def eval_scale(scale: float) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        xs = d_rc[:, 1] * float(scale)
        ys = -d_rc[:, 0] * float(scale)
        degs = _flip_180_local_search(xs, ys, degs_init)
        ov = _count_overlaps_xy_deg(xs, ys, degs)
        return ov, xs, ys, degs

    # First, expand upward until feasible.
    hi = float(scale0)
    lo = 0.0
    best_degs = degs_init
    for _ in range(max(1, int(max_expand))):
        ov, _xs, _ys, d = eval_scale(hi)
        if ov == 0:
            best_degs = d
            break
        lo = hi
        hi *= float(expand)
    else:
        raise ValueError("Could not find a non-overlapping scale (too noisy extraction).")

    # Binary search the smallest feasible scale.
    best_scale = hi
    for _ in range(max(1, int(binary_steps))):
        mid = 0.5 * (lo + hi)
        ov, _xs, _ys, d = eval_scale(mid)
        if ov == 0:
            best_scale = mid
            best_degs = d
            hi = mid
        else:
            lo = mid

    return float(best_scale), best_degs


def extract_group_from_image(
    img_path: Path,
    *,
    n_expected: int,
    q: int = 8,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
    decimals: int = 16,
    debug: bool = False,
) -> ExtractedGroup:
    img = np.asarray(Image.open(img_path).convert("RGB"))
    key, uniq, counts = _quantize_rgb(img, q=q)

    # Heuristic count bounds: per-tree fill size should be in a mid band.
    if min_count is None:
        min_count = max(200, int(0.2 * img.shape[0] * img.shape[1] / max(1, n_expected)))
    if max_count is None:
        max_count = int(0.35 * img.shape[0] * img.shape[1])

    tree_keys = _pick_tree_keys(
        uniq=uniq,
        counts=counts,
        q=q,
        n_expected=n_expected,
        min_count=int(min_count),
        max_count=int(max_count),
    )
    if len(tree_keys) != n_expected:
        raise ValueError(f"{img_path.name}: expected {n_expected} trees, found {len(tree_keys)} candidate colors")

    # Assign every non-background pixel to the nearest tree color center (in RGB space),
    # so we capture anti-aliased edges and outlines that don't exactly match the fill key.
    rgb_u = np.array([_key_to_rgb(int(k), q=q) for k in uniq.tolist()], dtype=np.int16)
    r = rgb_u[:, 0].astype(np.int16)
    g = rgb_u[:, 1].astype(np.int16)
    b = rgb_u[:, 2].astype(np.int16)
    is_bg = (r >= 240) & (g >= 240) & (b >= 240)
    is_red = (r >= 180) & (g <= 90) & (b <= 90)
    is_black = (r <= 70) & (g <= 70) & (b <= 70)
    valid = ~(is_bg | is_red | is_black)

    tree_rgb = np.array([_key_to_rgb(int(k), q=q) for k in tree_keys], dtype=np.int16)  # (N,3)
    assign = np.full(len(uniq), -1, dtype=np.int16)
    if np.any(valid):
        dif = rgb_u[valid][:, None, :] - tree_rgb[None, :, :]
        dist2 = np.sum(dif * dif, axis=2)  # (U_valid, N)
        min_dist2 = np.min(dist2, axis=1)
        nearest = np.argmin(dist2, axis=1).astype(np.int16)

        # Avoid assigning background-ish / UI-ish stray colors to a tree.
        # Empirically for these matplotlib renders, anti-aliased tree pixels are usually within ~80-160 RGB.
        max_dist = 160.0
        keep = min_dist2 <= (max_dist * max_dist)
        tmp = np.full(nearest.shape, -1, dtype=np.int16)
        tmp[keep] = nearest[keep]
        assign[valid] = tmp

    inv = np.searchsorted(uniq, key.reshape(-1))
    lbl = assign[inv].reshape(key.shape)  # -1 for background/UI, else [0..n_expected-1]

    centers0: List[np.ndarray] = []
    degs0: List[float] = []
    centers1: List[np.ndarray] = []
    degs1: List[float] = []
    lens: List[float] = []

    for ti in range(n_expected):
        mask = (lbl == np.int16(ti)).astype(np.uint8) * 255
        if cv2 is not None:
            nlab, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if nlab <= 1:
                raise ValueError(f"{img_path.name}: no component for tree index={ti}")
            # Pick largest non-background component.
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            mask = (labels == best).astype(np.uint8) * 255
            # Fill tiny holes / smooth jaggies slightly.
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        pts = np.column_stack(np.where(mask > 0))
        if len(pts) < 50:
            raise ValueError(f"{img_path.name}: too few pixels for a tree region (tree index={ti})")
        p0, p1, L = _estimate_tree_endpoints(pts)
        d = p1 - p0
        centers0.append(p0 + 0.2 * d)
        degs0.append(_deg_from_origin_tip(p0, p1))
        centers1.append(p1 - 0.2 * d)
        degs1.append(_deg_from_origin_tip(p1, p0))
        lens.append(float(L))

    med_len = float(np.median(lens))
    if not (med_len > 1e-6):
        raise ValueError(f"{img_path.name}: invalid median length")

    # Total tree height is exactly 1.0 in train.py geometry (from -0.2 to +0.8).
    scale0 = 1.0 / med_len

    c0 = np.stack(centers0, axis=0)
    c1 = np.stack(centers1, axis=0)
    d0 = np.asarray(degs0, dtype=np.float64)
    d1 = np.asarray(degs1, dtype=np.float64)
    use1 = np.zeros(n_expected, dtype=np.int8)

    def overlaps_for(centers_rc: np.ndarray, degs_np: np.ndarray, scale: float) -> int:
        mean_rc = centers_rc.mean(axis=0)
        d_rc = centers_rc - mean_rc[None, :]
        xs = d_rc[:, 1] * float(scale)
        ys = -d_rc[:, 0] * float(scale)
        return int(_count_overlaps_xy_deg(xs, ys, degs_np))

    centers_rc = c0.copy()
    degs_np = d0.copy()
    best_ov = overlaps_for(centers_rc, degs_np, scale0)
    # Greedy per-tree endpoint swap to reduce overlaps (also updates translation, unlike pure 180 flips).
    for _ in range(6):
        improved = False
        for i in range(n_expected):
            cand_centers = centers_rc.copy()
            cand_degs = degs_np.copy()
            if use1[i] == 0:
                cand_centers[i] = c1[i]
                cand_degs[i] = d1[i]
            else:
                cand_centers[i] = c0[i]
                cand_degs[i] = d0[i]
            ov = overlaps_for(cand_centers, cand_degs, scale0)
            if ov < best_ov:
                centers_rc = cand_centers
                degs_np = cand_degs
                use1[i] = 1 - use1[i]
                best_ov = ov
                improved = True
                if best_ov == 0:
                    break
        if not improved or best_ov == 0:
            break

    scale, degs_np = _solve_min_scale_no_overlap(centers_rc, degs_np, scale0=scale0)
    if debug:
        print(f"{img_path.name}: med_len_px={med_len:.3f} scale0={scale0:.6g} scale={scale:.6g} ov0={best_ov}")
    center_mean = np.mean(centers_rc, axis=0)

    xs: List[str] = []
    ys: List[str] = []
    ds: List[str] = []
    for i in range(n_expected):
        c = centers_rc[i] - center_mean
        x = float(c[1]) * float(scale)
        y = float(-c[0]) * float(scale)
        xs.append(train._format_s_prec(x, decimals=decimals))
        ys.append(train._format_s_prec(y, decimals=decimals))
        ds.append(train._format_s_prec(float(degs_np[i]), decimals=decimals))

    group = f"{n_expected:03d}"
    ids = [f"{group}_{i}" for i in range(n_expected)]
    df_g = pd.DataFrame({"id": ids, "x": xs, "y": ys, "deg": ds})

    score = float(train._group_score(df_g[["x", "y", "deg"]]))
    return ExtractedGroup(n=n_expected, source=img_path.name, df_group=df_g, group_score=score)


def _replace_group(base: pd.DataFrame, group_df: pd.DataFrame) -> pd.DataFrame:
    g = group_df["id"].iloc[0].split("_", 1)[0]
    keep = base[~base["id"].astype(str).str.startswith(f"{g}_")].copy()
    out = pd.concat([keep, group_df], ignore_index=True).sort_values("id").reset_index(drop=True)
    return out[["id", "x", "y", "deg"]]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="baseline_images", help="Directory containing *.png images.")
    p.add_argument("--base", required=True, help="Base submission CSV to patch (must include all groups).")
    p.add_argument("--out", required=True, help="Output patched CSV path.")
    p.add_argument("--q", type=int, default=8, help="RGB quantization factor (default 8).")
    p.add_argument("--decimals", type=int, default=16, help="Output decimals (default 16).")
    p.add_argument("--only-n", nargs="*", type=int, default=None, help="Only process specific N values.")
    p.add_argument("--prefer-best", action="store_true", help="Keep only image-derived groups that improve the base.")
    p.add_argument("--fix-direction", action="store_true", help="Apply train.apply_fix_direction to the patched output.")
    p.add_argument("--report", default=None, help="Optional report CSV path.")
    args = p.parse_args(argv)

    images_dir = Path(args.images_dir)
    base_path = Path(args.base)
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else None

    base = pd.read_csv(base_path, dtype=str)
    base = train.normalize_submission_df(base, path_hint=str(base_path))
    base_groups = train._submission_df_to_groups(base)

    only_n = set(int(x) for x in args.only_n) if args.only_n else None

    # Gather candidates per N (multiple images per N allowed; pick best score).
    best_by_n: Dict[int, ExtractedGroup] = {}
    for img_path in sorted(images_dir.glob("*.png")):
        n = _parse_n_from_filename(img_path)
        if n is None:
            continue
        if only_n is not None and n not in only_n:
            continue
        try:
            eg = extract_group_from_image(
                img_path,
                n_expected=int(n),
                q=int(args.q),
                decimals=int(args.decimals),
            )
        except Exception:
            continue

        prev = best_by_n.get(n)
        if prev is None or eg.group_score < prev.group_score:
            best_by_n[n] = eg

    patched = base.copy()
    rows = []
    for n in sorted(best_by_n.keys()):
        eg = best_by_n[n]
        base_g = base_groups.get(n)
        if base_g is None:
            continue
        base_score = float(train._group_score(base_g))
        improved = eg.group_score < base_score - 1e-12
        if args.prefer_best and not improved:
            rows.append(
                {
                    "n": n,
                    "used": 0,
                    "source": eg.source,
                    "base_score": base_score,
                    "img_score": eg.group_score,
                    "delta": eg.group_score - base_score,
                }
            )
            continue

        patched = _replace_group(patched, eg.df_group)
        rows.append(
            {
                "n": n,
                "used": 1,
                "source": eg.source,
                "base_score": base_score,
                "img_score": eg.group_score,
                "delta": eg.group_score - base_score,
            }
        )

    if args.fix_direction:
        patched = train.apply_fix_direction(patched, decimals=int(args.decimals))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    patched.to_csv(out_path, index=False)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report_path, index=False)

    try:
        score = float(train.score_submission(patched))
    except Exception:
        score = float("nan")

    used = sum(1 for r in rows if r.get("used") == 1)
    print(f"Patched groups from images: {used}/{len(rows)} -> {out_path}")
    if not math.isnan(score):
        print(f"Score: {score:.12f}")
    if report_path is not None:
        print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
