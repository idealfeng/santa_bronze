from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import train

try:
    from shapely.strtree import STRtree
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: shapely\n"
        "Install with: pip install shapely\n"
        f"Original import error: {e}"
    )


def _rng(seed: int) -> random.Random:
    r = random.Random(int(seed))
    # burn a bit for compatibility across Python versions
    for _ in range(20):
        r.random()
    return r


def _parse_deg_set(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        out = [0.0, 180.0]
    return out


def _wrap_deg(d: float) -> float:
    return (float(d) + 180.0) % 360.0 - 180.0


def _side_from_polys(polys) -> float:
    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")
    for p in polys:
        bx0, by0, bx1, by1 = p.bounds
        if bx0 < minx:
            minx = bx0
        if by0 < miny:
            miny = by0
        if bx1 > maxx:
            maxx = bx1
        if by1 > maxy:
            maxy = by1
    return max(maxx - minx, maxy - miny)


def _non_overlap(poly, polys: List, rtree: STRtree) -> bool:
    for j in rtree.query(poly):
        j = int(j)
        other = polys[j]
        if poly.intersects(other) and not poly.touches(other):
            return False
    return True


@dataclass
class Item:
    x: float
    y: float
    deg: float


def _items_from_group(df_group: pd.DataFrame) -> List[Item]:
    x = train._parse_s_floats(df_group["x"])
    y = train._parse_s_floats(df_group["y"])
    deg = train._parse_s_floats(df_group["deg"])
    return [Item(float(xi), float(yi), float(di)) for xi, yi, di in zip(x, y, deg)]


def _group_df_from_items(group_id: int, items: List[Item], *, decimals: int) -> pd.DataFrame:
    rows = []
    for i, it in enumerate(items):
        rows.append(
            {
                "id": f"{group_id:03d}_{i}",
                "x": "s" + f"{it.x:.{decimals}f}".rstrip("0").rstrip("."),
                "y": "s" + f"{it.y:.{decimals}f}".rstrip("0").rstrip("."),
                "deg": "s" + f"{_wrap_deg(it.deg):.{decimals}f}".rstrip("0").rstrip("."),
            }
        )
    return pd.DataFrame(rows, columns=["id", "x", "y", "deg"])


def _repair_place_items(
    *,
    fixed: List[Item],
    to_place: List[Item],
    rng: random.Random,
    allowed_deg: Sequence[float],
    move_xy: float,
    move_deg: float,
    trials: int,
) -> Optional[List[Item]]:
    placed: List[Item] = [Item(it.x, it.y, it.deg) for it in fixed]
    polys = [train._tree_polygon_scaled(it.x, it.y, it.deg) for it in placed]

    # base box around fixed items (fallback: centered at origin).
    if polys:
        bx0 = min(p.bounds[0] for p in polys)
        by0 = min(p.bounds[1] for p in polys)
        bx1 = max(p.bounds[2] for p in polys)
        by1 = max(p.bounds[3] for p in polys)
    else:
        bx0 = by0 = -0.5 * float(train.SCALE_FACTOR)
        bx1 = by1 = 0.5 * float(train.SCALE_FACTOR)

    # Expand search region a bit (scaled coordinates).
    pad = float(train.SCALE_FACTOR) * 0.6
    rx0, ry0, rx1, ry1 = bx0 - pad, by0 - pad, bx1 + pad, by1 + pad

    for it in to_place:
        # bias around old position + around box center + random
        cx = 0.5 * (rx0 + rx1)
        cy = 0.5 * (ry0 + ry1)

        ok = False
        for _ in range(int(trials)):
            mode = rng.random()
            if mode < 0.50:
                x = it.x + rng.uniform(-move_xy, move_xy)
                y = it.y + rng.uniform(-move_xy, move_xy)
            elif mode < 0.85:
                x = (cx / float(train.SCALE_FACTOR)) + rng.uniform(-0.6, 0.6)
                y = (cy / float(train.SCALE_FACTOR)) + rng.uniform(-0.6, 0.6)
            else:
                x = rng.uniform(train.XY_MIN, train.XY_MAX)
                y = rng.uniform(train.XY_MIN, train.XY_MAX)

            # angle
            if rng.random() < 0.8:
                base = rng.choice(allowed_deg)
                deg = base + rng.uniform(-move_deg, move_deg)
            else:
                deg = it.deg + rng.uniform(-move_deg, move_deg)

            poly = train._tree_polygon_scaled(x, y, deg)
            if not polys:
                placed.append(Item(x, y, deg))
                polys.append(poly)
                ok = True
                break

            rtree = STRtree(polys)
            if _non_overlap(poly, polys, rtree):
                placed.append(Item(x, y, deg))
                polys.append(poly)
                ok = True
                break

        if not ok:
            return None

    return placed


def lns_optimize_group(
    *,
    group_id: int,
    items: List[Item],
    rng: random.Random,
    iters: int,
    k_frac: float,
    k_min: int,
    k_max: int,
    allowed_deg: Sequence[float],
    move_xy: float,
    move_deg: float,
    trials: int,
    min_improve: float,
    time_limit_s: float,
    decimals: int,
    validate_fix_direction: bool,
) -> Tuple[List[Item], float]:
    start = time.time()
    polys = [train._tree_polygon_scaled(it.x, it.y, it.deg) for it in items]
    best_items = [Item(it.x, it.y, it.deg) for it in items]
    best_side = _side_from_polys(polys)

    n = len(items)
    idx_all = list(range(n))

    for t in range(int(iters)):
        if time.time() - start > time_limit_s:
            break

        # destroy size
        k = int(round(float(k_frac) * n))
        k = max(int(k_min), k)
        k = min(int(k_max), k, n)
        if k <= 0:
            continue

        rng.shuffle(idx_all)
        cut = idx_all[:k]
        keep = idx_all[k:]

        fixed = [best_items[i] for i in keep]
        removed = [best_items[i] for i in cut]

        cand = _repair_place_items(
            fixed=fixed,
            to_place=removed,
            rng=rng,
            allowed_deg=allowed_deg,
            move_xy=move_xy,
            move_deg=move_deg,
            trials=trials,
        )
        if cand is None:
            continue

        cand_polys = [train._tree_polygon_scaled(it.x, it.y, it.deg) for it in cand]
        side = _side_from_polys(cand_polys)
        if side < best_side - float(min_improve):
            # Robust overlap validation using the exact competition semantics (Decimal + STRtree).
            try:
                df_check = _group_df_from_items(int(group_id), cand, decimals=int(decimals))
                if validate_fix_direction:
                    df_check = train.apply_fix_direction(df_check, decimals=int(decimals))
                _ = train.score_submission(df_check)
            except Exception:
                continue
            best_side = float(side)
            best_items = cand

    return best_items, float(best_side)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input submission CSV.")
    p.add_argument("--out", required=True, help="Output submission CSV.")
    p.add_argument("--nmax", type=int, default=60, help="Optimize groups 1..N (default 60).")
    p.add_argument("--top", type=int, default=0, help="Only optimize worst top-K groups within 1..N (0 = all).")
    p.add_argument("--iters", type=int, default=1500, help="LNS iterations per group (default 1500).")
    p.add_argument("--time-per-group", type=float, default=60.0, help="Seconds per group (default 60).")
    p.add_argument("--k-frac", type=float, default=0.25, help="Fraction of items to destroy (default 0.25).")
    p.add_argument("--k-min", type=int, default=2, help="Min destroyed items (default 2).")
    p.add_argument("--k-max", type=int, default=12, help="Max destroyed items (default 12).")
    p.add_argument("--trials", type=int, default=200, help="Repair placement trials per item (default 200).")
    p.add_argument("--move-xy", type=float, default=0.10, help="XY move scale (default 0.10).")
    p.add_argument("--move-deg", type=float, default=15.0, help="Angle jitter scale (default 15).")
    p.add_argument("--allowed-deg", default="0,180,90,-90", help="Comma-separated anchor angles.")
    p.add_argument("--min-improve", type=float, default=1e-12, help="Minimum side improvement to accept.")
    p.add_argument("--decimals", type=int, default=16, help="Output decimals (default 16).")
    p.add_argument("--fix-direction", action="store_true", help="Apply train.py fix-direction after optimization.")
    p.add_argument("--suffix-min-propagate", action="store_true", help="Apply suffix-min-propagate after optimization.")
    p.add_argument("--suffix-min-mode", choices=["prefix", "prune"], default="prune")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score", action="store_true", help="Compute and print full score at end.")
    args = p.parse_args(argv)

    nmax = max(1, min(200, int(args.nmax)))
    rng = _rng(int(args.seed))
    allowed_deg = _parse_deg_set(str(args.allowed_deg))
    decimals = max(16, int(args.decimals))

    src = pd.read_csv(args.input, dtype=str)
    df = train.normalize_submission_df(src, path_hint=str(args.input))
    groups = train._submission_df_to_groups(df)

    # Select target groups in 1..nmax
    cand_ns = [n for n in sorted(groups.keys()) if 1 <= int(n) <= nmax]
    if not cand_ns:
        raise SystemExit(f"No groups in 1..{nmax} found.")

    # rank by current group score (higher is worse)
    scored: List[Tuple[float, int]] = []
    for n in cand_ns:
        s = float(train._group_score(groups[n]))
        scored.append((s, int(n)))
    scored.sort(reverse=True)

    topk = int(args.top)
    if topk > 0:
        target_ns = [n for _s, n in scored[:topk]]
    else:
        target_ns = [n for _s, n in scored]
    target_ns = sorted(target_ns)

    print(f"LNS optimizing groups: {target_ns[0]}..{target_ns[-1]} (count={len(target_ns)}) within n<= {nmax}")

    improved = 0
    for n in target_ns:
        orig_g = groups[n].reset_index(drop=True)[["x", "y", "deg"]].copy()
        g = orig_g
        items = _items_from_group(g)
        if len(items) <= 1:
            continue
        start_side = float(train._group_side(g))
        start_score = float(train._group_score(g))

        print(f"n={n:03d}: start side={start_side:.6f} score={start_score:.6f}")
        best_items, best_side = lns_optimize_group(
            group_id=int(n),
            items=items,
            rng=rng,
            iters=int(args.iters),
            k_frac=float(args.k_frac),
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            allowed_deg=allowed_deg,
            move_xy=float(args.move_xy),
            move_deg=float(args.move_deg),
            trials=int(args.trials),
            min_improve=float(args.min_improve),
            time_limit_s=float(args.time_per_group),
            decimals=decimals,
            validate_fix_direction=bool(args.fix_direction),
        )

        end_items = best_items
        out_df_group = _group_df_from_items(int(n), end_items, decimals=decimals)
        if args.fix_direction:
            out_df_group = train.apply_fix_direction(out_df_group, decimals=decimals)
        # Validate the group exactly as competition does; revert on failure.
        try:
            _ = train.score_submission(out_df_group)
        except Exception:
            out_df_group = _group_df_from_items(int(n), _items_from_group(orig_g), decimals=decimals)
            if args.fix_direction:
                out_df_group = train.apply_fix_direction(out_df_group, decimals=decimals)
        out_group = out_df_group[["x", "y", "deg"]].reset_index(drop=True)

        end_side = float(train._group_side(out_group))
        end_score = float(train._group_score(out_group))

        delta = start_score - end_score
        if delta > 0:
            improved += 1
        print(f"n={n:03d}: done side={end_side:.6f} score={end_score:.6f} (delta={delta:+.6f})")

        # Important: don't rewrite a group unless it truly improved; rewriting can introduce tiny
        # rounding changes that make extremely tight solutions overlap on re-parse.
        if delta > 0.0:
            groups[n] = out_group.reset_index(drop=True)[["x", "y", "deg"]]
        else:
            groups[n] = orig_g.reset_index(drop=True)[["x", "y", "deg"]]

    if args.suffix_min_propagate:
        groups = train._suffix_min_propagate_groups(
            groups,
            mode=str(args.suffix_min_mode),
            min_improve=1e-12,
            decimals=decimals,
            verbose=True,
        )
    out_df = train._groups_to_submission_df(groups)
    if args.fix_direction:
        out_df = train.apply_fix_direction(out_df, decimals=decimals)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (rows={len(out_df)}) | improved groups: {improved}")
    if args.score:
        try:
            print(f"Score: {train.score_submission(out_df):.12f}")
        except Exception as e:
            print(f"Score failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
