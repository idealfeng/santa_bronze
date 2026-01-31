from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

import train


SCALE = 1000.0

# Tree polygon (unscaled), matches train.py TREE_VERTS.
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

# Sparrow input polygon (scaled by 1000).
SPARROW_POLY: List[List[float]] = [
    [0, 800],
    [125, 500],
    [62.5, 500],
    [200, 250],
    [100, 250],
    [350, 0],
    [75, 0],
    [75, -200],
    [-75, -200],
    [-75, 0],
    [-350, 0],
    [-100, 250],
    [-200, 250],
    [-62.5, 500],
    [-125, 500],
]


def _parse_s(x) -> float:
    s = str(x).strip()
    if s.startswith("s") or s.startswith("S"):
        s = s[1:].strip()
    return float(s)


def _wrap_deg(d: float) -> float:
    return (float(d) + 180.0) % 360.0 - 180.0


def _fmt_s(v: float, *, decimals: int) -> str:
    s = f"{float(v):.{int(decimals)}f}".rstrip("0").rstrip(".")
    return "s" + (s if s else "0")


def _transform_point(x: float, y: float, tx: float, ty: float, deg: float) -> Tuple[float, float]:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    rx = x * c - y * s
    ry = x * s + y * c
    return rx + tx, ry + ty


def _group_score_from_placements(placements: Sequence[Tuple[float, float, float]]) -> Tuple[float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for tx, ty, deg in placements:
        for vx, vy in TREE_VERTS:
            px, py = _transform_point(vx, vy, tx, ty, deg)
            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)
    side = max(max_x - min_x, max_y - min_y)
    n = max(1, len(placements))
    return (side * side) / float(n), side


def _extract_group_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    prefix = f"{n:03d}_"
    g = df[df["id"].astype(str).str.startswith(prefix)].copy()
    return g.sort_values("id").reset_index(drop=True)


def _write_task_json(task_path: Path, *, n: int, strip_height: int) -> None:
    task = {
        "name": f"n{n}_h{strip_height}",
        "items": [
            {
                "id": 0,
                "demand": int(n),
                "shape": {"type": "simple_polygon", "data": SPARROW_POLY},
            }
        ],
        "strip_height": float(strip_height),
    }
    task_path.write_text(json.dumps(task), encoding="utf-8")


def _load_sparrow_solution_json(path: Path) -> List[Tuple[float, float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sol = data.get("solution", data)
    layout = sol["layout"]
    out: List[Tuple[float, float, float]] = []
    for it in layout["placed_items"]:
        t = it["transformation"]
        x = float(t["translation"][0]) / SCALE
        y = float(t["translation"][1]) / SCALE
        deg = float(t["rotation"])
        out.append((x, y, _wrap_deg(deg)))
    return out


def _validate_group_no_overlap(n: int, placements: Sequence[Tuple[float, float, float]], *, decimals: int) -> bool:
    rows = []
    for i, (x, y, deg) in enumerate(placements):
        rows.append(
            {
                "id": f"{n:03d}_{i}",
                "x": _fmt_s(x, decimals=decimals),
                "y": _fmt_s(y, decimals=decimals),
                "deg": _fmt_s(deg, decimals=decimals),
            }
        )
    gdf = pd.DataFrame(rows, columns=["id", "x", "y", "deg"])
    try:
        _ = train.score_submission(gdf)
        return True
    except Exception:
        return False


def _apply_group_patch(df: pd.DataFrame, *, n: int, placements: Sequence[Tuple[float, float, float]], decimals: int) -> pd.DataFrame:
    g = _extract_group_df(df, n)
    if len(g) != len(placements):
        raise ValueError(f"Group {n:03d} row count mismatch: csv={len(g)} placements={len(placements)}")
    idxs = g.index.to_list()
    # Need original df indices (g is reset_index); map by id match:
    prefix = f"{n:03d}_"
    orig_idxs = df[df["id"].astype(str).str.startswith(prefix)].sort_values("id").index.to_list()
    for i, (x, y, deg) in enumerate(placements):
        df.loc[orig_idxs[i], "x"] = _fmt_s(x, decimals=decimals)
        df.loc[orig_idxs[i], "y"] = _fmt_s(y, decimals=decimals)
        df.loc[orig_idxs[i], "deg"] = _fmt_s(deg, decimals=decimals)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sparrow-exe", required=True, help="Path to sparrow.exe")
    p.add_argument("--base", required=True, help="Base submission CSV (id,x,y,deg)")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--workdir", required=True, help="Work directory (isolates output/ per seed)")
    p.add_argument("--nmin", type=int, default=2)
    p.add_argument("--nmax", type=int, default=60)
    p.add_argument("--time", type=int, default=60, help="Seconds per N (global-time)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--strip-margin", type=float, default=1.003, help="Multiply current side*1000 by this margin")
    p.add_argument("--strip-min", type=int, default=2000)
    p.add_argument("--decimals", type=int, default=16)
    p.add_argument("--min-improve", type=float, default=1e-12, help="Only accept if new group score improves by this")
    args = p.parse_args(argv)

    sparrow_exe = Path(args.sparrow_exe).resolve()
    base_path = Path(args.base).resolve()
    out_path = Path(args.out).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if not sparrow_exe.exists():
        raise SystemExit(f"Missing sparrow exe: {sparrow_exe}")
    if not base_path.exists():
        raise SystemExit(f"Missing base CSV: {base_path}")

    df = train.normalize_submission_df(pd.read_csv(base_path, dtype=str), path_hint=str(base_path))
    df_work = df.copy()

    nmin = max(1, int(args.nmin))
    nmax = min(200, int(args.nmax))
    if nmin > nmax:
        raise SystemExit("--nmin must be <= --nmax")

    t_per = max(1, int(args.time))
    seed = int(args.seed)
    decimals = max(16, int(args.decimals))
    min_improve = float(args.min_improve)

    # workdir: copy base submission for provenance
    shutil.copy2(base_path, workdir / "submission_base.csv")

    improved = 0
    start = time.time()
    for n in range(nmin, nmax + 1):
        g = _extract_group_df(df_work, n)
        if len(g) != n:
            continue
        old_placements = [(_parse_s(r.x), _parse_s(r.y), _parse_s(r.deg)) for r in g.itertuples(index=False)]
        old_score, old_side = _group_score_from_placements(old_placements)

        strip_h = int(math.ceil(old_side * SCALE * float(args.strip_margin)))
        strip_h = max(int(args.strip_min), strip_h)

        task = workdir / f"task_n{n:03d}_h{strip_h}.json"
        _write_task_json(task, n=n, strip_height=strip_h)

        cmd = [str(sparrow_exe), "-i", str(task), "-t", str(t_per), "-s", str(seed)]
        res = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
        if res.returncode != 0:
            # Keep going; sparrow may fail on tight heights.
            print(f"n={n:03d} sparrow failed rc={res.returncode}")
            continue

        out_json = workdir / "output" / f"final_{task.stem}.json"
        if not out_json.exists():
            # fallback: any final_*.json newest
            cands = sorted((workdir / "output").glob("final_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            out_json = cands[0] if cands else out_json
        if not out_json.exists():
            print(f"n={n:03d} missing output json")
            continue

        new_placements = _load_sparrow_solution_json(out_json)
        if len(new_placements) != n:
            print(f"n={n:03d} bad placement count={len(new_placements)}")
            continue

        new_score, new_side = _group_score_from_placements(new_placements)
        if new_score >= old_score - min_improve:
            print(f"n={n:03d} noimp {old_score:.6f}->{new_score:.6f}")
            continue

        if not _validate_group_no_overlap(n, new_placements, decimals=decimals):
            print(f"n={n:03d} reject (overlap)")
            continue

        df_work = _apply_group_patch(df_work, n=n, placements=new_placements, decimals=decimals)
        improved += 1
        print(f"n={n:03d} improved {old_score:.6f}->{new_score:.6f} side {old_side:.3f}->{new_side:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_work.to_csv(out_path, index=False)

    dt = time.time() - start
    print(f"Wrote {out_path} | improved groups: {improved} | t={dt/60:.1f}m")
    try:
        print(f"Score: {train.score_submission(df_work):.12f}")
    except Exception as e:
        print(f"Score failed: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

