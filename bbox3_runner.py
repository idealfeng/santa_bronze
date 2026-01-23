from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import train


FINAL_SCORE_RE = re.compile(r"Final\\s+(?:Total\\s+)?Score:\\s*([0-9]+(?:\\.[0-9]+)?)")


def _windows_to_wsl_path(path: Path) -> str:
    path = path.resolve()
    drive = path.drive
    if not drive or len(drive) < 2 or drive[1] != ":":
        raise ValueError(f"Not a Windows drive path: {path}")
    drive_letter = drive[0].lower()
    rest = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive_letter}{rest}"


def _detect_wsl_distro(preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    try:
        out = subprocess.check_output(["wsl", "-l", "-q"], stderr=subprocess.STDOUT)
    except Exception:
        return None
    names = out.decode(errors="ignore").replace("\x00", "").splitlines()
    names = [n.strip() for n in names if n.strip()]
    names = [n for n in names if "docker-desktop" not in n.lower()]
    if not names:
        return None
    for n in names:
        if "ubuntu" in n.lower():
            return n
    return names[0]


def _run_bbox3(
    *,
    workdir: Path,
    n_value: int,
    r_value: int,
    timeout_sec: int,
    wsl_distro: Optional[str],
    omp_threads: Optional[int],
) -> tuple[int, str]:
    bbox3_path = workdir / "bbox3"
    if sys.platform.startswith("win"):
        distro = _detect_wsl_distro(wsl_distro)
        if not distro:
            raise RuntimeError(
                "No usable WSL distro found. Install Ubuntu (WSL) and retry, or pass --wsl-distro."
            )
        workdir_wsl = _windows_to_wsl_path(workdir)
        omp_prefix = f"OMP_NUM_THREADS={int(omp_threads)} " if omp_threads else ""
        cmd = f"cd {workdir_wsl} && chmod +x ./bbox3 && {omp_prefix}./bbox3 -n {n_value} -r {r_value}"
        res = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", cmd],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
        )
        out = (res.stdout or b"") + (res.stderr or b"")
        return int(res.returncode), out.decode("utf-8", errors="ignore")

    subprocess.run(["chmod", "+x", str(bbox3_path)], check=False)
    env = None
    if omp_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(omp_threads))
    res = subprocess.run(
        [str(bbox3_path), "-n", str(n_value), "-r", str(r_value)],
        capture_output=True,
        text=False,
        timeout=timeout_sec,
        cwd=str(workdir),
        env=env,
    )
    out = (res.stdout or b"") + (res.stderr or b"")
    return int(res.returncode), out.decode("utf-8", errors="ignore")


def _parse_bbox3_final_score(output: str) -> Optional[float]:
    m = FINAL_SCORE_RE.search(output or "")
    return float(m.group(1)) if m else None


def _load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df[["id", "x", "y", "deg"]].copy()


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _group_map(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]
    for g, sub in tmp.groupby("_group", sort=True):
        out[str(g)] = sub[["id", "x", "y", "deg"]].copy()
    return out


def _fast_group_score_map(df: pd.DataFrame) -> Dict[str, float]:
    npx = train._require_numpy()

    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]

    out: Dict[str, float] = {}
    for g, sub in tmp.groupby("_group", sort=True):
        x = train._parse_s_floats(sub["x"])
        y = train._parse_s_floats(sub["y"])
        deg = train._parse_s_floats(sub["deg"])
        X, Y = train._tree_pointcloud_xy(x, y, deg)
        side = train._side_from_points(X, Y)
        n = max(1, len(sub))
        out[str(g)] = float((side * side) / float(n))

    # Ensure stable ordering isn't required by caller
    _ = npx  # silence unused if train._require_numpy() is optimized away
    return out


def _prune_stash_dir(stash_dir: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    paths = sorted([p for p in stash_dir.glob("*.csv") if p.is_file()])
    if len(paths) <= keep:
        return

    def score_key(p: Path) -> float:
        m = re.search(r"_s([0-9]+(?:\\.[0-9]+)?)", p.name)
        return float(m.group(1)) if m else float("inf")

    paths.sort(key=score_key)
    for p in paths[keep:]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def _repair_by_donor(candidate: pd.DataFrame, donor: pd.DataFrame, *, max_repairs: int = 50) -> pd.DataFrame:
    donor_groups = _group_map(donor)
    fixed = candidate.copy()
    for _ in range(max_repairs):
        try:
            train.score_submission(fixed)
            return fixed
        except ValueError as e:
            m = re.search(r"Overlapping trees in group\\s+(\\d+)", str(e))
            if not m:
                raise
            g = m.group(1)
            if g not in donor_groups:
                raise
            fixed = fixed[fixed["id"].astype(str).str.split("_").str[0] != g].copy()
            fixed = pd.concat([fixed, donor_groups[g]], ignore_index=True).sort_values("id").reset_index(drop=True)
    raise RuntimeError("Too many overlap repairs; candidate is likely very invalid.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/submissions/santa-2025-csv/santa-2025.csv",
        help="Starting submission CSV.",
    )
    p.add_argument(
        "--donor",
        default=None,
        help="Fallback submission used to repair overlaps (defaults to --input).",
    )
    p.add_argument(
        "--bbox3",
        default="data/improvesanta/archive/bbox3",
        help="Path to bbox3 (Linux ELF). Will be copied into --workdir as ./bbox3.",
    )
    p.add_argument("--workdir", default="bbox3_work", help="Work directory for bbox3 (contains submission.csv).")
    p.add_argument("--wsl-distro", default=None, help="WSL distro name (e.g. Ubuntu). Auto-detect if omitted.")

    p.add_argument("--timeout", type=int, default=600, help="Per-run timeout seconds (default 600).")
    p.add_argument("--rounds", type=int, default=30, help="How many runs to attempt (default 30).")
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Limit OpenMP threads for bbox3 (helps control temperature when running multiple workers).",
    )
    p.add_argument("--n", nargs="+", type=int, default=[1200, 1500, 1800, 2000], help="bbox3 -n values.")
    p.add_argument("--r", nargs="+", type=int, default=[30, 60, 90], help="bbox3 -r values.")
    p.add_argument("--fix-direction", action="store_true", help="Apply train.py fix-direction after each bbox3 run.")
    p.add_argument("--delete-propagate", action="store_true", help="Apply train.py delete-propagate after each run.")
    p.add_argument("--insert-propagate", action="store_true", help="Apply train.py insert-propagate after each run.")
    p.add_argument(
        "--insert-propagate-nmax",
        type=int,
        default=60,
        help="With --insert-propagate, only attempt groups up to this n (default 60).",
    )
    p.add_argument(
        "--insert-propagate-topk",
        type=int,
        default=5,
        help="With --insert-propagate, validate up to top-K candidate inserts per group (default 5).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=16,
        help="Decimals for fix-direction output rounding (default 16).",
    )
    p.add_argument("--out", default="submission_bbox3_best.csv", help="Where to write the best found submission.")
    p.add_argument(
        "--stash-dir",
        default=None,
        help="Optional directory to save extra valid candidates (useful for later ensembling).",
    )
    p.add_argument(
        "--stash-every",
        type=int,
        default=0,
        help="If >0, also stash every N rounds (default 0 = never).",
    )
    p.add_argument(
        "--stash-keep",
        type=int,
        default=200,
        help="Max number of stashed CSVs to keep (default 200).",
    )
    p.add_argument(
        "--stash-min-improve",
        type=float,
        default=1e-10,
        help="Minimum per-group score improvement to consider a group improved for stashing (default 1e-10).",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input)
    donor_path = Path(args.donor) if args.donor else input_path
    bbox3_src = Path(args.bbox3)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)

    # Prepare working files
    shutil.copy2(bbox3_src, workdir / "bbox3")

    donor_df = _load_submission(donor_path)

    input_df = _load_submission(input_path)
    input_score = train.score_submission(input_df)
    best_df = input_df
    best_score = input_score
    best_group_scores = _fast_group_score_map(best_df)

    # Resume from an existing --out if it's better than --input (helps when you stop/restart runs).
    if out_path.exists():
        try:
            prev_df = _load_submission(out_path)
            prev_score = train.score_submission(prev_df)
            if prev_score <= best_score:
                best_df = prev_df
                best_score = prev_score
                best_group_scores = _fast_group_score_map(best_df)
                print(
                    f"Resuming from existing {out_path} score={prev_score:.12f} (input {input_score:.12f})"
                )
        except Exception as e:
            print(f"Warning: failed to resume from {out_path}: {e}")

    _atomic_write_csv(best_df, out_path)
    _atomic_write_csv(best_df, workdir / "submission.csv")

    stash_dir = Path(args.stash_dir) if args.stash_dir else None
    if stash_dir is not None:
        stash_dir.mkdir(parents=True, exist_ok=True)

    combos = [(n, r) for n in args.n for r in args.r]
    print(f"Start score: {best_score:.12f} | combos={len(combos)} | rounds={args.rounds}")

    start = time.time()
    for k in range(args.rounds):
        n_value, r_value = combos[k % len(combos)]
        shutil.copy2(out_path, workdir / "submission.csv")

        try:
            rc, output = _run_bbox3(
                workdir=workdir,
                n_value=n_value,
                r_value=r_value,
                timeout_sec=args.timeout,
                wsl_distro=args.wsl_distro,
                omp_threads=args.omp_threads,
            )
        except subprocess.TimeoutExpired:
            print(f"[{k+1:03d}] TIMEOUT n={n_value} r={r_value}")
            continue

        if rc != 0:
            head = "\n".join((output or "").splitlines()[:20])
            print(f"[{k+1:03d}] bbox3 FAILED (rc={rc}) n={n_value} r={r_value}\n{head}")
            if "libgomp.so.1" in output or "error while loading shared libraries" in output:
                print(
                    "\nUbuntu 里缺少 OpenMP 运行库。请在 Ubuntu 终端执行：\n"
                    "  sudo apt update\n"
                    "  sudo apt install -y libgomp1\n"
                )
            return 2

        reported = _parse_bbox3_final_score(output)
        if reported is not None and reported >= best_score and not (
            args.delete_propagate or args.insert_propagate or args.fix_direction
        ):
            print(
                f"[{k+1:03d}] skip n={n_value} r={r_value} (bbox3 {reported:.12f} >= best {best_score:.12f})"
            )
            continue

        cand_df = _load_submission(workdir / "submission.csv")
        if args.delete_propagate:
            groups = train._submission_df_to_groups(cand_df)
            groups = train._delete_propagate_groups(groups, verbose=False)
            cand_df = train._groups_to_submission_df(groups)
        if args.insert_propagate:
            groups = train._submission_df_to_groups(cand_df)
            groups = train._insert_propagate_groups(
                groups,
                nmax=int(args.insert_propagate_nmax),
                topk=int(args.insert_propagate_topk),
                verbose=False,
            )
            cand_df = train._groups_to_submission_df(groups)
        if args.fix_direction:
            cand_df = train.apply_fix_direction(cand_df, decimals=int(args.decimals))

        try:
            cand_df = _repair_by_donor(cand_df, donor_df)
            cand_score = train.score_submission(cand_df)
        except Exception as e:
            print(f"[{k+1:03d}] invalid n={n_value} r={r_value}: {e}")
            continue

        dt = time.time() - start

        improved_groups = 0
        if stash_dir is not None:
            try:
                cand_group_scores = _fast_group_score_map(cand_df)
                for g, s in cand_group_scores.items():
                    prev = best_group_scores.get(g)
                    if prev is None:
                        continue
                    if s < prev - float(args.stash_min_improve):
                        improved_groups += 1
            except Exception:
                improved_groups = 0

        if cand_score < best_score:
            best_score = cand_score
            best_df = cand_df
            best_group_scores = _fast_group_score_map(best_df)
            _atomic_write_csv(best_df, out_path)
            print(
                f"[{k+1:03d}] NEW BEST {best_score:.12f} (n={n_value} r={r_value}) t={dt/60:.1f}m"
            )
        else:
            print(f"[{k+1:03d}] keep {best_score:.12f} (cand {cand_score:.12f}) n={n_value} r={r_value}")

        if stash_dir is not None:
            do_stash = False
            if improved_groups > 0:
                do_stash = True
            elif int(args.stash_every) > 0 and ((k + 1) % int(args.stash_every) == 0):
                do_stash = True

            if do_stash:
                name = f"cand_{k+1:04d}_s{cand_score:.12f}_g{improved_groups:03d}_n{n_value}_r{r_value}.csv"
                _atomic_write_csv(cand_df, stash_dir / name)
                _prune_stash_dir(stash_dir, keep=int(args.stash_keep))

    print(f"Done. Best: {best_score:.12f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
