from __future__ import annotations

import argparse
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
) -> str:
    bbox3_path = workdir / "bbox3"
    if sys.platform.startswith("win"):
        distro = _detect_wsl_distro(wsl_distro)
        if not distro:
            raise RuntimeError(
                "No usable WSL distro found. Install Ubuntu (WSL) and retry, or pass --wsl-distro."
            )
        workdir_wsl = _windows_to_wsl_path(workdir)
        cmd = f"cd {workdir_wsl} && chmod +x ./bbox3 && ./bbox3 -n {n_value} -r {r_value}"
        res = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", cmd],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
        )
        out = (res.stdout or b"") + (res.stderr or b"")
        return out.decode("utf-8", errors="replace")

    subprocess.run(["chmod", "+x", str(bbox3_path)], check=False)
    res = subprocess.run(
        [str(bbox3_path), "-n", str(n_value), "-r", str(r_value)],
        capture_output=True,
        text=False,
        timeout=timeout_sec,
        cwd=str(workdir),
    )
    out = (res.stdout or b"") + (res.stderr or b"")
    return out.decode("utf-8", errors="replace")


def _parse_bbox3_final_score(output: str) -> Optional[float]:
    m = FINAL_SCORE_RE.search(output or "")
    return float(m.group(1)) if m else None


def _load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df[["id", "x", "y", "deg"]].copy()


def _group_map(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]
    for g, sub in tmp.groupby("_group", sort=True):
        out[str(g)] = sub[["id", "x", "y", "deg"]].copy()
    return out


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
    p.add_argument("--n", nargs="+", type=int, default=[1200, 1500, 1800, 2000], help="bbox3 -n values.")
    p.add_argument("--r", nargs="+", type=int, default=[30, 60, 90], help="bbox3 -r values.")
    p.add_argument("--fix-direction", action="store_true", help="Apply train.py fix-direction after each bbox3 run.")
    p.add_argument("--delete-propagate", action="store_true", help="Apply train.py delete-propagate after each run.")
    p.add_argument("--out", default="submission_bbox3_best.csv", help="Where to write the best found submission.")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    donor_path = Path(args.donor) if args.donor else input_path
    bbox3_src = Path(args.bbox3)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Prepare working files
    shutil.copy2(bbox3_src, workdir / "bbox3")
    shutil.copy2(input_path, workdir / "submission.csv")

    donor_df = _load_submission(donor_path)
    best_df = _load_submission(workdir / "submission.csv")
    best_score = train.score_submission(best_df)
    out_path = Path(args.out)
    best_df.to_csv(out_path, index=False)

    combos = [(n, r) for n in args.n for r in args.r]
    print(f"Start score: {best_score:.12f} | combos={len(combos)} | rounds={args.rounds}")

    start = time.time()
    for k in range(args.rounds):
        n_value, r_value = combos[k % len(combos)]
        shutil.copy2(out_path, workdir / "submission.csv")

        try:
            output = _run_bbox3(
                workdir=workdir,
                n_value=n_value,
                r_value=r_value,
                timeout_sec=args.timeout,
                wsl_distro=args.wsl_distro,
            )
        except subprocess.TimeoutExpired:
            print(f"[{k+1:03d}] TIMEOUT n={n_value} r={r_value}")
            continue

        reported = _parse_bbox3_final_score(output)
        if reported is not None and reported >= best_score:
            print(f"[{k+1:03d}] skip n={n_value} r={r_value} (bbox3 {reported:.12f} >= best {best_score:.12f})")
            continue

        cand_df = _load_submission(workdir / "submission.csv")
        if args.delete_propagate:
            groups = train._submission_df_to_groups(cand_df)
            groups = train._delete_propagate_groups(groups, verbose=False)
            cand_df = train._groups_to_submission_df(groups)
        if args.fix_direction:
            cand_df = train.apply_fix_direction(cand_df, decimals=12)

        try:
            cand_df = _repair_by_donor(cand_df, donor_df)
            cand_score = train.score_submission(cand_df)
        except Exception as e:
            print(f"[{k+1:03d}] invalid n={n_value} r={r_value}: {e}")
            continue

        dt = time.time() - start
        if cand_score < best_score:
            best_score = cand_score
            best_df = cand_df
            best_df.to_csv(out_path, index=False)
            print(
                f"[{k+1:03d}] NEW BEST {best_score:.12f} (n={n_value} r={r_value}) t={dt/60:.1f}m"
            )
        else:
            print(f"[{k+1:03d}] keep {best_score:.12f} (cand {cand_score:.12f}) n={n_value} r={r_value}")

    print(f"Done. Best: {best_score:.12f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
