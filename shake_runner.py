from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

import train


def _metric_overlap_groups(df: pd.DataFrame) -> list[int]:
    """
    Return list of group sizes N that contain overlaps, using the same collision semantics as train.score_submission.
    """
    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp = train.normalize_submission_df(tmp, path_hint="shake_runner")
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]

    bad: list[int] = []
    for g, sub in tmp.groupby("_group", sort=True):
        placed: list[train.ChristmasTree] = []
        for _, row in sub.iterrows():
            # train.score_submission strips leading "s" and passes string numbers to Decimal.
            placed.append(
                train.ChristmasTree(
                    str(row["x"])[1:],
                    str(row["y"])[1:],
                    str(row["deg"])[1:],
                )
            )
        polys = [t.polygon for t in placed]
        r_tree = train.STRtree(polys)
        ok = True
        for i, poly in enumerate(polys):
            for j in r_tree.query(poly):
                if int(j) == i:
                    continue
                other = polys[int(j)]
                if poly.intersects(other) and not poly.touches(other):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            try:
                bad.append(int(g))
            except Exception:
                pass
    bad.sort()
    return bad


def _detect_wsl_distro(prefer: Optional[str]) -> Optional[str]:
    if prefer:
        return str(prefer)
    try:
        res = subprocess.run(["wsl", "-l", "-q"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    names = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
    if not names:
        return None
    for n in names:
        if "ubuntu" in n.lower():
            return n
    return names[0]


def _windows_to_wsl_path(p: Path) -> str:
    s = str(p.resolve())
    # C:\a\b -> /mnt/c/a/b
    drive = s[0].lower()
    rest = s[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def _run_shake_public(
    *,
    workdir: Path,
    shake_path: Path,
    timeout_sec: int,
    wsl_distro: Optional[str],
    omp_threads: Optional[int],
) -> tuple[int, str]:
    local = workdir / "shake_public"
    if sys.platform.startswith("win"):
        distro = _detect_wsl_distro(wsl_distro)
        if not distro:
            raise RuntimeError("No usable WSL distro found. Install Ubuntu (WSL) or pass --wsl-distro.")
        workdir_wsl = _windows_to_wsl_path(workdir)
        omp_prefix = f"OMP_NUM_THREADS={int(omp_threads)} " if omp_threads else ""
        cmd = (
            f"cd {workdir_wsl} && chmod +x ./shake_public && "
            f"{omp_prefix}./shake_public --input=submission.csv --output=submission.csv"
        )
        res = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", cmd],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
        )
        out = (res.stdout or b"") + (res.stderr or b"")
        return int(res.returncode), out.decode("utf-8", errors="ignore")

    subprocess.run(["chmod", "+x", str(local)], check=False)
    env = None
    if omp_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(omp_threads))
    res = subprocess.run(
        [str(local), "--input=submission.csv", "--output=submission.csv"],
        capture_output=True,
        text=False,
        timeout=timeout_sec,
        cwd=str(workdir),
        env=env,
    )
    out = (res.stdout or b"") + (res.stderr or b"")
    return int(res.returncode), out.decode("utf-8", errors="ignore")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input submission CSV.")
    p.add_argument("--donor", default=None, help="Fallback submission to replace overlapping groups (optional).")
    p.add_argument("--shake", required=True, help="Path to shake_public (Linux ELF).")
    p.add_argument("--workdir", default="shake_work", help="Working directory (contains submission.csv).")
    p.add_argument("--wsl-distro", default=None, help="WSL distro name (e.g. Ubuntu). Auto-detect if omitted.")
    p.add_argument("--timeout", type=int, default=3600, help="Timeout seconds for a single run (default 3600).")
    p.add_argument("--omp-threads", type=int, default=None, help="Set OMP_NUM_THREADS for shake_public.")
    p.add_argument("--fix-direction", action="store_true", help="Apply train.py fix-direction after shake.")
    p.add_argument("--suffix-min-propagate", action="store_true", help="Apply suffix-min-propagate after shake.")
    p.add_argument("--suffix-min-mode", choices=["prefix", "prune"], default="prefix")
    p.add_argument("--decimals", type=int, default=16, help="Decimals for fix-direction / suffix-min output.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    donor_path = Path(args.donor) if args.donor else None
    shake_path = Path(args.shake)
    out_path = Path(args.out)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Prepare working files
    shutil.copy2(shake_path, workdir / "shake_public")
    df = pd.read_csv(input_path, dtype=str)
    df = train.normalize_submission_df(df, path_hint=str(input_path))
    df.to_csv(workdir / "submission.csv", index=False)

    rc, output = _run_shake_public(
        workdir=workdir,
        shake_path=shake_path,
        timeout_sec=int(args.timeout),
        wsl_distro=str(args.wsl_distro) if args.wsl_distro else None,
        omp_threads=int(args.omp_threads) if args.omp_threads else None,
    )
    if rc != 0:
        head = "\n".join((output or "").splitlines()[:60])
        print(f"shake_public FAILED (rc={rc})\n{head}")
        return 2

    cand = pd.read_csv(workdir / "submission.csv", dtype=str)
    cand = train.normalize_submission_df(cand, path_hint=str(workdir / "submission.csv"))
    if args.fix_direction:
        cand = train.apply_fix_direction(cand, decimals=int(args.decimals))
    if args.suffix_min_propagate:
        groups = train._submission_df_to_groups(cand)
        groups = train._suffix_min_propagate_groups(
            groups,
            mode=str(args.suffix_min_mode),
            min_improve=1e-12,
            decimals=int(args.decimals),
            verbose=True,
        )
        cand = train._groups_to_submission_df(groups)
        if args.fix_direction:
            cand = train.apply_fix_direction(cand, decimals=int(args.decimals))

    # Optional overlap repair (replace whole overlapping groups from donor).
    if donor_path is not None:
        donor = pd.read_csv(donor_path, dtype=str)
        donor = train.normalize_submission_df(donor, path_hint=str(donor_path))
        groups = train._submission_df_to_groups(cand)
        donor_groups = train._submission_df_to_groups(donor)
        repaired_total = 0
        # Iterate a few times in case downstream post-processing reorders / reserializes values.
        for _ in range(3):
            bad = _metric_overlap_groups(cand)
            if not bad:
                break
            repaired = 0
            for n in bad:
                d = donor_groups.get(int(n))
                if d is None:
                    continue
                groups[int(n)] = d.reset_index(drop=True)[["x", "y", "deg"]].copy()
                repaired += 1
            repaired_total += repaired
            if repaired:
                cand = train._groups_to_submission_df(groups)
                if args.fix_direction:
                    cand = train.apply_fix_direction(cand, decimals=int(args.decimals))
            else:
                break
        if repaired_total:
            print(f"Repaired overlapping groups from donor: {repaired_total}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (rows={len(cand)})")
    try:
        score = float(train.score_submission(cand))
        print(f"Score: {score:.12f}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
