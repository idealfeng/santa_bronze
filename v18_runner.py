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


def _run_v18(
    *,
    workdir: Path,
    iters: int,
    restarts: int,
    timeout_sec: int,
    wsl_distro: Optional[str],
    omp_threads: Optional[int],
) -> tuple[int, str]:
    bin_path = workdir / "v18"
    if sys.platform.startswith("win"):
        distro = _detect_wsl_distro(wsl_distro)
        if not distro:
            raise RuntimeError(
                "No usable WSL distro found. Install Ubuntu (WSL) and retry, or pass --wsl-distro."
            )
        workdir_wsl = _windows_to_wsl_path(workdir)
        omp_prefix = f"OMP_NUM_THREADS={int(omp_threads)} " if omp_threads else ""
        cmd = (
            f"cd {workdir_wsl} && chmod +x ./v18 && "
            f"{omp_prefix}./v18 -i submission.csv -o submission_out.csv -n {int(iters)} -r {int(restarts)}"
        )
        res = subprocess.run(
            ["wsl", "-d", distro, "bash", "-lc", cmd],
            capture_output=True,
            text=False,
            timeout=timeout_sec,
        )
        out = (res.stdout or b"") + (res.stderr or b"")
        return int(res.returncode), out.decode("utf-8", errors="ignore")

    subprocess.run(["chmod", "+x", str(bin_path)], check=False)
    env = None
    if omp_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(omp_threads))
    res = subprocess.run(
        [str(bin_path), "-i", "submission.csv", "-o", "submission_out.csv", "-n", str(int(iters)), "-r", str(int(restarts))],
        capture_output=True,
        text=False,
        timeout=timeout_sec,
        cwd=str(workdir),
        env=env,
    )
    out = (res.stdout or b"") + (res.stderr or b"")
    return int(res.returncode), out.decode("utf-8", errors="ignore")


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
    train._require_numpy()

    tmp = df[["id", "x", "y", "deg"]].copy()
    tmp["_group"] = tmp["id"].astype(str).str.split("_").str[0]

    out: Dict[str, float] = {}
    for g, sub in tmp.groupby("_group", sort=True):
        out[str(g)] = train.fast_score_submission(sub)
    return out


def _prune_stash_dir(stash_dir: Path, *, keep: int) -> None:
    keep = max(0, int(keep))
    paths = sorted([p for p in stash_dir.glob("*.csv") if p.is_file()])
    if len(paths) <= keep:
        return

    def score_key(p: Path) -> tuple[int, str]:
        m = re.search(r"_s([0-9]+(?:\\.[0-9]+)?)_", p.name)
        if m:
            try:
                return (0, f"{float(m.group(1)):020.12f}")
            except Exception:
                pass
        return (1, p.name)

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
    p.add_argument("--input", default="submission_best.csv", help="Starting submission CSV.")
    p.add_argument("--donor", default=None, help="Fallback submission used to repair overlaps (defaults to --input).")
    p.add_argument(
        "--v18",
        default="data/submissions_more/a.exe",
        help="Path to Tree Packer v18 binary (Linux ELF). Copied into --workdir as ./v18.",
    )
    p.add_argument("--workdir", default="v18_work", help="Work directory (contains submission.csv).")
    p.add_argument("--wsl-distro", default=None, help="WSL distro name (e.g. Ubuntu). Auto-detect if omitted.")

    p.add_argument("--timeout", type=int, default=3600, help="Per-run timeout seconds (default 3600).")
    p.add_argument("--rounds", type=int, default=10, help="How many runs to attempt (default 10).")
    p.add_argument("--iters", type=int, default=15000, help="v18 -n iterations (default 15000).")
    p.add_argument("--restarts", type=int, default=16, help="v18 -r restarts (default 16).")
    p.add_argument(
        "--iters-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of -n values to cycle across rounds (cartesian product with --restarts-list).",
    )
    p.add_argument(
        "--restarts-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of -r values to cycle across rounds (cartesian product with --iters-list).",
    )
    p.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Limit OpenMP threads for v18 (helps control temperature when running multiple workers).",
    )
    p.add_argument("--fix-direction", action="store_true", help="Apply train.py fix-direction after each run.")
    p.add_argument("--delete-propagate", action="store_true", help="Apply train.py delete-propagate after each run.")
    p.add_argument("--insert-propagate", action="store_true", help="Apply train.py insert-propagate after each run.")
    p.add_argument("--insert-propagate-nmax", type=int, default=60)
    p.add_argument("--insert-propagate-topk", type=int, default=5)
    p.add_argument("--decimals", type=int, default=16, help="Decimals for fix-direction output rounding (default 16).")
    p.add_argument("--out", default="submission_v18_best.csv", help="Where to write the best found submission.")

    p.add_argument("--stash-dir", default=None, help="Optional directory to save extra valid candidates.")
    p.add_argument("--stash-every", type=int, default=0, help="If >0, also stash every N rounds (default 0).")
    p.add_argument("--stash-keep", type=int, default=200, help="Max number of stashed CSVs to keep (default 200).")
    p.add_argument(
        "--stash-min-improve",
        type=float,
        default=1e-10,
        help="Minimum per-group score improvement to consider a group improved for stashing (default 1e-10).",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input)
    donor_path = Path(args.donor) if args.donor else input_path
    v18_src = Path(args.v18)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    donor_df = _load_submission(donor_path)

    shutil.copy2(v18_src, workdir / "v18")

    input_df = _load_submission(input_path)
    input_score = train.score_submission(input_df)
    best_df = input_df
    best_score = input_score
    best_group_scores = _fast_group_score_map(best_df)

    if out_path.exists():
        try:
            prev_df = _load_submission(out_path)
            prev_score = train.score_submission(prev_df)
            if prev_score <= best_score:
                best_df = prev_df
                best_score = prev_score
                best_group_scores = _fast_group_score_map(best_df)
                print(f"Resuming from existing {out_path} score={prev_score:.12f} (input {input_score:.12f})")
        except Exception as e:
            print(f"Warning: failed to resume from {out_path}: {e}")

    _atomic_write_csv(best_df, out_path)
    _atomic_write_csv(best_df, workdir / "submission.csv")

    stash_dir = Path(args.stash_dir) if args.stash_dir else None
    if stash_dir is not None:
        stash_dir.mkdir(parents=True, exist_ok=True)

    iters_list = [int(args.iters)] if args.iters_list is None else [int(v) for v in args.iters_list]
    restarts_list = [int(args.restarts)] if args.restarts_list is None else [int(v) for v in args.restarts_list]
    combos = [(n, r) for n in iters_list for r in restarts_list]
    print(f"Start score: {best_score:.12f} | combos={len(combos)} | rounds={int(args.rounds)}")

    start = time.time()
    for k in range(int(args.rounds)):
        iters, restarts = combos[k % len(combos)]
        shutil.copy2(out_path, workdir / "submission.csv")
        try:
            rc, output = _run_v18(
                workdir=workdir,
                iters=iters,
                restarts=restarts,
                timeout_sec=int(args.timeout),
                wsl_distro=args.wsl_distro,
                omp_threads=args.omp_threads,
            )
        except subprocess.TimeoutExpired:
            print(f"[{k+1:03d}] TIMEOUT iters={iters} restarts={restarts}")
            continue

        if rc != 0:
            head = "\n".join((output or "").splitlines()[:30])
            print(f"[{k+1:03d}] v18 FAILED (rc={rc})\n{head}")
            if "libgomp.so.1" in output or "error while loading shared libraries" in output:
                print(
                    "\nUbuntu 里缺少 OpenMP 运行库。请在 Ubuntu 终端执行：\n"
                    "  sudo apt update\n"
                    "  sudo apt install -y libgomp1\n"
                )
            return 2

        out_file = workdir / "submission_out.csv"
        if not out_file.exists():
            head = "\n".join((output or "").splitlines()[:30])
            print(f"[{k+1:03d}] v18 produced no output file: {out_file}\n{head}")
            continue

        cand_df = _load_submission(out_file)
        if args.delete_propagate or args.insert_propagate:
            groups = train._submission_df_to_groups(cand_df)
            if args.delete_propagate:
                groups = train._delete_propagate_groups(groups, verbose=False)
            if args.insert_propagate:
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
            print(f"[{k+1:03d}] invalid iters={iters} restarts={restarts}: {e}")
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
            print(f"[{k+1:03d}] NEW BEST {best_score:.12f} t={dt/60:.1f}m")
        else:
            print(f"[{k+1:03d}] keep {best_score:.12f} (cand {cand_score:.12f}) t={dt/60:.1f}m")

        if stash_dir is not None:
            do_stash = False
            if improved_groups > 0:
                do_stash = True
            elif int(args.stash_every) > 0 and ((k + 1) % int(args.stash_every) == 0):
                do_stash = True
            if do_stash:
                name = f"cand_{k+1:04d}_s{cand_score:.12f}_g{improved_groups:03d}_it{iters}_r{restarts}.csv"
                _atomic_write_csv(cand_df, stash_dir / name)
                _prune_stash_dir(stash_dir, keep=int(args.stash_keep))

    print(f"Done. Best: {best_score:.12f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
