from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


@dataclass(frozen=True)
class GroupCheck:
    n: int
    min_gap_unscaled: float
    min_gap_scaled: float
    unsafe: bool


def _group_min_gap_scaled(df_group: pd.DataFrame, *, eps_scaled: float) -> float:
    placed = [
        train.ChristmasTree(str(r["x"])[1:], str(r["y"])[1:], str(r["deg"])[1:])
        for _, r in df_group.iterrows()
    ]
    polys = [t.polygon for t in placed]
    r_tree = STRtree(polys)

    min_d: Optional[float] = None
    seen: Set[Tuple[int, int]] = set()

    # Only search near neighbors: if the envelope of poly buffered by eps doesn't hit another poly,
    # then distance > eps by construction.
    for i, poly in enumerate(polys):
        env = poly.buffer(float(eps_scaled)).envelope
        for j in r_tree.query(env):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            other = polys[j]

            # intersects implies distance == 0; still compute distance for reporting.
            d = float(poly.distance(other))
            if min_d is None or d < min_d:
                min_d = d
                if min_d <= 0.0:
                    return 0.0

    # If nothing queried, distance is > eps_scaled; return eps_scaled + tiny (for reporting).
    if min_d is None:
        return float(eps_scaled) + 1.0
    return float(min_d)


def analyze_groups(
    df: pd.DataFrame,
    *,
    min_gap: float,
    only: Optional[Iterable[int]] = None,
) -> List[GroupCheck]:
    eps_scaled = float(min_gap) * float(train.SCALE_FACTOR)
    groups = train._submission_df_to_groups(df)
    out: List[GroupCheck] = []
    only_set = set(int(x) for x in only) if only is not None else None
    for n in sorted(groups.keys()):
        if only_set is not None and int(n) not in only_set:
            continue
        g = groups[int(n)].reset_index(drop=True)
        d_scaled = _group_min_gap_scaled(g, eps_scaled=eps_scaled)
        d_unscaled = float(d_scaled) / float(train.SCALE_FACTOR)
        out.append(
            GroupCheck(
                n=int(n),
                min_gap_unscaled=float(d_unscaled),
                min_gap_scaled=float(d_scaled),
                unsafe=bool(d_unscaled < float(min_gap)),
            )
        )
    return out


def repair_by_donor(
    candidate: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    min_gap: float,
    only: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    cand_groups = train._submission_df_to_groups(candidate)
    donor_groups = train._submission_df_to_groups(donor)

    checks = analyze_groups(candidate, min_gap=min_gap, only=only)
    repaired: List[int] = []
    for chk in checks:
        if not chk.unsafe:
            continue
        if chk.n not in donor_groups:
            continue
        cand_groups[chk.n] = donor_groups[chk.n].copy().reset_index(drop=True)
        repaired.append(int(chk.n))

    out = train._groups_to_submission_df(cand_groups)
    return out, repaired


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Candidate submission CSV.")
    p.add_argument("--donor", required=True, help="Donor submission CSV to replace unsafe groups.")
    p.add_argument("--out", required=True, help="Output repaired CSV path.")
    p.add_argument(
        "--min-gap",
        type=float,
        default=1e-12,
        help="Mark groups with min polygon distance < this as unsafe (default 1e-12).",
    )
    p.add_argument(
        "--only-groups",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of group sizes to check/repair (e.g. 176 180).",
    )
    args = p.parse_args(argv)

    in_path = Path(args.input)
    donor_path = Path(args.donor)
    out_path = Path(args.out)

    cand = train.normalize_submission_df(pd.read_csv(in_path, dtype=str), path_hint=str(in_path))
    donor = train.normalize_submission_df(pd.read_csv(donor_path, dtype=str), path_hint=str(donor_path))

    min_gap = float(args.min_gap)
    only = args.only_groups

    before_score = None
    try:
        before_score = float(train.score_submission(cand))
    except Exception:
        pass

    fixed, repaired = repair_by_donor(cand, donor, min_gap=min_gap, only=only)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(out_path, index=False)

    after_score = None
    try:
        after_score = float(train.score_submission(fixed))
    except Exception:
        pass

    # Report
    if repaired:
        repaired_str = ", ".join(str(x) for x in repaired)
        print(f"Repaired groups (min_gap<{min_gap:g}): {repaired_str}")
    else:
        print(f"No groups required repair (min_gap>={min_gap:g}).")

    if before_score is not None:
        print(f"Score before: {before_score:.12f}")
    if after_score is not None:
        print(f"Score after:  {after_score:.12f}")
    print(f"Wrote: {out_path}")

    # Exit non-zero if still unsafe (helps in automation).
    checks = analyze_groups(fixed, min_gap=min_gap, only=only)
    still_bad = [c.n for c in checks if c.unsafe]
    if still_bad:
        print(f"WARNING: still unsafe groups: {still_bad}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

