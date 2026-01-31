import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train


def _to_groups(path: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, dtype=str)
    df = train.normalize_submission_df(df, path_hint=path)
    # {n:int -> df[x,y,deg]} (no id column)
    return train._submission_df_to_groups(df)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--donor", required=True, help="Fallback submission used to replace overlapping groups.")
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    inp = Path(args.input)
    donor = Path(args.donor)
    out = Path(args.out)

    inp_groups = _to_groups(str(inp))
    donor_groups = _to_groups(str(donor))

    repaired = 0
    out_groups: dict[int, pd.DataFrame] = {}

    for n in range(1, 201):
        if n not in inp_groups:
            raise ValueError(f"Missing group {n:03d} in input: {inp}")
        if n not in donor_groups:
            raise ValueError(f"Missing group {n:03d} in donor: {donor}")

        cand = inp_groups[n]
        if train._group_has_overlap(cand):
            cand = donor_groups[n]
            repaired += 1

        out_groups[n] = cand[["x", "y", "deg"]].reset_index(drop=True)

    if args.verbose:
        print(f"Repaired overlapping groups from donor: {repaired}")
        # (missing groups would have raised)

    df_out = train._groups_to_submission_df(out_groups)
    df_out.to_csv(out, index=False)
    print(f"Wrote {out} (rows={len(df_out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
