from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Negate the steering column (y[:, 0]) of a windowed dataset -- e.g. to "
        "correct a sign-convention mismatch between recordings (see check_steer_sign.py to "
        "diagnose whether a dataset actually needs this before running it)."
    )
    parser.add_argument("--prefix", required=True, help="Input prefix for {prefix}_X.npy/_y.npy")
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Where to save the result. Defaults to --prefix (overwrites y.npy in place, X.npy "
        "untouched since it isn't read). Pass a different prefix to keep the original and write "
        "a new copy instead (X.npy is copied over unchanged so the new prefix is self-contained).",
    )
    parser.add_argument(
        "--only",
        choices=["all", "positive", "negative"],
        default="all",
        help="'all' mirrors every row (full sign flip). 'positive'/'negative' only negate rows "
        "currently on that side of zero, leaving the other side untouched (one-directional, not "
        "a mirror -- see the difference discussed when this was last done by hand).",
    )
    args = parser.parse_args()

    y = np.load(f"{args.prefix}_y.npy")
    output_prefix = args.output_prefix or args.prefix
    in_place = output_prefix == args.prefix

    steer = y[:, 0]
    if args.only == "all":
        mask = np.ones(len(y), dtype=bool)
    elif args.only == "positive":
        mask = steer > 0
    else:
        mask = steer < 0

    n_flipped = int(mask.sum())
    y = y.copy()
    y[mask, 0] = -y[mask, 0]

    if not in_place:
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f"{args.prefix}_X.npy", f"{output_prefix}_X.npy")

    np.save(f"{output_prefix}_y.npy", y)

    print(f"negated {n_flipped} of {len(y)} rows (--only {args.only})")
    print(f"steering: mean={y[:,0].mean():+.4f} min={y[:,0].min():+.4f} max={y[:,0].max():+.4f}")
    if in_place:
        print(f"overwrote {output_prefix}_y.npy")
    else:
        print(f"saved {output_prefix}_X.npy (copy) and {output_prefix}_y.npy (negated)")


if __name__ == "__main__":
    main()
