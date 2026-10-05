from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a windowed X/y dataset to samples with positive steering, then "
        "split the result 85:15 into train/val."
    )
    parser.add_argument("--prefix", required=True, help="Input prefix for {prefix}_X.npy/_y.npy")
    parser.add_argument("--output_prefix", required=True, help="Where to save the filtered {out}_train/_val_X/_y.npy")
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of the filtered (positive-steering) samples held out for val. Split is "
        "by position (last fraction -> val), not shuffled -- same time-based convention used "
        "elsewhere in this pipeline, since shuffling risks leaking near-duplicate overlapping "
        "windows across the train/val boundary.",
    )
    args = parser.parse_args()

    X = np.load(f"{args.prefix}_X.npy")
    y = np.load(f"{args.prefix}_y.npy")
    if len(X) != len(y):
        raise ValueError(f"X/y row counts differ: {len(X)}/{len(y)}")

    keep_mask = y[:, 0] > 0
    print(f"total samples: {len(y)}")
    print(f"positive-steering samples kept: {keep_mask.sum()} ({keep_mask.mean() * 100:.1f}%)")

    X_kept = X[keep_mask]
    y_kept = y[keep_mask]

    if len(X_kept) == 0:
        print("No positive-steering samples found -- nothing saved.")
        return

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    split = int(len(X_kept) * (1.0 - args.val_fraction))
    X_train, y_train = X_kept[:split], y_kept[:split]
    X_val, y_val = X_kept[split:], y_kept[split:]

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.output_prefix}_train_X.npy", X_train)
    np.save(f"{args.output_prefix}_train_y.npy", y_train)
    np.save(f"{args.output_prefix}_val_X.npy", X_val)
    np.save(f"{args.output_prefix}_val_y.npy", y_val)
    print(
        f"Saved {len(X_train)} train samples to {args.output_prefix}_train_X/_y.npy and "
        f"{len(X_val)} val samples to {args.output_prefix}_val_X/_y.npy "
        f"({int((1 - args.val_fraction) * 100)}:{int(args.val_fraction * 100)} split)"
    )


if __name__ == "__main__":
    main()
