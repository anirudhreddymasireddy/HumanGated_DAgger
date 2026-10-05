from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def find_intervention_segments(
    y: np.ndarray,
    z: np.ndarray,
    steer_tol: float = 0.1,
    speed_tol: float = 0.01,
    history_length: int = 8,
    merge_gap: int = 30,
) -> Tuple[np.ndarray, list[tuple[int, int]]]:
    """Find row ranges where the measured feedback (y) and commanded value (z)
    significantly disagree -- the HG-DAgger intervention/divergence segments.

    /ackermann_drive and /ackermann_drive_feedback's steering sign convention
    relative to each other has changed between recordings more than once
    (hardware/driver edits land between bags) -- sometimes they need negating
    to compare, sometimes they don't. Rather than hardcode one direction, this
    auto-detects it per call: whichever of z_steer or -z_steer gives the lower
    *median* |y_steer - z_steer| is taken as aligned, on the assumption that
    most of any recording is normal tracking (low disagreement), with
    divergence being the minority the threshold is meant to isolate. Picking
    the wrong direction would inflate the "normal" baseline difference across
    nearly the whole recording, which is exactly the symptom that surfaces it.

    A row is "diverged" when |y_steer - z_steer_aligned| > steer_tol or
    |y_speed - z_speed| > speed_tol. Each contiguous diverged run becomes one
    segment, extended backwards by `history_length` rows (the lead-in context),
    clamped at 0. During one real intervention the two streams often re-agree
    for a brief moment (e.g. steering sweeping through a matching value), which
    would fragment a single event into several segments -- so segments whose
    quiet gap is smaller than `merge_gap` rows are merged into one, as are any
    that touch or overlap after the back-extension.

    Returns (keep_mask, segments) where segments is a list of (start, end)
    half-open row ranges after extension/merging.
    """
    if len(y) != len(z):
        raise ValueError("y and z must have matching lengths")

    median_raw = np.median(np.abs(y[:, 0] - z[:, 0]))
    median_negated = np.median(np.abs(y[:, 0] - (-z[:, 0])))
    if median_negated < median_raw:
        z_steer_aligned = -z[:, 0]
        print(f"[extract_interventions] steering sign: negating z (median diff {median_negated:.4f} < raw {median_raw:.4f})")
    else:
        z_steer_aligned = z[:, 0]
        print(f"[extract_interventions] steering sign: using z as-is (median diff {median_raw:.4f} <= negated {median_negated:.4f})")

    diverged = (np.abs(y[:, 0] - z_steer_aligned) > steer_tol) | (
        np.abs(y[:, 1] - z[:, 1]) > speed_tol
    )

    segments: list[tuple[int, int]] = []
    n = len(diverged)
    i = 0
    while i < n:
        if not diverged[i]:
            i += 1
            continue
        start = i
        while i < n and diverged[i]:
            i += 1
        segments.append((max(0, start - history_length), i))

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    keep_mask = np.zeros(n, dtype=bool)
    for start, end in merged:
        keep_mask[start:end] = True

    return keep_mask, merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HG-DAgger intervention segments: rows where /ackermann_drive_feedback (y) "
        "and /ackermann_drive (z) significantly diverge, plus the history leading into each divergence."
    )
    parser.add_argument("--prefix", required=True, help="Input prefix for {prefix}_X.npy/_y.npy/_z.npy")
    parser.add_argument("--output_prefix", required=True, help="Where to save the extracted {out}_X/_y/_z.npy")
    parser.add_argument("--steer_tol", type=float, default=0.1)
    parser.add_argument("--speed_tol", type=float, default=0.01)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument(
        "--merge_gap",
        type=int,
        default=30,
        help="Diverged runs separated by fewer than this many quiet rows are treated as "
        "one intervention instead of fragmenting a single event into several segments.",
    )
    parser.add_argument(
        "--val_segments",
        type=int,
        default=2,
        help="How many of the detected segments (the trailing ones) go to val; the rest go to "
        "train. Split is by whole segments, never mid-segment. Set to 0 to write a single "
        "{out}_X/_y/_z.npy with everything instead.",
    )
    args = parser.parse_args()

    X = np.load(f"{args.prefix}_X.npy")
    y = np.load(f"{args.prefix}_y.npy")
    z = np.load(f"{args.prefix}_z.npy")
    if not (len(X) == len(y) == len(z)):
        raise ValueError(f"X/y/z row counts differ: {len(X)}/{len(y)}/{len(z)}")

    keep_mask, segments = find_intervention_segments(
        y, z,
        steer_tol=args.steer_tol,
        speed_tol=args.speed_tol,
        history_length=args.history_length,
        merge_gap=args.merge_gap,
    )

    print(f"total samples: {len(y)}")
    print(f"diverged segments found (after back-extension and merging): {len(segments)}")
    for start, end in segments:
        print(f"  rows [{start}, {end})  length {end - start}")
    print(f"samples kept: {keep_mask.sum()} ({keep_mask.mean() * 100:.1f}% of recording)")

    if keep_mask.sum() == 0:
        print("No divergence found -- nothing saved.")
        return

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    def _mask_for(segs: list[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(len(y), dtype=bool)
        for start, end in segs:
            mask[start:end] = True
        return mask

    if args.val_segments <= 0:
        np.save(f"{args.output_prefix}_X.npy", X[keep_mask])
        np.save(f"{args.output_prefix}_y.npy", y[keep_mask])
        np.save(f"{args.output_prefix}_z.npy", z[keep_mask])
        print(f"Saved to {args.output_prefix}_X.npy, _y.npy, _z.npy")
        return

    if args.val_segments >= len(segments):
        raise ValueError(
            f"--val_segments={args.val_segments} but only {len(segments)} segments were found; "
            "at least one must remain for training"
        )

    train_mask = _mask_for(segments[: -args.val_segments])
    val_mask = _mask_for(segments[-args.val_segments :])

    np.save(f"{args.output_prefix}_train_X.npy", X[train_mask])
    np.save(f"{args.output_prefix}_train_y.npy", y[train_mask])
    np.save(f"{args.output_prefix}_train_z.npy", z[train_mask])
    np.save(f"{args.output_prefix}_val_X.npy", X[val_mask])
    np.save(f"{args.output_prefix}_val_y.npy", y[val_mask])
    np.save(f"{args.output_prefix}_val_z.npy", z[val_mask])
    print(
        f"Saved {train_mask.sum()} train samples ({len(segments) - args.val_segments} segments) to "
        f"{args.output_prefix}_train_X/_y/_z.npy and {val_mask.sum()} val samples "
        f"({args.val_segments} segments) to {args.output_prefix}_val_X/_y/_z.npy"
    )


if __name__ == "__main__":
    main()
