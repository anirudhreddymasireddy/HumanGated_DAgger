from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from src.parse_bag import parse_bag_pose_modelcars


def compute_absolute_states(
    poses: np.ndarray,
    commands: np.ndarray,
) -> np.ndarray:
    """Convert absolute pose/command data into absolute state features.

    Position (x, y) is used as-is in the map frame -- no relative/ego-motion
    differencing. Heading is encoded as sin(theta)/cos(theta) rather than a
    raw angle so the wraparound at +-pi doesn't look like a discontinuity to
    the network.

    No velocity features: this reads pose directly from /pose_modelcars (raw
    mocap), which carries position + orientation only, not velocity -- that
    only existed before because localization.py's Kalman filter computed it,
    and localization is no longer part of this pipeline.

    Note: this makes the model's input tied to absolute map-frame position, so
    it can only be expected to behave sensibly at (x, y, heading) combinations
    similar to what it was trained on -- it does not generalize the way a
    frame-relative motion representation would to unseen positions.
    """
    if len(poses) != len(commands):
        raise ValueError("poses and commands must have matching lengths")

    states = []
    for i in range(len(poses)):
        x = poses[i, 0]
        y = poses[i, 1]
        theta = poses[i, 2]
        delta, speed = commands[i]
        states.append([x, y, np.sin(theta), np.cos(theta), delta, speed])

    return np.asarray(states, dtype=np.float32)


def build_windows(
    states: np.ndarray, history_length: int = 8, state_features: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window training samples from a state sequence.

    Only the first `state_features` columns (x, y, sin(theta), cos(theta)) go
    into X. The trailing steer/speed columns are excluded from the window: at
    inference time that feature would have to be the model's own last
    prediction (no ground-truth feedback is wired in), which lets the network
    learn to copy its previous output instead of reacting to state (causal
    confusion), and collapses to a near-constant command once deployed.
    """
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if len(states) <= history_length:
        raise ValueError("not enough states to build any windows")

    X, y = [], []
    for i in range(history_length, len(states)):
        window = states[i - history_length : i, :state_features]
        label = states[i, 4:6]
        X.append(window)
        y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_train_val_windows(
    states: np.ndarray,
    history_length: int = 8,
    state_features: int = 4,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a state sequence into a time-based (non-shuffled) train/val holdout,
    then window each side separately.

    Splitting *after* windowing and shuffling the rows (the old approach) leaks:
    consecutive windows overlap by history_length-1 steps, so a "held out" row can
    be a near-duplicate of a training row, making val metrics optimistic. Splitting
    the raw time series first -- train on the first (1 - val_fraction) portion,
    validate on a contiguous trailing chunk the model never trained on -- gives an
    honest estimate of how the model does on driving it hasn't seen, at the cost of
    discarding up to history_length-1 samples at the split boundary.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    split = int(len(states) * (1.0 - val_fraction))
    train_states = states[:split]
    val_states = states[split:]

    X_train, y_train = build_windows(train_states, history_length=history_length, state_features=state_features)
    X_val, y_val = build_windows(val_states, history_length=history_length, state_features=state_features)

    return X_train, y_train, X_val, y_val


def save_dataset(X: np.ndarray, y: np.ndarray, prefix: str | Path) -> None:
    prefix = str(prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)


def append_dagger_round(
    demo_prefix: str | Path,
    dagger_prefix: str | Path,
    output_prefix: str | Path,
) -> None:
    """Concatenate demonstration data and DAgger correction data."""
    demo_prefix = str(demo_prefix)
    dagger_prefix = str(dagger_prefix)
    output_prefix = str(output_prefix)

    X_demo = np.load(f"{demo_prefix}_X.npy")
    y_demo = np.load(f"{demo_prefix}_y.npy")
    X_dagger = np.load(f"{dagger_prefix}_X.npy")
    y_dagger = np.load(f"{dagger_prefix}_y.npy")

    X_combined = np.concatenate([X_demo, X_dagger], axis=0)
    y_combined = np.concatenate([y_demo, y_dagger], axis=0)

    save_dataset(X_combined, y_combined, output_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sliding-window imitation learning datasets from a bag")
    parser.add_argument("--bag_path", required=True)
    parser.add_argument("--output_prefix", default="data/demo")
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of the bag (by time, trailing chunk) held out for validation. "
        "Set to 0 to disable splitting and write a single {output_prefix}_X.npy/_y.npy instead.",
    )
    parser.add_argument(
        "--target_rigid_body",
        default="6",
        help="Which /pose_modelcars rigid body to read (matches localization.py's own "
        "target_rigid_body parameter). The topic carries every tracked body each message; "
        "this filters down to the one that's the car.",
    )
    args = parser.parse_args()

    poses, commands = parse_bag_pose_modelcars(args.bag_path, target_rigid_body=args.target_rigid_body)
    states = compute_absolute_states(poses, commands)

    if args.val_fraction <= 0.0:
        X, y = build_windows(states, history_length=args.history_length)
        save_dataset(X, y, args.output_prefix)
        print(f"Saved {len(X)} samples to {args.output_prefix}_X.npy and {args.output_prefix}_y.npy")
        return

    X_train, y_train, X_val, y_val = build_train_val_windows(
        states, history_length=args.history_length, val_fraction=args.val_fraction
    )
    save_dataset(X_train, y_train, f"{args.output_prefix}_train")
    save_dataset(X_val, y_val, f"{args.output_prefix}_val")
    print(
        f"Saved {len(X_train)} train samples to {args.output_prefix}_train_X.npy/_y.npy and "
        f"{len(X_val)} val samples to {args.output_prefix}_val_X.npy/_y.npy "
        f"({int((1 - args.val_fraction) * 100)}:{int(args.val_fraction * 100)} time-based split)"
    )


if __name__ == "__main__":
    main()
