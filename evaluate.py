from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import MLPPolicy, evaluate_model, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an imitation-learning checkpoint")
    parser.add_argument("--checkpoint", default="model_checkpoints/mlp_bc.pt")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--history_length", type=int, default=None, help="Defaults to the checkpoint's own history_length; only set this to intentionally check against a mismatched window size")
    parser.add_argument(
        "--data_prefix",
        default="demo_val",
        help="Dataset prefix to evaluate on -- point this at a val-split prefix built by "
        "'src.build_dataset' (e.g. demo3_val) for an honest holdout evaluation. The whole "
        "dataset at this prefix is evaluated; splitting now happens once at build time, not here.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = MLPPolicy(history_length=checkpoint["history_length"], input_features=checkpoint["input_features"])
    model.load_state_dict(checkpoint["model_state"])

    history_length = args.history_length if args.history_length is not None else checkpoint["history_length"]
    X, y = load_dataset(args.data_dir, history_length=history_length, data_prefix=args.data_prefix)
    X = X.astype("float32")
    y = y.astype("float32")

    metrics = evaluate_model(
        model=model,
        X=X,
        y=y,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
