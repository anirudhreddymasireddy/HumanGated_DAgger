from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MLPPolicy, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trained imitation policy outputs")
    parser.add_argument("--checkpoint", default="model_checkpoints/mlp_bc.pt")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--history_length", type=int, default=None, help="Defaults to the checkpoint's own history_length; only set this to intentionally check against a mismatched window size")
    parser.add_argument("--data_prefix", default="demo")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", default="model_checkpoints/visualization.png")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = MLPPolicy(history_length=checkpoint["history_length"], input_features=checkpoint["input_features"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    history_length = args.history_length if args.history_length is not None else checkpoint["history_length"]
    X, y = load_dataset(args.data_dir, history_length=history_length, data_prefix=args.data_prefix)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    X_flat = X.reshape(len(X), -1)
    X_norm = (X_flat - mean) / std
    X_norm = X_norm.reshape(X.shape[0], *X.shape[1:])

    with torch.no_grad():
        preds = model(torch.from_numpy(X_norm[: args.limit])).numpy()

    targets = y[: args.limit]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(targets[:, 0], label="target steering", color="tab:blue", alpha=0.7)
    axes[0].plot(preds[:, 0], label="pred steering", color="tab:red", linestyle="--")
    axes[0].set_ylabel("steering")
    axes[0].set_title("Trained policy output vs target")
    axes[0].legend()

    axes[1].plot(targets[:, 1], label="target speed", color="tab:green", alpha=0.7)
    axes[1].plot(preds[:, 1], label="pred speed", color="tab:orange", linestyle="--")
    axes[1].set_ylabel("speed")
    axes[1].set_xlabel("sample")
    axes[1].legend()

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
