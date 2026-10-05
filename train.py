from __future__ import annotations

import argparse
from pathlib import Path

from model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an imitation-learning policy")
    parser.add_argument("--data_dir", default="data", help="Directory containing demo_X.npy and demo_y.npy")
    parser.add_argument("--output_dir", default="model_checkpoints", help="Where to save the checkpoint")
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256, help="Larger batches mean far fewer Python/dispatch calls per epoch on this CPU (~3x faster at 256 vs 64), same total compute")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data_prefix", default="demo", help="Prefix for the {prefix}_X.npy / {prefix}_y.npy dataset files")
    parser.add_argument("--val_prefix", default=None, help="If set, load a separate pre-split {prefix}_X.npy/_y.npy for validation instead of randomly splitting data_prefix")
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        history_length=args.history_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_prefix=args.data_prefix,
        val_prefix=args.val_prefix,
    )


if __name__ == "__main__":
    main()
