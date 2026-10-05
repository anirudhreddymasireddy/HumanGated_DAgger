from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import MLPPolicy
from ros_node import build_state_vector


def _load_model(checkpoint_path: str | Path) -> tuple[MLPPolicy, np.ndarray, np.ndarray, int, int]:
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history_length = int(checkpoint.get("history_length", 8))
    input_features = int(checkpoint.get("input_features", 7))
    model = MLPPolicy(history_length=history_length, input_features=input_features)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    return model, mean, std, history_length, input_features


def simulate_robot_path(
    checkpoint_path: str | Path = "model_checkpoints/mlp_bc.pt",
    num_steps: int = 40,
    output_path: str | Path | None = "model_checkpoints/dummy_path.png",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model, mean, std, history_length, input_features = _load_model(checkpoint_path)

    poses: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    history: list[np.ndarray] = []
    current_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    poses.append(current_pose.copy())
    prev_command = np.array([0.0, 0.1], dtype=np.float32)

    for _ in range(num_steps):
        state = build_state_vector(current_pose)
        history.append(state)
        if len(history) > history_length:
            history.pop(0)

        if len(history) < history_length:
            padded = np.zeros((history_length, input_features), dtype=np.float32)
            padded[-len(history):] = np.asarray(history, dtype=np.float32)
            window = padded
        else:
            window = np.asarray(history[-history_length:], dtype=np.float32)

        window_flat = window.reshape(1, -1)
        window_norm = (window_flat - mean) / std
        window_norm = window_norm.reshape(1, history_length, input_features)
        with torch.no_grad():
            action = model(torch.from_numpy(window_norm)).numpy()[0]

        steering = float(np.clip(action[0], -0.35, 0.35))
        speed = float(np.clip(action[1], 0.0, 0.3))
        prev_command = np.array([steering, speed], dtype=np.float32)
        actions.append(prev_command.copy())

        dt = 0.1
        wheelbase = 0.3
        omega = speed * np.tan(steering) / wheelbase if abs(steering) > 1e-6 else 0.0
        next_x = current_pose[0] + speed * np.cos(current_pose[2]) * dt
        next_y = current_pose[1] + speed * np.sin(current_pose[2]) * dt
        next_theta = current_pose[2] + omega * dt
        next_pose = np.array([next_x, next_y, next_theta], dtype=np.float32)
        poses.append(next_pose.copy())
        current_pose = next_pose

    if output_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("matplotlib is required to visualize the path") from exc

        xs = [p[0] for p in poses]
        ys = [p[1] for p in poses]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.scatter(xs[0], ys[0], c="green", s=60, label="start")
        ax.scatter(xs[-1], ys[-1], c="red", s=60, label="end")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    return poses, actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a dummy kinematic-state rollout and visualize the robot path")
    parser.add_argument("--checkpoint", default="model_checkpoints/mlp_bc.pt")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="model_checkpoints/dummy_path.png")
    args = parser.parse_args()

    poses, actions = simulate_robot_path(
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        output_path=args.output,
    )
    print(f"Simulated {len(poses)} path points and {len(actions)} actions")
    if args.output:
        print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
