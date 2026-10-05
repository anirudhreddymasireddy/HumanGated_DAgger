from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConstantActionBaseline(nn.Module):
    """A simple baseline that repeats the most recent observed action."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_action = x[:, -1, 6:8]
        return last_action


class MLPPolicy(nn.Module):
    """A compact MLP policy for ego-motion imitation learning."""

    def __init__(self, history_length: int = 8, input_features: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.history_length = history_length
        self.input_features = input_features
        in_features = history_length * input_features

        # BatchNorm1d right on the raw flattened window: dx/dy (meters), dtheta
        # (radians), vx/vy (m/s), omega (rad/s) all live on different scales, so
        # this normalizes the input itself rather than relying only on the
        # dataset-level mean/std computed once at train time.
        self.input_norm = nn.BatchNorm1d(in_features)
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        # Linear -> NonLinear -> Linear heads instead of a single Linear, so
        # steering/speed each get their own small nonlinear readout off the
        # shared trunk features.
        self.steer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim ),
            nn.ReLU(),
            nn.Linear(hidden_dim , 1),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        x_norm = self.input_norm(x_flat)
        features = self.feature_extractor(x_norm)
        steering = self.steer_head(features).squeeze(-1)
        speed = self.speed_head(features).squeeze(-1)
        return torch.stack((steering, speed), dim=-1)


def load_dataset(
    data_dir: str | Path,
    history_length: int | None = None,
    data_prefix: str = "demo",
) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"{data_prefix}_X.npy")
    y = np.load(data_dir / f"{data_prefix}_y.npy")

    if history_length is not None and X.shape[1] != history_length:
        raise ValueError(f"Expected history length {history_length}, got {X.shape[1]}")

    return X, y


def _normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X_flat = X.reshape(len(X), -1)
    X_norm = (X_flat - mean) / std
    return X_norm.reshape(X.shape[0], *X.shape[1:])


def train_model(
    data_dir: str | Path,
    output_dir: str | Path,
    history_length: int = 8,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    data_prefix: str = "demo",
    val_prefix: str | None = None,
) -> dict[str, Any]:
    """Train a policy checkpoint.

    If `val_prefix` is given, train/val come from separate pre-split files
    (e.g. a time-based holdout built by the caller) instead of a random
    shuffle-split of `data_prefix`. A random split of sliding windows leaks:
    adjacent windows overlap by history_length-1 steps, so "held out" rows can
    be near-duplicates of training rows, making val metrics optimistic.
    """
    input_features: int
    if val_prefix is not None:
        X_train, y_train = load_dataset(data_dir, history_length=history_length, data_prefix=data_prefix)
        X_val, y_val = load_dataset(data_dir, history_length=history_length, data_prefix=val_prefix)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        input_features = X_train.shape[-1]
    else:
        X, y = load_dataset(data_dir, history_length=history_length, data_prefix=data_prefix)

        X = X.astype(np.float32)
        y = y.astype(np.float32)
        input_features = X.shape[-1]

        rng = np.random.default_rng(seed)
        indices = np.arange(len(X))
        rng.shuffle(indices)

        split = int(0.9 * len(indices))
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

    flat_train = X_train.reshape(len(X_train), -1)
    mean = flat_train.mean(axis=0, keepdims=True)
    std = flat_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train_norm = _normalize_windows(X_train, mean, std)
    X_val_norm = _normalize_windows(X_val, mean, std)

    train_ds = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val))

    # drop_last: BatchNorm1d needs >1 sample per batch in train mode to compute
    # batch statistics; a trailing batch of size 1 would crash.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPPolicy(history_length=history_length, input_features=input_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Near convergence the epoch-to-epoch val loss just oscillates in a noise
    # band, so the final epoch is an arbitrary draw from it -- keep the weights
    # from whichever epoch actually validated best instead. copy.deepcopy, not a
    # reference: state_dict() tensors are views into the live model and would
    # silently keep training.
    import copy

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        if val_losses and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:03d} | val_loss={epoch_val_loss:.4f}")

    if best_state is None:  # no val batches at all; fall back to final weights
        best_state = model.state_dict()
        best_val_loss = float("nan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mlp_bc.pt"
    torch.save(
        {
            "model_state": best_state,
            "history_length": history_length,
            "input_features": input_features,
            "mean": mean,
            "std": std,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path} (best epoch {best_epoch:03d}, val_loss={best_val_loss:.4f})")
    return {
        "checkpoint_path": str(checkpoint_path),
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if mean is None or std is None:
        X_norm = X
    else:
        X_norm = _normalize_windows(X, mean, std)

    dataset = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())

    preds_arr = np.concatenate(preds, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    mse = float(np.mean((preds_arr - targets_arr) ** 2))
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}
