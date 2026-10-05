from pathlib import Path

import numpy as np

from src.build_dataset import (
    append_dagger_round,
    build_train_val_windows,
    build_windows,
    compute_absolute_states,
    save_dataset,
)


def test_compute_absolute_states_and_windows() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    commands = np.array([[0.0, 0.5], [0.1, 0.6], [0.2, 0.7]], dtype=np.float32)

    states = compute_absolute_states(poses, commands)
    assert states.shape == (3, 6)

    X, y = build_windows(states, history_length=1)
    assert X.shape == (2, 1, 4)
    assert y.shape == (2, 2)


def test_build_train_val_windows_is_time_based_not_shuffled() -> None:
    n = 20
    poses = np.column_stack(
        [np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
    )
    commands = np.tile(np.array([0.1, 0.5], dtype=np.float32), (n, 1))
    states = compute_absolute_states(poses, commands)

    X_train, y_train, X_val, y_val = build_train_val_windows(states, history_length=2, val_fraction=0.15)

    # split point: int(20 * 0.85) = 17 -> train windows from states[:17], val windows from states[17:]
    assert X_train.shape == (15, 2, 4)
    assert y_train.shape == (15, 2)
    assert X_val.shape == (1, 2, 4)
    assert y_val.shape == (1, 2)

    # val's x-position values must all come from the trailing (later-time) chunk, confirming
    # no shuffling occurred -- train and val don't share any of the same underlying rows
    assert X_train[:, :, 0].max() < X_val[:, :, 0].min()


def test_append_dagger_round(tmp_path: Path) -> None:
    X_demo = np.zeros((2, 1, 8), dtype=np.float32)
    y_demo = np.zeros((2, 2), dtype=np.float32)
    X_dagger = np.ones((1, 1, 8), dtype=np.float32)
    y_dagger = np.ones((1, 2), dtype=np.float32)

    save_dataset(X_demo, y_demo, tmp_path / "demo")
    save_dataset(X_dagger, y_dagger, tmp_path / "dagger")

    append_dagger_round(tmp_path / "demo", tmp_path / "dagger", tmp_path / "combined")

    X_combined = np.load(tmp_path / "combined_X.npy")
    y_combined = np.load(tmp_path / "combined_y.npy")

    assert X_combined.shape == (3, 1, 8)
    assert y_combined.shape == (3, 2)
