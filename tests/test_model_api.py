import numpy as np
import torch

from model import MLPPolicy, evaluate_model, load_dataset


def test_policy_output_shape() -> None:
    x = np.ones((3, 4, 8), dtype=np.float32)
    x_tensor = torch.from_numpy(x)

    policy = MLPPolicy(history_length=4, input_features=8)
    out = policy(x_tensor)

    assert out.shape == (3, 2)


def test_evaluate_model_returns_metrics(tmp_path) -> None:
    X = np.random.default_rng(0).standard_normal((8, 2, 8)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((8, 2)).astype(np.float32)

    metrics = evaluate_model(MLPPolicy(history_length=2, input_features=8), X, y)

    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_load_dataset_reads_numpy_files(tmp_path) -> None:
    X = np.ones((3, 2, 8), dtype=np.float32)
    y = np.ones((3, 2), dtype=np.float32)
    np.save(tmp_path / "demo_X.npy", X)
    np.save(tmp_path / "demo_y.npy", y)

    loaded_X, loaded_y = load_dataset(tmp_path)

    assert loaded_X.shape == X.shape
    assert loaded_y.shape == y.shape
