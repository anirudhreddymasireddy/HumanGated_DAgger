import numpy as np

from visualize_dummy_path import simulate_robot_path


def test_simulate_robot_path_returns_positions_and_actions() -> None:
    poses, actions = simulate_robot_path(
        checkpoint_path="model_checkpoints/mlp_bc.pt",
        num_steps=5,
        output_path=None,
    )

    assert len(poses) == 6
    assert len(actions) == 5
    assert all(np.isfinite(p).all() for p in poses)
    assert all(np.isfinite(a).all() for a in actions)
