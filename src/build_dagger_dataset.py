from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from src.build_dataset import save_dataset
from src.parse_bag import (
    ConverterOptions,
    SequentialReader,
    StorageOptions,
    _resolve_bag_uri,
    deserialize_message,
    get_message,
    normalize_rigid_body_name,
    quaternion_to_yaw,
)


def parse_bag_dagger(
    bag_path: str | Path,
    target_rigid_body: str = "6",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a DAgger ROS2 bag into three timestamp-matched arrays:

    poses    -- [x, y, yaw]              from /pose_modelcars (target rigid body)
    feedback -- [steering_angle, speed]  from /ackermann_drive_feedback (measured)
    drive    -- [steering_angle, speed]  from /ackermann_drive (commanded; the
                joystick/policy stream, kept separately so human interventions
                can be told apart from the measured motor response)

    Matching uses each message's own embedded header.stamp (bag-recorded time
    as fallback for headerless messages). The topic with the fewest messages
    is the base; every base entry is paired with its single closest-in-time
    entry from each other topic, so all three outputs have exactly
    min(count(pose), count(feedback), count(drive)) rows and row i of each
    array refers to the same moment.
    """
    if SequentialReader is None or StorageOptions is None or ConverterOptions is None:
        raise ImportError(
            "rosbag2_py is required to parse these ROS2 bag files. Source your ROS workspace first."
        )

    bag_dir, bag_uri = _resolve_bag_uri(bag_path)
    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    target = normalize_rigid_body_name(target_rigid_body)

    def _drive_fields(msg) -> list[float]:
        if hasattr(msg, "drive"):
            return [msg.drive.steering_angle, msg.drive.speed]
        return [msg.steering_angle, msg.speed]

    def _stamp(msg, bag_time_ns: int) -> float:
        if hasattr(msg, "header"):
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return bag_time_ns * 1e-9

    pose_times: list[float] = []
    pose_data: list[list[float]] = []
    fb_times: list[float] = []
    fb_data: list[list[float]] = []
    drive_times: list[float] = []
    drive_data: list[list[float]] = []

    while reader.has_next():
        topic, rawdata, timestamp_ns = reader.read_next()
        msg_type = topic_types.get(topic)
        if not msg_type or get_message is None or deserialize_message is None:
            continue

        if topic.endswith("/pose_modelcars") or topic == "pose_modelcars":
            msg = deserialize_message(rawdata, get_message(msg_type))
            t = _stamp(msg, timestamp_ns)
            for rb in msg.rigidbodies:
                if normalize_rigid_body_name(rb.rigid_body_name) != target:
                    continue
                p = rb.pose.position
                o = rb.pose.orientation
                pose_times.append(t)
                pose_data.append([p.x, p.y, quaternion_to_yaw(o.x, o.y, o.z, o.w)])
                break

        elif topic.endswith("/ackermann_drive_feedback") or topic == "ackermann_drive_feedback":
            msg = deserialize_message(rawdata, get_message(msg_type))
            fb_times.append(_stamp(msg, timestamp_ns))
            fb_data.append(_drive_fields(msg))

        elif topic.endswith("/ackermann_drive") or topic == "ackermann_drive":
            msg = deserialize_message(rawdata, get_message(msg_type))
            drive_times.append(_stamp(msg, timestamp_ns))
            drive_data.append(_drive_fields(msg))

    if not pose_data or not fb_data or not drive_data:
        raise ValueError(
            f"Bag {bag_uri} is missing one of the required topics "
            f"(/pose_modelcars for rigid body {target_rigid_body!r}, "
            f"/ackermann_drive_feedback, /ackermann_drive)"
        )

    streams = {
        "pose": (np.asarray(pose_times, dtype=np.float64), np.asarray(pose_data, dtype=np.float32)),
        "feedback": (np.asarray(fb_times, dtype=np.float64), np.asarray(fb_data, dtype=np.float32)),
        "drive": (np.asarray(drive_times, dtype=np.float64), np.asarray(drive_data, dtype=np.float32)),
    }

    base_name = min(streams, key=lambda k: len(streams[k][0]))
    base_times = streams[base_name][0]

    aligned: dict[str, np.ndarray] = {}
    for name, (times, values) in streams.items():
        if name == base_name:
            aligned[name] = values
        else:
            idx = np.array([int(np.argmin(np.abs(times - t))) for t in base_times])
            aligned[name] = values[idx]

    return aligned["pose"], aligned["feedback"], aligned["drive"]


def compute_states_with_drive(
    poses: np.ndarray, feedback: np.ndarray
) -> np.ndarray:
    """[x, y, sin(theta), cos(theta), fb_steer, fb_speed] per row, matching
    src/build_dataset.py's compute_absolute_states feature layout."""
    if len(poses) != len(feedback):
        raise ValueError("poses and feedback must have matching lengths")
    x = poses[:, 0]
    y = poses[:, 1]
    theta = poses[:, 2]
    return np.column_stack(
        [x, y, np.sin(theta), np.cos(theta), feedback[:, 0], feedback[:, 1]]
    ).astype(np.float32)


def build_windows_with_drive(
    states: np.ndarray,
    drive: np.ndarray,
    history_length: int = 8,
    state_features: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows like src/build_dataset.py's build_windows, plus a third
    z array: the /ackermann_drive commanded value at the same row the label y
    comes from. X, y, and z always have the same number of rows."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if len(states) != len(drive):
        raise ValueError("states and drive must have matching lengths")
    if len(states) <= history_length:
        raise ValueError("not enough states to build any windows")

    X, y, z = [], [], []
    for i in range(history_length, len(states)):
        X.append(states[i - history_length : i, :state_features])
        y.append(states[i, 4:6])
        z.append(drive[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(z, dtype=np.float32),
    )


def save_dagger_dataset(X: np.ndarray, y: np.ndarray, z: np.ndarray, prefix: str | Path) -> None:
    save_dataset(X, y, prefix)
    np.save(f"{prefix}_z.npy", z)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build X/y/z sliding-window npy files from a DAgger bag: X = pose windows, "
        "y = /ackermann_drive_feedback (measured), z = /ackermann_drive (commanded)."
    )
    parser.add_argument("--bag_path", required=True)
    parser.add_argument("--output_prefix", default="data/dagger")
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of the bag (by time, trailing chunk) held out for validation. "
        "Set to 0 to write a single {output_prefix}_X/_y/_z.npy instead.",
    )
    parser.add_argument("--target_rigid_body", default="6")
    args = parser.parse_args()

    poses, feedback, drive = parse_bag_dagger(args.bag_path, target_rigid_body=args.target_rigid_body)
    states = compute_states_with_drive(poses, feedback)

    if args.val_fraction <= 0.0:
        X, y, z = build_windows_with_drive(states, drive, history_length=args.history_length)
        save_dagger_dataset(X, y, z, args.output_prefix)
        print(f"Saved {len(X)} samples to {args.output_prefix}_X.npy, _y.npy, and _z.npy")
        return

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    split = int(len(states) * (1.0 - args.val_fraction))
    X_train, y_train, z_train = build_windows_with_drive(
        states[:split], drive[:split], history_length=args.history_length
    )
    X_val, y_val, z_val = build_windows_with_drive(
        states[split:], drive[split:], history_length=args.history_length
    )
    save_dagger_dataset(X_train, y_train, z_train, f"{args.output_prefix}_train")
    save_dagger_dataset(X_val, y_val, z_val, f"{args.output_prefix}_val")
    print(
        f"Saved {len(X_train)} train samples to {args.output_prefix}_train_X/_y/_z.npy and "
        f"{len(X_val)} val samples to {args.output_prefix}_val_X/_y/_z.npy "
        f"({int((1 - args.val_fraction) * 100)}:{int(args.val_fraction * 100)} time-based split)"
    )


if __name__ == "__main__":
    main()
