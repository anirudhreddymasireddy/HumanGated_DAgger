from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:  # pragma: no cover - depends on environment
    SequentialReader = None
    StorageOptions = None
    ConverterOptions = None
    deserialize_message = None
    get_message = None


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to yaw in radians."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _resolve_bag_uri(bag_path: str | Path) -> tuple[Path, str]:
    bag_path = Path(bag_path)
    if bag_path.is_dir():
        db_files = sorted(bag_path.glob("*.db3"))
        if not db_files:
            raise FileNotFoundError(f"No .db3 bag file found in directory: {bag_path}")
        return bag_path, str(db_files[0])
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return bag_path.parent, str(bag_path)
    if bag_path.exists():
        return bag_path.parent, str(bag_path)
    raise FileNotFoundError(f"Bag path not found: {bag_path}")


def parse_bag(bag_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ROS2 bag into pose, twist, and command arrays.

    The parser accepts the recorded topics for kinematic state and drive feedback.
    It returns arrays of timestamps, absolute pose (x, y, theta), twist (vx, vy, omega_z),
    and aligned commands (steering_angle, speed).
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

    pose_data: list[list[float]] = []
    twist_data: list[list[float]] = []
    command_data: list[list[float]] = []
    pose_times: list[float] = []
    command_times: list[float] = []

    while reader.has_next():
        topic, rawdata, timestamp_ns = reader.read_next()
        t = timestamp_ns * 1e-9

        if topic.endswith("/kinematic_state") or topic == "kinematic_state":
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            pose = msg.pose_with_covariance.pose
            twist = msg.twist_with_covariance.twist
            orientation = pose.orientation
            yaw = quaternion_to_yaw(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
            pose_data.append([pose.position.x, pose.position.y, yaw])
            twist_data.append([twist.linear.x, twist.linear.y, twist.angular.z])
            pose_times.append(t)

        elif topic.endswith("/ackermann_drive_feedback") or topic.endswith("/ackermann_cmd") or topic in {"ackermann_drive_feedback", "ackermann_cmd"}:
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            if hasattr(msg, "drive"):
                command_data.append([msg.drive.steering_angle, msg.drive.speed])
            else:
                command_data.append([msg.steering_angle, msg.speed])
            command_times.append(t)

    if not pose_data or not command_data:
        raise ValueError(f"No usable pose/command messages were found in bag: {bag_uri}")

    pose_times = np.asarray(pose_times, dtype=np.float32)
    pose_arr = np.asarray(pose_data, dtype=np.float32)
    twist_arr = np.asarray(twist_data, dtype=np.float32)
    command_times = np.asarray(command_times, dtype=np.float32)
    command_arr = np.asarray(command_data, dtype=np.float32)

    aligned_commands = []
    for t in pose_times:
        idx = int(np.argmin(np.abs(command_times - t)))
        aligned_commands.append(command_arr[idx])

    return pose_times, pose_arr, twist_arr, np.asarray(aligned_commands, dtype=np.float32)


def parse_bag_ordered(bag_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ROS2 bag into pose, twist, and command arrays, aligned by arrival
    order instead of by timestamp value.

    parse_bag() aligns /kinematic_state to /ackermann_drive_feedback by nearest
    recorded timestamp. That breaks if /kinematic_state's own timestamp is
    unreliable (observed on a real recording: frozen for long stretches, then
    jumping by a fixed amount) -- nearest-timestamp matching then aliases many
    state samples onto the same single command. rosbag2 still writes messages
    in true arrival order even when a message's own timestamp field is bad, so
    this instead pairs each feedback message with whichever /kinematic_state
    message most recently arrived before it, by read position, not by
    timestamp value. This is the alignment build_dataset.py uses by default.

    Returns (poses, twists, commands), one row per /ackermann_drive_feedback
    message -- there's no meaningful separate "pose_times" here since ordering
    is positional, not time-value-based.
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

    latest_pose: list[float] | None = None
    latest_twist: list[float] | None = None
    pose_data: list[list[float]] = []
    twist_data: list[list[float]] = []
    command_data: list[list[float]] = []

    while reader.has_next():
        topic, rawdata, _timestamp_ns = reader.read_next()

        if topic.endswith("/kinematic_state") or topic == "kinematic_state":
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            pose = msg.pose_with_covariance.pose
            twist = msg.twist_with_covariance.twist
            orientation = pose.orientation
            yaw = quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
            latest_pose = [pose.position.x, pose.position.y, yaw]
            latest_twist = [twist.linear.x, twist.linear.y, twist.angular.z]

        elif topic.endswith("/ackermann_drive_feedback") or topic.endswith("/ackermann_cmd") or topic in {"ackermann_drive_feedback", "ackermann_cmd"}:
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            if latest_pose is None:
                continue  # no state observed yet; nothing to pair this command with
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            if hasattr(msg, "drive"):
                command = [msg.drive.steering_angle, msg.drive.speed]
            else:
                command = [msg.steering_angle, msg.speed]

            pose_data.append(latest_pose)
            twist_data.append(latest_twist)
            command_data.append(command)

    if not pose_data:
        raise ValueError(f"No usable paired pose/command messages were found in bag: {bag_uri}")

    return (
        np.asarray(pose_data, dtype=np.float32),
        np.asarray(twist_data, dtype=np.float32),
        np.asarray(command_data, dtype=np.float32),
    )


def normalize_rigid_body_name(name: object) -> str:
    """Match localization.py's rigid-body-name normalization (strips leading zeros)."""
    return str(name).strip().lstrip("0") or "0"


def parse_bag_pose_modelcars(
    bag_path: str | Path,
    target_rigid_body: str = "6",
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a ROS2 bag into pose and command arrays, reading raw mocap pose
    directly from /pose_modelcars (mocap4r2_msgs/RigidBodies) instead of
    localization.py's /kinematic_state.

    /pose_modelcars carries position + orientation only, no velocity -- so
    unlike parse_bag()/parse_bag_ordered() there is no twist array here.
    RigidBodies is a list of every tracked body each message, so entries are
    filtered down to `target_rigid_body` (matching localization.py's own
    target_rigid_body parameter and name-normalization).

    Alignment with /ackermann_drive_feedback is by nearest *embedded* message
    timestamp (each message's own header.stamp, populated by
    AckermannDriveStamped/RigidBodies' own publisher clock) -- not the bag's
    recording-time metadata, which is what caused /kinematic_state's
    timestamp-freeze bug via localization.py's old republishing pipeline.
    Reading /pose_modelcars directly sidesteps that pipeline entirely.

    /pose_modelcars publishes faster than /ackermann_drive_feedback, so the
    two topics have different message counts. Matching walks whichever list
    is smaller (usually the feedback one) and, for each of its entries, picks
    the single closest-in-time entry from the larger list -- so every output
    row is a genuinely-closest pairing, and the result has exactly
    min(len(poses), len(commands)) rows, no threshold, nothing dropped. If a
    feedback message has no header field (plain AckermannDrive instead of
    AckermannDriveStamped), the bag's own recorded time is used for it
    instead -- that field was independently verified healthy for this topic.

    Returns (poses, commands): poses is [x, y, yaw], commands is
    [steering_angle, speed].
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

    pose_times: list[float] = []
    pose_data: list[list[float]] = []
    command_times: list[float] = []
    command_data: list[list[float]] = []

    while reader.has_next():
        topic, rawdata, timestamp_ns = reader.read_next()

        if topic.endswith("/pose_modelcars") or topic == "pose_modelcars":
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            for rb in msg.rigidbodies:
                if normalize_rigid_body_name(rb.rigid_body_name) != target:
                    continue
                p = rb.pose.position
                o = rb.pose.orientation
                yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
                pose_times.append(t)
                pose_data.append([p.x, p.y, yaw])
                break

        elif topic.endswith("/ackermann_drive_feedback") or topic.endswith("/ackermann_cmd") or topic in {"ackermann_drive_feedback", "ackermann_cmd"}:
            msg_type = topic_types.get(topic)
            if not msg_type or get_message is None or deserialize_message is None:
                continue
            msg_cls = get_message(msg_type)
            msg = deserialize_message(rawdata, msg_cls)
            if hasattr(msg, "header"):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                t = timestamp_ns * 1e-9
            if hasattr(msg, "drive"):
                command = [msg.drive.steering_angle, msg.drive.speed]
            else:
                command = [msg.steering_angle, msg.speed]
            command_times.append(t)
            command_data.append(command)

    if not pose_data or not command_data:
        raise ValueError(
            f"No usable pose/command messages were found in bag: {bag_uri} "
            f"(target_rigid_body={target_rigid_body!r})"
        )

    pose_times_arr = np.asarray(pose_times, dtype=np.float64)
    pose_arr = np.asarray(pose_data, dtype=np.float32)
    command_times_arr = np.asarray(command_times, dtype=np.float64)
    command_arr = np.asarray(command_data, dtype=np.float32)

    if len(pose_times_arr) <= len(command_times_arr):
        kept_poses = pose_arr
        kept_commands = np.stack(
            [command_arr[int(np.argmin(np.abs(command_times_arr - t)))] for t in pose_times_arr]
        )
    else:
        kept_commands = command_arr
        kept_poses = np.stack(
            [pose_arr[int(np.argmin(np.abs(pose_times_arr - t)))] for t in command_times_arr]
        )

    return (
        np.asarray(kept_poses, dtype=np.float32),
        np.asarray(kept_commands, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a ROS2 bag into numpy arrays")
    parser.add_argument("--bag_path", required=True)
    parser.add_argument("--output_prefix", default="data/demo")
    args = parser.parse_args()

    _, poses, twists, commands = parse_bag(args.bag_path)
    np.save(f"{args.output_prefix}_poses.npy", poses)
    np.save(f"{args.output_prefix}_twists.npy", twists)
    np.save(f"{args.output_prefix}_commands.npy", commands)
    print(f"Saved parsed arrays to {args.output_prefix}_poses.npy, {args.output_prefix}_twists.npy, and {args.output_prefix}_commands.npy")


if __name__ == "__main__":
    main()
