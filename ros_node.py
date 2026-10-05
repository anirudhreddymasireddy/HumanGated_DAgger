from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

try:
    import torch
except ImportError:  # pragma: no cover - depends on environment
    torch = None

try:
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
    from mocap4r2_msgs.msg import RigidBodies
except ImportError:  # pragma: no cover - depends on environment
    rclpy = None
    Node = object  # type: ignore[assignment]
    AckermannDrive = None  # type: ignore[assignment]
    AckermannDriveStamped = None  # type: ignore[assignment]
    RigidBodies = None  # type: ignore[assignment]

from model import MLPPolicy


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def build_state_vector(
    current_pose: np.ndarray | None,
) -> np.ndarray:
    """Build the 4-dimensional absolute state vector expected by the trained policy:
    x, y, sin(theta), cos(theta) -- matching src/build_dataset.py's
    compute_absolute_states, which is what the checkpoint was trained on.

    No velocity features: pose comes from /pose_modelcars (raw mocap), which
    carries position + orientation only. Heading is encoded as sin/cos rather
    than a raw angle so the wraparound at +-pi doesn't look like a discontinuity
    to the network. Deliberately excludes the previous command: at inference time
    that value would have to be the model's own last prediction (no ground-truth
    feedback is wired in), which lets the network learn to copy its previous
    output instead of reacting to state (causal confusion) and collapses to a
    near-constant command in deployment.

    Note: because this uses absolute position, the model can only be expected to
    behave sensibly at (x, y, heading) combinations similar to what it was trained
    on -- it does not generalize the way a frame-relative motion representation would
    to unseen positions.
    """
    if current_pose is None:
        current_pose = np.zeros(3, dtype=np.float32)

    current_pose = np.asarray(current_pose, dtype=np.float32)

    x, y, theta = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])

    state = np.array(
        [x, y, math.sin(theta), math.cos(theta)],
        dtype=np.float32,
    )
    return state


def build_command_message(
    steering_angle: float,
    speed: float,
    stamp: Any | None = None,
) -> Any:
    if AckermannDriveStamped is not None:
        msg = AckermannDriveStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        return msg

    if AckermannDrive is None:
        raise RuntimeError("AckermannDrive message type is unavailable")

    msg = AckermannDrive()
    msg.steering_angle = steering_angle
    msg.speed = speed
    return msg


def _extract_pose(msg: Any, target_rigid_body: str) -> np.ndarray | None:
    """Pick the target rigid body's pose out of a mocap4r2_msgs/RigidBodies message."""
    for rb in msg.rigidbodies:
        name = getattr(rb, "rigid_body_name", None)
        if name is None or str(name) != target_rigid_body:
            continue
        p = rb.pose.position
        orientation = rb.pose.orientation
        yaw = quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        return np.array([p.x, p.y, yaw], dtype=np.float32)
    return None


class ImitationControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("imitation_controller")

        self.declare_parameter("checkpoint", "model_checkpoints_dagger2/mlp_bc.pt")
        self.declare_parameter("command_topic", "/ackermann_drive")
        self.declare_parameter("target_rigid_body", "6")
        #self.declare_parameter("publish_rate_hz",50.0)

        checkpoint_path = Path(self.get_parameter("checkpoint").value).expanduser()
        self.target_rigid_body = str(self.get_parameter("target_rigid_body").value)

        if torch is None:
            raise RuntimeError("PyTorch is required to run the imitation controller")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # history_length/input_features come from the checkpoint itself, not a
        # separately-declared parameter: a mismatch here previously caused the
        # published model_state to load into a differently-shaped MLPPolicy.
        self.history_length = int(checkpoint["history_length"])
        self.input_features = int(checkpoint["input_features"])

        self.model = MLPPolicy(history_length=self.history_length, input_features=self.input_features)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.history: list[np.ndarray] = []
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32)

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.cmd_pub = None
        if AckermannDriveStamped is not None:
            self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("command_topic").value, qos_reliable)
        else:
            self.get_logger().warning("AckermannDriveStamped message type is unavailable; commands will not be published")

        self.create_timer(0.03 , self.publish_command) # / self.get_parameter("publish_rate_hz").value
        self.state_sub = None
        if RigidBodies is not None:
            self.state_sub = self.create_subscription(RigidBodies, '/pose_modelcars', self.state_cb, qos_be)
            self.get_logger().info("Subscribed to /pose_modelcars using RigidBodies")
        else:
            self.get_logger().warning("mocap4r2_msgs/RigidBodies is unavailable; the node will not receive inputs")

    def _handle_state(self, msg: Any) -> None:
        pose = _extract_pose(msg, self.target_rigid_body)
        if pose is None:
            return

        state = build_state_vector(pose)
        self.history.append(state)
        if len(self.history) > self.history_length:
            self.history.pop(0)
        self.get_logger().info(f"Received state; history size={len(self.history)}")

    def state_cb(self, msg: Any) -> None:
        self._handle_state(msg)

    def publish_command(self) -> None:
        if len(self.history) < self.history_length:
            return

        window = np.asarray(self.history[-self.history_length :], dtype=np.float32)
        window_flat = window.reshape(1, -1)
        window_norm = (window_flat - self.mean) / self.std
        window_norm = window_norm.reshape(1, self.history_length, self.input_features)

        with torch.no_grad():
            action = self.model(torch.from_numpy(window_norm)).numpy()[0]

        steering_angle = float(np.clip(action[0], -0.5, 0.5))
        speed = float(np.clip(action[1], 0.0, 1.0))
        if self.cmd_pub is not None:
            msg = build_command_message(
                steering_angle,
                speed,
                stamp=self.get_clock().now().to_msg(),
            )
            self.cmd_pub.publish(msg)
            if hasattr(msg, "drive"):
                steering_value = msg.drive.steering_angle
                speed_value = msg.drive.speed
            else:
                steering_value = msg.steering_angle
                speed_value = msg.speed
            self.get_logger().info(f"Publishing command: steering={steering_value:.3f}, speed={speed_value:.3f}")


def main() -> None:
    if rclpy is None:
        raise RuntimeError("rclpy is required to run the imitation controller")
    rclpy.init()
    node = ImitationControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if "ExternalShutdownException" not in str(type(exc).__name__):
            raise
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
