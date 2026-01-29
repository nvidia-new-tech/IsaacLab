# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot device interface for reading leader arm joint positions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    from lerobot.robots.so_follower import SO100Follower, SO101Follower, SOFollowerRobotConfig

    LEROBOT_AVAILABLE = True
except ImportError:
    try:
        from lerobot.robots.so_follower import SO100Follower, SO101Follower, SOFollowerRobotConfig

        LEROBOT_AVAILABLE = True
    except ImportError:
        SO100Follower = None
        SO101Follower = None
        SOFollowerRobotConfig = None
        LEROBOT_AVAILABLE = False

from .device_base import DeviceBase


class LeRobotDevice(DeviceBase):
    """LeRobot device that reads leader arm joint angles via lerobot.Robot."""

    _JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    _GRIPPER_NAME = "gripper"

    def __init__(
        self,
        port: str,
        robot_type: str = "soarm101",
        *,
        calibrate: bool = True,
        robot_id: str | None = None,
        calibration_dir: str | Path | None = None,
    ):
        """Initialize the LeRobot device interface.

        Args:
            port: Serial port for the LeRobot device.
            robot_type: Robot type identifier. Defaults to "soarm101".
            calibrate: Whether to run calibration if no calibration is found.
            robot_id: Optional robot id to select a specific calibration file.
            calibration_dir: Optional directory to store calibration files.
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot dependency not available. Install with `pip install lerobot`.")
        super().__init__()
        self._port = port
        self._robot_type = robot_type.lower()
        self._robot = self._create_robot(self._robot_type, port, robot_id, calibration_dir)
        self._additional_callbacks: dict[Any, Callable] = {}
        # Avoid interactive calibration prompts by default.
        if hasattr(self._robot, "connect"):
            self._robot.connect(calibrate=calibrate)
        # For leader arm use, keep motors torque-disabled to avoid locking.
        if hasattr(self._robot, "bus") and hasattr(self._robot.bus, "disable_torque"):
            self._robot.bus.disable_torque()
        if hasattr(self._robot, "is_calibrated") and not self._robot.is_calibrated:
            raise RuntimeError(
                "LeRobot has no calibration registered. "
                "Re-run with calibrate=True or provide a calibration_dir/robot_id."
            )

    def __del__(self):
        if hasattr(self._robot, "disconnect"):
            self._robot.disconnect()

    def reset(self):
        """Reset the device state if supported."""
        if hasattr(self._robot, "reset"):
            self._robot.reset()

    def add_callback(self, key: Any, func: Callable):
        """Add callbacks for compatibility with the teleop interface."""
        self._additional_callbacks[key] = func

    def advance(self) -> dict[str, Any]:
        """Read leader arm joint angles from the device.

        Returns:
            Dict with:
                - joint_pos: 5-DoF arm joint positions.
                - gripper: 1-DoF gripper position.
        """
        data = self._robot.get_observation()
        joint_pos, gripper = self._extract_joint_state(data)
        return {"joint_pos": joint_pos, "gripper": gripper}

    @staticmethod
    def _create_robot(
        robot_type: str, port: str, robot_id: str | None = None, calibration_dir: str | Path | None = None
    ):
        calibration_path = Path(calibration_dir) if calibration_dir is not None else None
        if robot_type in {"soarm101", "so101", "so101_follower"}:
            return SO101Follower(SOFollowerRobotConfig(port=port, id=robot_id, calibration_dir=calibration_path))
        if robot_type in {"soarm100", "so100", "so100_follower"}:
            return SO100Follower(SOFollowerRobotConfig(port=port, id=robot_id, calibration_dir=calibration_path))
        raise ValueError(f"Unsupported LeRobot robot_type: {robot_type}")

    @staticmethod
    def _extract_joint_state(data: Any) -> tuple[Any, Any]:
        """Extract 5-DoF arm joints and 1-DoF gripper from read data."""
        if isinstance(data, dict) and all(f"{name}.pos" in data for name in LeRobotDevice._JOINT_ORDER):
            joint_pos = [data[f"{name}.pos"] for name in LeRobotDevice._JOINT_ORDER]
            gripper = data.get(f"{LeRobotDevice._GRIPPER_NAME}.pos")
            return joint_pos, gripper
        if isinstance(data, dict):
            if "joint_pos" in data and "gripper" in data:
                return data["joint_pos"], data["gripper"]
            for key in ("qpos", "joint_positions", "positions", "joints"):
                if key in data:
                    return LeRobotDevice._split_sequence(data[key])
        return LeRobotDevice._split_sequence(data)

    @staticmethod
    def _split_sequence(values: Any) -> tuple[Any, Any]:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            seq = values
        elif hasattr(values, "tolist"):
            seq = values.tolist()
        else:
            seq = list(values)
        if len(seq) < 6:
            raise ValueError(f"Expected at least 6 joint values, got {len(seq)}")
        return seq[:5], seq[5]
