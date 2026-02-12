# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad, lerobot. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--port", type=str, default=None, help="Serial port for LeRobot (e.g. /dev/ttyACM0).")
parser.add_argument(
    "--lerobot_calibrate",
    action="store_true",
    default=True,
    help="Run LeRobot calibration if no calibration is found.",
)
parser.add_argument(
    "--lerobot_calibrate_at_start",
    action="store_true",
    default=True,
    help="Calibrate initial pose offset between leader and sim on start.",
)
parser.add_argument(
    "--lerobot_arm_gain",
    type=float,
    default=1.0,
    help="Scale factor for leader arm joint deltas before applying to the sim.",
)
parser.add_argument(
    "--lerobot_gripper_gain",
    type=float,
    default=1.0,
    help="Scale factor for leader gripper delta before applying to the sim.",
)
parser.add_argument(
    "--lerobot_shoulder_pan_center_raw",
    type=float,
    default=0.0,
    help="Leader shoulder_pan raw value at physical center (normalized).",
)
parser.add_argument(
    "--lerobot_shoulder_pan_left_raw",
    type=float,
    default=-100.0,
    help="Leader shoulder_pan raw value at left limit (normalized).",
)
parser.add_argument(
    "--lerobot_shoulder_pan_right_raw",
    type=float,
    default=100.0,
    help="Leader shoulder_pan raw value at right limit (normalized).",
)
parser.add_argument(
    "--lerobot_shoulder_pan_left_deg",
    type=float,
    default=-90.0,
    help="Virtual shoulder_pan degrees to map at left raw limit.",
)
parser.add_argument(
    "--lerobot_shoulder_pan_right_deg",
    type=float,
    default=90.0,
    help="Virtual shoulder_pan degrees to map at right raw limit.",
)
parser.add_argument(
    "--lerobot_mapping_calibrate",
    action="store_true",
    default=False,
    help="Print raw lerobot joint values for calibration.",
)
parser.add_argument(
    "--lerobot_mapping_debug",
    action="store_true",
    default=False,
    help="Print lerobot raw-to-action mapping per joint.",
)
parser.add_argument(
    "--lerobot_joint_center_raw",
    type=str,
    default="5.66,0.67,3.21,1.03,1.04,7.11",
    help="Comma-separated center raw values for joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.",
)
parser.add_argument(
    "--lerobot_joint_left_raw",
    type=str,
    default="-100.00,-99.75,-99.55,-98.54,92.82,0.00",
    help="Comma-separated left/raw-min values for joints (same order as center).",
)
parser.add_argument(
    "--lerobot_joint_right_raw",
    type=str,
    default="100.00,99.92,99.19,73.68,-92.38,100.00",
    help="Comma-separated right/raw-max values for joints (same order as center).",
)
parser.add_argument(
    "--lerobot_joint_left_deg",
    type=str,
    default="-110,-100,-100,-95,-160,-10",
    help="Comma-separated target degrees at left/raw-min for joints (same order as center).",
)
parser.add_argument(
    "--lerobot_joint_right_deg",
    type=str,
    default="110,100,90,95,160,100",
    help="Comma-separated target degrees at right/raw-max for joints (same order as center).",
)
parser.add_argument("--lerobot_id", type=str, default=None, help="Optional LeRobot calibration id.")
parser.add_argument(
    "--lerobot_calibration_dir",
    type=str,
    default=None,
    help="Directory to store LeRobot calibration files.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch

from isaaclab.devices import (
    LeRobotDevice,
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

def _parse_csv_floats(value: str | None, expected: int, name: str) -> list[float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated values, got {len(parts)}: {value}")
    return [float(p) for p in parts]


def _map_raw_to_deg(raw: float, left_raw: float, center_raw: float, right_raw: float, left_deg: float, right_deg: float) -> float:
    if left_raw < right_raw:
        if raw <= left_raw:
            return left_deg
        if raw >= right_raw:
            return right_deg
        if raw < center_raw:
            span = left_raw - center_raw
            return 0.0 if span == 0 else (raw - center_raw) / span * left_deg
        span = right_raw - center_raw
        return 0.0 if span == 0 else (raw - center_raw) / span * right_deg
    if raw >= left_raw:
        return left_deg
    if raw <= right_raw:
        return right_deg
    if raw > center_raw:
        span = left_raw - center_raw
        return 0.0 if span == 0 else (raw - center_raw) / span * left_deg
    span = right_raw - center_raw
    return 0.0 if span == 0 else (raw - center_raw) / span * right_deg


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    lerobot_joint_ids = None
    lerobot_joint_limits = None
    lerobot_leader_zero = None
    lerobot_sim_zero = None
    lerobot_calibrate_pending = args_cli.lerobot_calibrate_at_start
    lerobot_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    joint_center_raw = _parse_csv_floats(args_cli.lerobot_joint_center_raw, 6, "--lerobot_joint_center_raw")
    joint_left_raw = _parse_csv_floats(args_cli.lerobot_joint_left_raw, 6, "--lerobot_joint_left_raw")
    joint_right_raw = _parse_csv_floats(args_cli.lerobot_joint_right_raw, 6, "--lerobot_joint_right_raw")
    joint_left_deg = _parse_csv_floats(args_cli.lerobot_joint_left_deg, 6, "--lerobot_joint_left_deg")
    joint_right_deg = _parse_csv_floats(args_cli.lerobot_joint_right_deg, 6, "--lerobot_joint_right_deg")
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "lerobot":
                if not args_cli.port:
                    logger.error("LeRobot teleop requires --port to be specified (e.g. /dev/ttyACM0)")
                    env.close()
                    simulation_app.close()
                    return
                teleop_interface = LeRobotDevice(
                    port=args_cli.port,
                    calibrate=args_cli.lerobot_calibrate,
                    robot_id=args_cli.lerobot_id,
                    calibration_dir=args_cli.lerobot_calibration_dir,
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    keyboard_reset_interface = None
    if args_cli.teleop_device.lower() != "keyboard":
        try:
            keyboard_reset_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
            for key, callback in teleoperation_callbacks.items():
                keyboard_reset_interface.add_callback(key, callback)
            keyboard_reset_interface.reset()
        except Exception as e:
            logger.warning(f"Failed to enable global keyboard reset: {e}")
            keyboard_reset_interface = None

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    success_prev = None
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()
                if args_cli.teleop_device.lower() == "lerobot" and isinstance(action, dict):
                    if args_cli.lerobot_mapping_calibrate:
                        raw_list = list(action["joint_pos"]) + [action["gripper"]]
                        raw_msg = ", ".join(f"{name}={val:7.2f}" for name, val in zip(lerobot_joint_names, raw_list))
                        print(f"[LEROBOT RAW] {raw_msg}")

                if args_cli.teleop_device.lower() == "lerobot":
                    if isinstance(action, dict):
                        # Map LeRobot normalized joint values to sim joint limits.
                        if lerobot_joint_ids is None:
                            robot = env.scene["robot"]
                            joint_names = [
                                "shoulder_pan",
                                "shoulder_lift",
                                "elbow_flex",
                                "wrist_flex",
                                "wrist_roll",
                                "gripper",
                            ]
                            lerobot_joint_ids, _ = robot.find_joints(joint_names)
                            lerobot_joint_limits = robot.data.soft_joint_pos_limits[0, lerobot_joint_ids].clone()

                        arm_vals = torch.tensor(list(action["joint_pos"]), dtype=torch.float32, device=env.device)
                        grip_val = torch.tensor([action["gripper"]], dtype=torch.float32, device=env.device)
                        raw_vals = torch.cat([arm_vals, grip_val], dim=0)
                        raw_norm_vals = raw_vals.clone()

                        # If values already look like radians (within limits), pass through.
                        mins = lerobot_joint_limits[:, 0]
                        maxs = lerobot_joint_limits[:, 1]
                        in_range = torch.logical_and(raw_vals >= (mins - 0.1), raw_vals <= (maxs + 0.1)).all()
                        if not in_range:
                            # Map arm from [-100, 100] and gripper from [0, 100] into joint limits.
                            arm_norm = raw_vals[:-1].clamp(-100.0, 100.0)
                            grip_norm = raw_vals[-1].clamp(0.0, 100.0)
                            arm_scaled = mins[:-1] + (arm_norm + 100.0) * (maxs[:-1] - mins[:-1]) / 200.0
                            grip_scaled = mins[-1] + (grip_norm) * (maxs[-1] - mins[-1]) / 100.0
                            raw_vals = torch.cat([arm_scaled, grip_scaled.unsqueeze(0)], dim=0)

                        if lerobot_calibrate_pending or lerobot_leader_zero is None or lerobot_sim_zero is None:
                            lerobot_leader_zero = raw_vals.clone()
                            robot = env.scene["robot"]
                            lerobot_sim_zero = robot.data.joint_pos[0, lerobot_joint_ids].clone()
                            lerobot_calibrate_pending = False

                        gains = torch.tensor(
                            [args_cli.lerobot_arm_gain] * 5 + [args_cli.lerobot_gripper_gain],
                            dtype=torch.float32,
                            device=env.device,
                        )
                        action = lerobot_sim_zero + (raw_vals - lerobot_leader_zero) * gains

                        # Per-joint asymmetric mapping using measured raw limits (optional).
                        if all(
                            v is not None
                            for v in (joint_center_raw, joint_left_raw, joint_right_raw, joint_left_deg, joint_right_deg)
                        ):
                            pan_deg = None
                            debug_degs = [None] * 6
                            for idx in range(6):
                                raw_val = raw_norm_vals[idx]
                                left_raw = joint_left_raw[idx]
                                right_raw = joint_right_raw[idx]
                                center_raw = joint_center_raw[idx]
                                deg = _map_raw_to_deg(
                                    raw_val,
                                    left_raw,
                                    center_raw,
                                    right_raw,
                                    joint_left_deg[idx],
                                    joint_right_deg[idx],
                                )
                                action[idx] = lerobot_sim_zero[idx] + math.radians(deg)
                                debug_degs[idx] = deg
                                if idx == 0:
                                    pan_deg = deg
                            if args_cli.lerobot_mapping_debug:
                                debug_msg = ", ".join(
                                    f"{name} raw={raw_norm_vals[i]:7.2f} deg={debug_degs[i]:7.2f} rad={action[i]:7.3f}"
                                    for i, name in enumerate(lerobot_joint_names)
                                )
                                print(f"[LEROBOT MAP] {debug_msg}")
                        else:
                            # Shoulder pan: asymmetric mapping using measured raw limits.
                            raw_pan = raw_norm_vals[0]
                            left_raw = args_cli.lerobot_shoulder_pan_left_raw
                            right_raw = args_cli.lerobot_shoulder_pan_right_raw
                            center_raw = args_cli.lerobot_shoulder_pan_center_raw
                            if raw_pan <= left_raw:
                                pan_deg = args_cli.lerobot_shoulder_pan_left_deg
                            elif raw_pan >= right_raw:
                                pan_deg = args_cli.lerobot_shoulder_pan_right_deg
                            elif raw_pan < center_raw:
                                span = center_raw - left_raw
                                pan_deg = (
                                    0.0
                                    if span <= 0
                                    else (raw_pan - center_raw)
                                    * (abs(args_cli.lerobot_shoulder_pan_left_deg) / span)
                                )
                            else:
                                span = right_raw - center_raw
                                pan_deg = (
                                    0.0
                                    if span <= 0
                                    else (raw_pan - center_raw)
                                    * (abs(args_cli.lerobot_shoulder_pan_right_deg) / span)
                            )
                            action[0] = lerobot_sim_zero[0] + math.radians(pan_deg)

                        action = action.clamp(mins, maxs)
                    elif not torch.is_tensor(action):
                        action = torch.tensor(action, dtype=torch.float32, device=env.device)

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                else:
                    env.sim.render()

                if "Stack" in args_cli.task:
                    success = stack_mdp.cubes_stacked(env)
                    success_any = bool(success.any().item())
                    if success_prev is None or success_any != success_prev:
                        success_prev = success_any
                        print(
                            f"[SUCCESS] cubes_stacked={success_any} "
                            f"(count={int(success.sum().item())}/{env.num_envs})"
                        )

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
