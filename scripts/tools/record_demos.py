# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful.
                              (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib
import math

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, lerobot. Not all tasks support all built-ins."
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
    default="-110,-100,-100,95,-170,-10",
    help="Comma-separated left/deg-min values for joints (same order as center).",
)
parser.add_argument(
    "--lerobot_joint_right_deg",
    type=str,
    default="110,100,90,-95,170,100",
    help="Comma-separated right/deg-max values for joints (same order as center).",
)
parser.add_argument(
    "--lerobot_id",
    type=str,
    default=None,
    help="LeRobot robot id for calibration lookup (e.g. soarm101).",
)
parser.add_argument(
    "--lerobot_calibration_dir",
    type=str,
    default=None,
    help="Optional LeRobot calibration directory override.",
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
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

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import logging
import os
import time

import gymnasium as gym
import torch

import omni.ui as ui

from isaaclab.devices import LeRobotDevice, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

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
            span = center_raw - left_raw
            return 0.0 if span <= 0 else (raw - center_raw) * (abs(left_deg) / span)
        span = right_raw - center_raw
        return 0.0 if span <= 0 else (raw - center_raw) * (abs(right_deg) / span)
    if raw >= left_raw:
        return left_deg
    if raw <= right_raw:
        return right_deg
    if raw > center_raw:
        span = left_raw - center_raw
        return 0.0 if span <= 0 else (raw - center_raw) * (abs(left_deg) / span)
    span = center_raw - right_raw
    return 0.0 if span <= 0 else (raw - center_raw) * (abs(right_deg) / span)


def _get_stack_success_params(env, success_term: object | None) -> tuple[float, float, float, float | None]:
    xy_threshold = 0.04
    height_threshold = 0.005
    height_diff = 0.0468
    gripper_open_min = None
    params = None
    if success_term is not None:
        params = getattr(success_term, "params", None)
    if params is None and hasattr(env.cfg, "terminations"):
        params = getattr(env.cfg.terminations, "success", None)
        params = getattr(params, "params", None) if params is not None else None
    if isinstance(params, dict):
        xy_threshold = params.get("xy_threshold", xy_threshold)
        height_threshold = params.get("height_threshold", height_threshold)
        height_diff = params.get("height_diff", height_diff)
        gripper_open_min = params.get("gripper_open_min", gripper_open_min)
    return xy_threshold, height_threshold, height_diff, gripper_open_min


def _stack_success_conditions(env, success_term: object | None) -> tuple[bool, bool, bool, torch.Tensor | None, float | None]:
    robot = env.scene["robot"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    cube_3 = env.scene["cube_3"]

    xy_threshold, height_threshold, height_diff, gripper_open_min = _get_stack_success_params(env, success_term)
    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)
    xy_ok = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)

    h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
    h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)
    height_ok = torch.logical_and(h_dist_c12 - height_diff < height_threshold, pos_diff_c12[:, 2] < 0.0)
    height_ok = torch.logical_and(height_ok, h_dist_c23 - height_diff < height_threshold)
    height_ok = torch.logical_and(height_ok, pos_diff_c23[:, 2] < 0.0)

    gripper_ok = torch.ones_like(xy_ok)
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)
        gripper_ok = (suction_cup_status == -1).to(torch.float32).squeeze(1)
    elif hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        target = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        if len(gripper_joint_ids) == 1:
            if gripper_open_min is not None:
                gripper_ok = robot.data.joint_pos[:, gripper_joint_ids[0]] >= gripper_open_min
            else:
                gripper_ok = torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]], target, atol=0.0001, rtol=0.0001
                )
        elif len(gripper_joint_ids) >= 2:
            gripper_ok = torch.logical_and(
                torch.isclose(robot.data.joint_pos[:, gripper_joint_ids[0]], target, atol=0.0001, rtol=0.0001),
                torch.isclose(robot.data.joint_pos[:, gripper_joint_ids[1]], target, atol=0.0001, rtol=0.0001),
            )
        else:
            gripper_ok = torch.zeros_like(xy_ok)

    gripper_pos = None
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        if len(gripper_joint_ids) >= 1:
            gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]].clone()

    return (
        bool(xy_ok.any().item()),
        bool(height_ok.any().item()),
        bool(gripper_ok.any().item()),
        gripper_pos,
        gripper_open_min,
    )


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "lerobot":
                if not args_cli.port:
                    logger.error("LeRobot teleop requires --port to be specified (e.g. /dev/ttyACM0)")
                    exit(1)
                teleop_interface = LeRobotDevice(
                    port=args_cli.port,
                    calibrate=args_cli.lerobot_calibrate,
                    robot_id=args_cli.lerobot_id,
                    calibration_dir=args_cli.lerobot_calibration_dir,
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, handtracking, lerobot")
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    try:
        teleop_interface.add_callback("R", reset_recording_instance)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to add callback for key R: {e}")

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

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    last_success_log_time = 0.0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            action = teleop_interface.advance()
            is_lerobot_action = isinstance(action, dict) and "joint_pos" in action and "gripper" in action
            if is_lerobot_action:
                if args_cli.lerobot_mapping_calibrate:
                    raw_list = list(action["joint_pos"]) + [action["gripper"]]
                    raw_msg = ", ".join(f"{name}={val:7.2f}" for name, val in zip(lerobot_joint_names, raw_list))
                    print(f"[LEROBOT RAW] {raw_msg}")

                if lerobot_joint_ids is None:
                    robot = env.scene["robot"]
                    lerobot_joint_ids, _ = robot.find_joints(lerobot_joint_names)
                    lerobot_joint_limits = robot.data.soft_joint_pos_limits[0, lerobot_joint_ids].clone()

                arm_vals = torch.tensor(list(action["joint_pos"]), dtype=torch.float32, device=env.device)
                grip_val = torch.tensor([action["gripper"]], dtype=torch.float32, device=env.device)
                raw_vals = torch.cat([arm_vals, grip_val], dim=0)
                raw_norm_vals = raw_vals.clone()

                mins = lerobot_joint_limits[:, 0]
                maxs = lerobot_joint_limits[:, 1]
                in_range = torch.logical_and(raw_vals >= (mins - 0.1), raw_vals <= (maxs + 0.1)).all()
                if not in_range:
                    arm_norm = raw_vals[:-1].clamp(-100.0, 100.0)
                    grip_norm = raw_vals[-1].clamp(0.0, 100.0)
                    arm_scaled = mins[:-1] + (arm_norm + 100.0) * (maxs[:-1] - mins[:-1]) / 200.0
                    grip_scaled = mins[-1] + (grip_norm) * (maxs[-1] - mins[-1]) / 100.0
                    raw_vals = torch.cat([arm_scaled, grip_scaled.unsqueeze(0)], dim=0)

                if lerobot_calibrate_pending or lerobot_leader_zero is None or lerobot_sim_zero is None:
                    lerobot_leader_zero = raw_vals.clone()
                    lerobot_sim_zero = env.scene["robot"].data.joint_pos[0, lerobot_joint_ids].clone()
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
                    debug_degs = [None] * 6
                    for idx in range(6):
                        deg = _map_raw_to_deg(
                            raw_norm_vals[idx],
                            joint_left_raw[idx],
                            joint_center_raw[idx],
                            joint_right_raw[idx],
                            joint_left_deg[idx],
                            joint_right_deg[idx],
                        )
                        action[idx] = lerobot_sim_zero[idx] + math.radians(deg)
                        debug_degs[idx] = deg
                    if args_cli.lerobot_mapping_debug:
                        debug_msg = ", ".join(
                            f"{name} raw={raw_norm_vals[i]:7.2f} deg={debug_degs[i]:7.2f} rad={action[i]:7.3f}"
                            for i, name in enumerate(lerobot_joint_names)
                        )
                        print(f"[LEROBOT MAP] {debug_msg}")

                action = action.clamp(mins, maxs)
            elif not torch.is_tensor(action):
                action = torch.tensor(action, dtype=torch.float32, device=env.device)
            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            if "Stack" in args_cli.task:
                now = time.time()
                if now - last_success_log_time >= 5.0:
                    last_success_log_time = now
                    success = stack_mdp.cubes_stacked(env)
                    success_any = bool(success.any().item())
                    print(
                        f"[SUCCESS] cubes_stacked={success_any} "
                        f"(count={int(success.sum().item())}/{env.num_envs})"
                    )
                    xy_ok, height_ok, gripper_ok, gripper_pos, gripper_open_min = _stack_success_conditions(
                        env, success_term
                    )
                    print(f"[SUCCESS] xy_aligned={xy_ok}")
                    print(f"[SUCCESS] height_aligned={height_ok}")
                    print(f"[SUCCESS] gripper_open={gripper_ok}")
                    if gripper_pos is not None:
                        mean_pos = float(gripper_pos.mean().item())
                        min_pos = float(gripper_pos.min().item())
                        max_pos = float(gripper_pos.max().item())
                        print(
                            f"[SUCCESS] gripper_pos_mean={mean_pos:.4f} "
                            f"min={min_pos:.4f} max={max_pos:.4f} "
                            f"open_min={gripper_open_min}"
                        )

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
