# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg


@configclass
class Soarm101CubeStackLeRobotEnvCfg(stack_joint_pos_env_cfg.Soram101CubeStackEnvCfg):
    """SOARM101 stack config for LeRobot joint-position teleop."""

    def __post_init__(self):
        super().__post_init__()

        # Direct joint-position control for 1:1 mapping from leader arm.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"], scale=1.0, use_default_offset=False
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"], scale=1.0, use_default_offset=False
        )

        # Disable URDF joint drives to rely on actuator PD control.
        self.scene.robot.spawn.joint_drive = None
        # Explicit PD gains (Kp/Kd) for joint-position control.
        self.scene.robot.actuators["arm"].stiffness = {
            "shoulder_pan": 400.0,
            "shoulder_lift": 600.0,
            "elbow_flex": 400.0,
            "wrist_flex": 200.0,
            "wrist_roll": 100.0,
        }
        self.scene.robot.actuators["arm"].damping = {
            "shoulder_pan": 10.0,
            "shoulder_lift": 15.0,
            "elbow_flex": 10.0,
            "wrist_flex": 5.0,
            "wrist_roll": 2.0,
        }
        self.scene.robot.actuators["gripper"].stiffness = 60.0
        self.scene.robot.actuators["gripper"].damping = 20.0
