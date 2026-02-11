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
        # Increase contact friction for more reliable grasping.
        self.sim.physics_material.static_friction = 2.0
        self.sim.physics_material.dynamic_friction = 1.5
        self.sim.physics_material.friction_combine_mode = "multiply"
        # Explicit PD gains (Kp/Kd) for joint-position control.
        self.scene.robot.actuators["arm"].effort_limit_sim = 30.0
        self.scene.robot.actuators["arm"].velocity_limit_sim = 6.0
        self.scene.robot.actuators["arm"].stiffness = {
            "shoulder_pan": 800.0,
            "shoulder_lift": 900.0,
            "elbow_flex": 700.0,
            "wrist_flex": 500.0,
            "wrist_roll": 300.0,
        }
        self.scene.robot.actuators["arm"].damping = {
            "shoulder_pan": 60.0,
            "shoulder_lift": 70.0,
            "elbow_flex": 55.0,
            "wrist_flex": 45.0,
            "wrist_roll": 35.0,
        }
        self.scene.robot.actuators["gripper"].effort_limit_sim = 400.0
        self.scene.robot.actuators["gripper"].velocity_limit_sim = 1.0
        self.scene.robot.actuators["gripper"].stiffness = 32000.0
        self.scene.robot.actuators["gripper"].damping = 1200.0
