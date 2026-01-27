# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM101 robots.

The following configurations are available:

* :obj:`SO_ARM101_CFG`: SO101 robot arm configuration.

Reference: 
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Point to the data directory in the same robots folder
ROBOTS_DATA_DIR = Path(__file__).resolve().parent / "data"

##
# Configuration
##

SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        asset_path=str(ROBOTS_DATA_DIR / "so_arm101.urdf"),
        activate_contact_sensors=False, # set as false while waiting for capsule implementation
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),

    ),

    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": -0.0,
            "wrist_flex": 1.57,
            "wrist_roll": -0.0,
            "gripper": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=5.0,#base line 1.9
            velocity_limit_sim=4.0,#base line 1.5
            stiffness={
                "shoulder_pan": 400.0, #base line 200  # Highest - moves all mass
                "shoulder_lift": 600.0,#base line 170  # Slightly less than rotation
                "elbow_flex": 400.0, #base line 120 # Reduced based on less mass
                "wrist_flex": 200.0, #base line 80 # Reduced for less mass
                "wrist_roll": 100.0, #base line 50 # Low mass to move
            },
            damping={
                "shoulder_pan": 10.0, #base line 80
                "shoulder_lift": 15.0, #base line 65
                "elbow_flex": 10.0, #base line 45
                "wrist_flex": 5.0, #base line 30
                "wrist_roll": 2.0, #base line 20
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

