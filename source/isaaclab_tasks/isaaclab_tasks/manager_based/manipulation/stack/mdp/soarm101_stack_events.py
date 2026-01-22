# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event handlers for SO-ARM101 stack task."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the default pose for SO-ARM101 robot in all environments.
    
    Args:
        env: The environment.
        env_ids: The environment indices.
        default_pose: The default joint positions for SO-ARM101 (7 joints: 6 arm + 1 gripper).
        asset_cfg: The scene entity configuration.
    """
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize SO-ARM101 joint positions by adding Gaussian noise.
    
    This function adds Gaussian noise to the arm joints but keeps the gripper
    joint fixed at its default position (single-joint gripper for SO-ARM101).
    
    Args:
        env: The environment.
        env_ids: The environment indices.
        mean: The mean of the Gaussian distribution.
        std: The standard deviation of the Gaussian distribution.
        asset_cfg: The scene entity configuration.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits (only for arm joints, not gripper)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos[:, :joint_pos_limits.shape[1]] = joint_pos[:, :joint_pos_limits.shape[1]].clamp_(
        joint_pos_limits[..., 0], 
        joint_pos_limits[..., 1]
    )

    # Don't noise the gripper pose (last joint only for SO-ARM101)
    joint_pos[:, -1] = asset.data.default_joint_pos[env_ids, -1]

    # Set into the physics simulation
    # Only set arm joints via position target (first N joints where N = num of limits)
    num_arm_joints = joint_pos_limits.shape[1]
    asset.set_joint_position_target(joint_pos[:, :num_arm_joints], env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel[:, :num_arm_joints], env_ids=env_ids)
    # Write only arm joint state to sim
    asset.write_joint_state_to_sim(joint_pos[:, :num_arm_joints], joint_vel[:, :num_arm_joints], env_ids=env_ids)


def sample_random_color(base=(0.75, 0.75, 0.75), variation=0.1):
    """Generate a randomized color while maintaining brightness.
    
    Generates a randomized color that stays close to the base color while preserving 
    overall brightness. The relative balance between the R, G, and B components is 
    maintained by ensuring that the sum of random offsets is zero.

    Args:
        base: The base RGB color with each component between 0 and 1.
        variation: Maximum deviation to sample for each channel before balancing.

    Returns:
        A new RGB color with balanced random variation.
    """
    # Generate random offsets for each channel in the range [-variation, variation]
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    # Compute the average offset
    avg_offset = sum(offsets) / 3
    # Adjust offsets so their sum is zero (maintaining brightness)
    balanced_offsets = [offset - avg_offset for offset in offsets]

    # Apply the balanced offsets to the base color and clamp each channel between 0 and 1
    new_color = tuple(max(0, min(1, base_component + offset)) for base_component, offset in zip(base, balanced_offsets))

    return new_color


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    default_intensity: float = 3000.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize scene lighting dome light parameters.
    
    Args:
        env: The environment.
        env_ids: The environment indices.
        intensity_range: The range of light intensity values.
        color_variation: The color variation for randomization.
        textures: List of texture file paths for dome light.
        default_intensity: The default light intensity.
        default_color: The default light color.
        default_texture: The default texture file path.
        asset_cfg: The scene entity configuration.
    """
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(default_intensity)

    color_attr = light_prim.GetAttribute("inputs:color")
    color_attr.Set(default_color)

    texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
    texture_file_attr.Set(default_texture)

    if not hasattr(env.cfg, "eval_mode") or not env.cfg.eval_mode:
        return

    if env.cfg.eval_type in ["light_intensity", "all"]:
        # Sample new light intensity
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        # Set light intensity to light prim
        intensity_attr.Set(new_intensity)

    if env.cfg.eval_type in ["light_color", "all"]:
        # Sample new light color
        new_color = sample_random_color(base=default_color, variation=color_variation)
        # Set light color to light prim
        color_attr.Set(new_color)

    if env.cfg.eval_type in ["light_texture", "all"]:
        # Sample new light texture (background)
        new_texture = random.sample(textures, 1)[0]
        # Set light texture to light prim
        texture_file_attr.Set(new_texture)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """Sample random object poses in the workspace.
    
    Args:
        num_objects: Number of objects to sample poses for.
        min_separation: Minimum separation distance between objects.
        pose_range: Dictionary defining position and rotation ranges.
        max_sample_tries: Maximum number of sampling attempts.

    Returns:
        List of sampled poses.
    """
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    """Randomize object poses in the environment.
    
    Args:
        env: The environment.
        env_ids: The environment indices.
        asset_cfgs: List of scene entity configurations for objects.
        min_separation: Minimum separation distance between objects.
        pose_range: Dictionary defining position and rotation ranges.
        max_sample_tries: Maximum number of sampling attempts.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )
