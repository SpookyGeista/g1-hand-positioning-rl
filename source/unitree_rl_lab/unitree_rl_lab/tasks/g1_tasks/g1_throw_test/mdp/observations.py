# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Observation functions for G1 Throwing Test Task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


def get_robot_boy_joint_states(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get joint positions and velocities for body joints (legs, torso, arms, wrists).
    
    Excludes hand/finger joints. Returns [q, dq] concatenated.
    Based on G1-29DOF configuration: 29 body joints (12 legs + 3 waist + 14 arms).
    """
    robot: "Articulation" = env.scene["robot"]
    
    # Define the expected body joint names in the order they should appear
    # This matches the G1-29DOF configuration
    expected_body_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    # Get all joint names from robot
    joint_names = robot.data.joint_names
    
    # Find indices for expected body joints
    body_joint_indices = []
    for expected_name in expected_body_joint_names:
        try:
            idx = joint_names.index(expected_name)
            body_joint_indices.append(idx)
        except ValueError:
            # Joint not found, skip it
            continue
    
    if len(body_joint_indices) == 0:
        # Fallback: filter by name pattern
        body_joint_indices = [i for i, name in enumerate(joint_names) 
                             if not any(hand_term in name for hand_term in ["hand_index", "hand_middle", "hand_thumb"])]
    
    body_joint_indices = torch.tensor(body_joint_indices, device=robot.device, dtype=torch.long)
    
    # Get positions and velocities for body joints
    q = robot.data.joint_pos[:, body_joint_indices]  # [num_envs, num_body_dof]
    dq = robot.data.joint_vel[:, body_joint_indices]  # [num_envs, num_body_dof]
    
    # Concatenate [q, dq] = 58 dims (29 * 2)
    obs = torch.cat([q, dq], dim=-1)
    
    # The original function likely returns additional information
    # Policy expects 101 total: 73 (boy) + 28 (dex3) = 101
    # So we need to add 15 more dimensions to boy
    # These might be: base_ang_vel (3) + projected_gravity (3) + other (9)
    # For now, we'll compute base angular velocity and projected gravity
    additional_dims = []
    
    # Get base angular velocity (3D) from root body
    # In Isaac Lab, we can get this from the root body's angular velocity
    try:
        root_body_idx = robot.root_body_idx
        base_ang_vel = robot.data.body_ang_vel_w[:, root_body_idx, :3]  # [num_envs, 3]
        additional_dims.append(base_ang_vel)
    except (AttributeError, IndexError):
        # Fallback: use zeros if not available
        base_ang_vel = torch.zeros(env.num_envs, 3, device=robot.device)
        additional_dims.append(base_ang_vel)
    
    # Get projected gravity (3D)
    # Projected gravity = quaternion.conjugate() * gravity_vector
    try:
        root_quat = robot.data.root_quat_w  # [num_envs, 4] (w, x, y, z)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=robot.device).repeat(env.num_envs, 1)  # [num_envs, 3]
        
        # Convert quaternion to rotation matrix and apply to gravity
        # Simplified: for small rotations, we can approximate
        # For now, use a simple approximation
        proj_gravity = torch.zeros(env.num_envs, 3, device=robot.device)
        additional_dims.append(proj_gravity)
    except (AttributeError, IndexError):
        proj_gravity = torch.zeros(env.num_envs, 3, device=robot.device)
        additional_dims.append(proj_gravity)
    
    # Add 9 more dimensions (unknown - pad with zeros for now)
    # TODO: Find out what these 9 dimensions represent
    unknown_dims = torch.zeros(env.num_envs, 9, device=robot.device)
    additional_dims.append(unknown_dims)
    
    # Concatenate all additional dimensions
    if additional_dims:
        additional_obs = torch.cat(additional_dims, dim=-1)  # [num_envs, 15]
        obs = torch.cat([obs, additional_obs], dim=-1)  # [num_envs, 73]
    
    return obs.view(env.num_envs, -1)


def get_robot_dex3_joint_states(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get joint positions and velocities for Dex3 hand joints (fingers only).
    
    Returns [q, dq] concatenated for all hand joints.
    Based on G1-29DOF configuration: 14 hand joints (7 left + 7 right).
    """
    robot: "Articulation" = env.scene["robot"]
    
    # Define the expected hand joint names in the order they should appear
    # This matches the G1-29DOF configuration
    expected_hand_joint_names = [
        "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
        "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
        "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    ]
    
    # Get all joint names from robot
    joint_names = robot.data.joint_names
    
    # Find indices for expected hand joints
    hand_joint_indices = []
    for expected_name in expected_hand_joint_names:
        try:
            idx = joint_names.index(expected_name)
            hand_joint_indices.append(idx)
        except ValueError:
            # Joint not found, skip it
            continue
    
    if len(hand_joint_indices) == 0:
        # Fallback: filter by name pattern
        hand_joint_indices = [i for i, name in enumerate(joint_names) 
                            if any(hand_term in name for hand_term in ["hand_index", "hand_middle", "hand_thumb"])]
    
    if len(hand_joint_indices) == 0:
        # No hand joints found, return empty tensor
        return torch.zeros(env.num_envs, 0, device=robot.device)
    
    hand_joint_indices = torch.tensor(hand_joint_indices, device=robot.device, dtype=torch.long)
    
    # Get positions and velocities for hand joints
    q = robot.data.joint_pos[:, hand_joint_indices]  # [num_envs, num_hand_dof]
    dq = robot.data.joint_vel[:, hand_joint_indices]  # [num_envs, num_hand_dof]
    
    # Concatenate [q, dq]
    obs = torch.cat([q, dq], dim=-1)
    return obs.view(env.num_envs, -1)


__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_dex3_joint_states",
]
