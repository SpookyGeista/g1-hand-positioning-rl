# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Events for G1 Throwing Test Task"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation, RigidObject


# Global buffers for ball release tracking
_ball_released = None
_steps_since_reset = None


def reset_ball_and_robot(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None):
    """Reset ball and robot positions for new episode."""
    global _ball_released, _steps_since_reset
    
    robot: "Articulation" = env.scene["robot"]
    ball: "RigidObject" = env.scene["ball"]
    device = env.device
    
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    
    # Set right_wrist_yaw_joint to 90 degrees (π/2) for trained policy BEFORE reset
    # This ensures the position is set correctly during reset
    import math
    target_pos = math.pi / 2  # 90 Grad
    
    try:
        # Try to get joint index using get_dof_index
        right_wrist_yaw_idx = robot.get_dof_index("right_wrist_yaw_joint")
    except (AttributeError, KeyError, TypeError):
        # Fallback: search in joint_names
        try:
            joint_names = robot.data.joint_names
            if "right_wrist_yaw_joint" in joint_names:
                right_wrist_yaw_idx = joint_names.index("right_wrist_yaw_joint")
            else:
                right_wrist_yaw_idx = None
        except (AttributeError, TypeError):
            right_wrist_yaw_idx = None
    
    # Update default_joint_pos BEFORE reset so it's used during reset
    if right_wrist_yaw_idx is not None:
        robot.data.default_joint_pos[env_ids, right_wrist_yaw_idx] = target_pos
    
    # Reset robot to default joint positions (from config)
    # Robot base is fixed, so we only reset joints
    # This will use the updated default_joint_pos
    robot.reset(env_ids)
    
    # CRITICAL: Force set the position AFTER reset using write_joint_state_to_sim
    # This directly writes to the simulation, bypassing any config issues
    if right_wrist_yaw_idx is not None:
        # Get current joint positions and velocities for all joints
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = robot.data.joint_vel.clone()
        
        # Set the target position for right_wrist_yaw_joint
        joint_pos[env_ids, right_wrist_yaw_idx] = target_pos
        joint_vel[env_ids, right_wrist_yaw_idx] = 0.0  # Also set velocity to 0
        
        # Update data structures
        robot.data.joint_pos[env_ids, right_wrist_yaw_idx] = target_pos
        robot.data.default_joint_pos[env_ids, right_wrist_yaw_idx] = target_pos
        
        # Write directly to simulation - this is the key!
        robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        
        # Also update action offset (important for PD control)
        try:
            action_term = env.action_manager.get_term("joint_pos")
            if hasattr(action_term, "_offset") and hasattr(action_term, "joint_names"):
                action_joint_names = action_term.joint_names
                if "right_wrist_yaw_joint" in action_joint_names:
                    action_idx = action_joint_names.index("right_wrist_yaw_joint")
                    action_term._offset[env_ids, action_idx] = target_pos
        except (AttributeError, KeyError, ValueError):
            pass
    
    # Reset ball position (in right hand area)
    num_envs_to_reset = len(env_ids) if env_ids is not None else env.scene.num_envs
    ball_pos = torch.zeros(num_envs_to_reset, 3, device=device)
    ball_pos[:, 0] = -0.0005  # x
    ball_pos[:, 1] = 0.23872  # y
    ball_pos[:, 2] = 0.84455  # z
    
    ball_rot = torch.zeros(num_envs_to_reset, 4, device=device)
    ball_rot[:, 0] = 1.0  # w
    
    ball_vel = torch.zeros(num_envs_to_reset, 6, device=device)
    
    root_state = torch.cat([ball_pos, ball_rot, ball_vel], dim=-1)
    ball.write_root_state_to_sim(root_state, env_ids)
    
    # Initialize release buffers
    if _ball_released is None:
        _ball_released = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=device)
    if _steps_since_reset is None:
        _steps_since_reset = torch.zeros(env.scene.num_envs, dtype=torch.long, device=device)
    
    _ball_released[env_ids] = False
    _steps_since_reset[env_ids] = 0


def release_reset_buffers(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None):
    """Initialize/reset ball release tracking buffers."""
    global _ball_released, _steps_since_reset
    
    device = env.device
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    
    if _ball_released is None:
        _ball_released = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=device)
    if _steps_since_reset is None:
        _steps_since_reset = torch.zeros(env.scene.num_envs, dtype=torch.long, device=device)
    
    _ball_released[env_ids] = False
    _steps_since_reset[env_ids] = 0


def step_release_on_hand_height(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None = None,
    z_thresh: float = 0.65,
    v_min: float = 3.0,
    min_steps_since_reset: int = 1,
    max_hand_ball_dist: float = 0.15,
    **kwargs,  # Accept additional kwargs to be compatible with Isaac Lab's parameter passing
):
    """Release ball when hand reaches threshold height with sufficient velocity."""
    global _ball_released, _steps_since_reset
    
    robot: "Articulation" = env.scene["robot"]
    ball: "RigidObject" = env.scene["ball"]
    device = env.device
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    
    if _ball_released is None:
        _ball_released = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=device)
    if _steps_since_reset is None:
        _steps_since_reset = torch.zeros(env.scene.num_envs, dtype=torch.long, device=device)
    
    # Increment step counter
    _steps_since_reset[env_ids] += 1
    
    # Skip if already released or not enough steps
    mask = (~_ball_released[env_ids]) & (_steps_since_reset[env_ids] >= min_steps_since_reset)
    if not mask.any():
        return
    
    active_env_ids = env_ids[mask] if env_ids is not None else torch.arange(env.scene.num_envs, device=env.device)[mask]
    
    # Get right hand palm position (or wrist position as proxy)
    try:
        hand_body_indices = robot.find_bodies("right_wrist_yaw_link")[0]
        if len(hand_body_indices) == 0:
            # Fallback to right_wrist_roll_link
            hand_body_indices = robot.find_bodies("right_wrist_roll_link")[0]
    except Exception:
        return
    
    if len(hand_body_indices) == 0:
        return
    
    hand_pos = robot.data.body_pos_w[active_env_ids, hand_body_indices[0], :3]  # [num_active, 3]
    hand_vel = robot.data.body_lin_vel_w[active_env_ids, hand_body_indices[0], :3]  # [num_active, 3]
    
    # Get ball position
    ball_pos = ball.data.root_pos_w[active_env_ids, :3]  # [num_active, 3]
    
    # Check conditions
    hand_height = hand_pos[:, 2]  # z-coordinate
    hand_vz = hand_vel[:, 2]  # z-velocity
    hand_ball_dist = torch.norm(hand_pos - ball_pos, dim=-1)
    
    # Release condition: hand above threshold, sufficient upward velocity, ball close to hand
    release_mask = (
        (hand_height > z_thresh) &
        (hand_vz > v_min) &
        (hand_ball_dist < max_hand_ball_dist)
    )
    
    if release_mask.any():
        release_env_ids = active_env_ids[release_mask]
        _ball_released[release_env_ids] = True
        
        # Optionally disable ball-robot collision or apply release force
        # For now, we just mark it as released


__all__ = [
    "reset_ball_and_robot",
    "release_reset_buffers",
    "step_release_on_hand_height",
]
