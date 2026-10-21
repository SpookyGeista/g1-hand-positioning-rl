# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Reward functions for G1 Throwing Test Task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject


# Global buffer for tracking if ball is released
_ball_released = None


def _get_ball_released(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get ball release status from events module."""
    global _ball_released
    # Import here to avoid circular dependency
    from .events import _ball_released as ball_released_global
    if ball_released_global is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    return ball_released_global


def reward_ball_height(
    env: "ManagerBasedRLEnv",
    target_height: float = 1.8,
    sigma: float = 0.3,
    scale: float = 1.0,
    only_after_release: bool = True,
) -> torch.Tensor:
    """Reward for ball reaching target height."""
    ball: "RigidObject" = env.scene["ball"]
    ball_pos = ball.data.root_pos_w[:, :3]  # [num_envs, 3]
    ball_height = ball_pos[:, 2]
    
    if only_after_release:
        released = _get_ball_released(env)
        mask = released
        if not mask.any():
            return torch.zeros(env.scene.num_envs, device=env.device)
    else:
        mask = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    
    height_diff = ball_height - target_height
    reward = scale * torch.exp(-0.5 * (height_diff / sigma) ** 2)
    reward[~mask] = 0.0
    return reward


def reward_ball_xy_gauss_base(
    env: "ManagerBasedRLEnv",
    target_fwd: float = -0.35,
    target_lat: float = 0.16,
    sigma_fwd: float = 0.25,
    sigma_lat: float = 0.25,
    scale: float = 1.0,
    only_after_release: bool = True,
) -> torch.Tensor:
    """Reward for ball landing at target XY position."""
    ball: "RigidObject" = env.scene["ball"]
    ball_pos = ball.data.root_pos_w[:, :3]  # [num_envs, 3]
    
    if only_after_release:
        released = _get_ball_released(env)
        mask = released
        if not mask.any():
            return torch.zeros(env.scene.num_envs, device=env.device)
    else:
        mask = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    
    fwd_diff = ball_pos[:, 0] - target_fwd
    lat_diff = ball_pos[:, 1] - target_lat
    
    reward = scale * torch.exp(
        -0.5 * ((fwd_diff / sigma_fwd) ** 2 + (lat_diff / sigma_lat) ** 2)
    )
    reward[~mask] = 0.0
    return reward


def reward_ball_xy_velocity_penalty(
    env: "ManagerBasedRLEnv",
    scale: float = 1.0,
    only_after_release: bool = True,
) -> torch.Tensor:
    """Penalty for ball horizontal velocity (encourages vertical throw)."""
    ball: "RigidObject" = env.scene["ball"]
    ball_vel = ball.data.root_lin_vel_w  # [num_envs, 3]
    
    if only_after_release:
        released = _get_ball_released(env)
        mask = released
        if not mask.any():
            return torch.zeros(env.scene.num_envs, device=env.device)
    else:
        mask = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    
    xy_vel = torch.norm(ball_vel[:, :2], dim=-1)
    reward = -scale * xy_vel
    reward[~mask] = 0.0
    return reward


def reward_ball_release_timing(
    env: "ManagerBasedRLEnv",
    target_velocity: float = 2.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for releasing ball at target velocity."""
    # This would need hand velocity tracking - simplified for now
    return torch.zeros(env.scene.num_envs, device=env.device)


def reward_hand_closed(
    env: "ManagerBasedRLEnv",
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for keeping hand closed (holding ball)."""
    # Simplified - would need hand joint positions
    return torch.zeros(env.scene.num_envs, device=env.device)


def reward_ball_upward_velocity(
    env: "ManagerBasedRLEnv",
    target_vz: float = 4.4,
    sigma_vz: float = 1.0,
    scale: float = 1.0,
    only_after_release: bool = True,
) -> torch.Tensor:
    """Reward for ball upward velocity matching target."""
    ball: "RigidObject" = env.scene["ball"]
    ball_vel = ball.data.root_lin_vel_w  # [num_envs, 3]
    vz = ball_vel[:, 2]
    
    if only_after_release:
        released = _get_ball_released(env)
        mask = released
        if not mask.any():
            return torch.zeros(env.scene.num_envs, device=env.device)
    else:
        mask = torch.ones(env.scene.num_envs, dtype=torch.bool, device=env.device)
    
    vz_diff = vz - target_vz
    reward = scale * torch.exp(-0.5 * (vz_diff / sigma_vz) ** 2)
    reward[~mask] = 0.0
    return reward


def reward_hand_vz_pre_release(
    env: "ManagerBasedRLEnv",
    v_cap: float = 2.5,
    alpha: float = 0.2,
    z_band_lo: float = 0.65,
    z_release: float = 0.8,
    v_target: float = 3.43,
    sigma_v: float = 0.35,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward for hand upward velocity before release."""
    # Simplified - would need hand velocity tracking
    return torch.zeros(env.scene.num_envs, device=env.device)


def reward_hand_vz_pre_release_improved(
    env: "ManagerBasedRLEnv",
    v_cap: float = 2.5,
    alpha: float = 0.2,
    z_band_lo: float = 0.65,
    z_release: float = 0.7,
    v_target: float = 3.43,
    sigma_v: float = 0.35,
    scale: float = 1.0,
    ball_dist_threshold: float = 0.15,
) -> torch.Tensor:
    """Improved reward for hand upward velocity before release."""
    # Simplified - would need hand velocity tracking
    return torch.zeros(env.scene.num_envs, device=env.device)


def reward_hand_az_pre_release(
    env: "ManagerBasedRLEnv",
    scale: float = 1.0,
    use_palm_link: bool = True,
) -> torch.Tensor:
    """Reward for hand acceleration before release."""
    # Simplified - would need hand acceleration tracking
    return torch.zeros(env.scene.num_envs, device=env.device)


__all__ = [
    "reward_ball_height",
    "reward_ball_xy_gauss_base",
    "reward_ball_xy_velocity_penalty",
    "reward_ball_release_timing",
    "reward_hand_closed",
    "reward_ball_upward_velocity",
    "reward_hand_vz_pre_release",
    "reward_hand_vz_pre_release_improved",
    "reward_hand_az_pre_release",
]








