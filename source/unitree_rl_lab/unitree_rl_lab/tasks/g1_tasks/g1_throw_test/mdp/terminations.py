# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Termination functions for G1 Throwing Test Task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject


def step_limit(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate when episode length exceeds maximum."""
    return env.episode_length_buf >= env.max_episode_length


def ball_at_rest_on_ground(env: "ManagerBasedRLEnv", ground_z: float = 0.0, velocity_threshold: float = 0.1) -> torch.Tensor:
    """Terminate when ball is at rest on the ground."""
    ball: "RigidObject" = env.scene["ball"]
    
    # Get ball position and velocity
    ball_pos = ball.data.root_pos_w[:, :3]  # [num_envs, 3]
    ball_vel = ball.data.root_lin_vel_w  # [num_envs, 3]
    
    # Check if ball is on ground and velocity is low
    on_ground = ball_pos[:, 2] <= ground_z + 0.05  # within 5cm of ground
    at_rest = torch.norm(ball_vel, dim=-1) < velocity_threshold
    
    return on_ground & at_rest


def ball_ground_contact(env: "ManagerBasedRLEnv", ground_z: float = 0.0) -> torch.Tensor:
    """Terminate when ball contacts the ground."""
    ball: "RigidObject" = env.scene["ball"]
    
    # Get ball position
    ball_pos = ball.data.root_pos_w[:, :3]  # [num_envs, 3]
    
    # Check if ball is at or below ground level
    return ball_pos[:, 2] <= ground_z + 0.03  # ball radius is 0.03


__all__ = [
    "step_limit",
    "ball_at_rest_on_ground",
    "ball_ground_contact",
]
