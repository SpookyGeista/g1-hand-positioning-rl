from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg
else:
    from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.hand_positioning_V2.mdp.commands import HandPositionCommand
try:
    from isaaclab.utils.math import quat_apply
except ImportError:
    from isaaclab.utils.math import quat_rotate as quat_apply


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def track_hand_position_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.1
) -> torch.Tensor:
    """Reward for tracking hand position using exponential kernel."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.position_error), dim=-1)
    return torch.exp(-error / (std**2))


def track_hand_position_l2(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Penalty for hand position error using L2 norm."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.position_error, dim=-1)
    return -error


def hand_position_hold_reward(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.05, std: float = 0.02
) -> torch.Tensor:
    """Reward for holding hand position within threshold."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    error_norm = torch.norm(command.position_error, dim=-1)
    # Exponential reward when within threshold
    reward = torch.exp(-error_norm / std)
    # Bonus for being within threshold
    within_threshold = (error_norm < threshold).float()
    return reward * (1.0 + 2.0 * within_threshold)


def hand_palm_horizontal_reward(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.1
) -> torch.Tensor:
    """Reward for keeping hand palm horizontal (parallel to ground)."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    
    # Get hand body quaternion in world frame (shape: [num_envs, 4])
    hand_quat_w = robot.data.body_quat_w[:, command.hand_body_index]
    
    # Normalize quaternion to prevent numerical issues
    quat_norm = torch.norm(hand_quat_w, dim=-1, keepdim=True)
    # Avoid division by zero - if norm is too small, use identity quaternion
    quat_norm = torch.clamp(quat_norm, min=1e-6)
    hand_quat_w = hand_quat_w / quat_norm
    
    # Replace any NaN or inf values with identity quaternion [1, 0, 0, 0]
    is_invalid = torch.any(torch.isnan(hand_quat_w) | torch.isinf(hand_quat_w), dim=-1)
    identity_quat = torch.zeros_like(hand_quat_w)
    identity_quat[:, 0] = 1.0
    hand_quat_w = torch.where(is_invalid.unsqueeze(-1), identity_quat, hand_quat_w)
    
    # Define local axis that corresponds to hand palm normal.
    # For the current G1 hand asset, the palm-Normale entspricht näherungsweise
    # der +X-Achse des Hand-Frames, nicht der +Z-Achse.
    hand_z_local = torch.tensor([1.0, 0.0, 0.0], device=env.device)
    
    # Rotate z-axis from hand frame to world frame
    # quat_apply rotates a vector by a quaternion
    hand_z_world = quat_apply(hand_quat_w, hand_z_local.unsqueeze(0).repeat(env.num_envs, 1))
    
    # Replace NaN/inf in rotated vector
    hand_z_world = torch.nan_to_num(hand_z_world, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize the rotated vector to unit length
    z_norm = torch.norm(hand_z_world, dim=-1, keepdim=True)
    z_norm = torch.clamp(z_norm, min=1e-6)
    hand_z_world = hand_z_world / z_norm
    
    # World up vector
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    
    # Dot product: should be close to 1.0 when hand palm is horizontal (z-axis points up)
    alignment = torch.sum(hand_z_world * world_up, dim=-1)
    
    # Clamp alignment to valid range [-1, 1] to avoid numerical issues
    alignment = torch.clamp(alignment, -1.0, 1.0)
    
    # Error from perfect horizontal (alignment should be 1.0)
    error = 1.0 - alignment
    
    # Clamp error to valid range [0, 2]
    error = torch.clamp(error, 0.0, 2.0)
    
    # Exponential reward: closer to horizontal = higher reward
    reward = torch.exp(-error / (std**2))
    
    # Replace any remaining NaN/inf with safe default (0.0 for reward)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Final clamp to ensure reward is in [0, 1]
    return torch.clamp(reward, 0.0, 1.0)


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize high joint torques using L2 norm (sum of squares).
    
    This reward term penalizes high joint torques to encourage smoother,
    more energy-efficient motions. The penalty is the sum of squared torques
    for all specified joints.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying which joints to penalize.
        
    Returns:
        Penalty value (higher = more penalty for high torques).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get torques for specified joints
    # Try applied_torque first, fallback to joint_torques if available
    if hasattr(asset.data, 'applied_torque') and asset.data.applied_torque is not None:
        torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, 'joint_torques') and asset.data.joint_torques is not None:
        torques = asset.data.joint_torques[:, asset_cfg.joint_ids]
    else:
        # If no torque data available, return zero penalty
        return torch.zeros(env.num_envs, device=env.device)
    
    # Return L2 norm (sum of squares) of torques
    # This penalizes high torques quadratically
    return torch.sum(torch.square(torques), dim=-1)

