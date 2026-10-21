from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.hand_positioning.mdp.commands import HandPositionCommand


def hand_position_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the position error between target and current hand position (relative to base)."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    error = command.position_error.view(env.num_envs, -1)
    # Replace NaN/inf with zeros to prevent propagation
    return torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)


def hand_target_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the target hand position (relative to base)."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    target = command.target_pos_rel.view(env.num_envs, -1)
    # Replace NaN/inf with zeros to prevent propagation
    return torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


def hand_current_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current hand position (relative to base)."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    current = command.current_hand_pos_rel.view(env.num_envs, -1)
    # Replace NaN/inf with zeros to prevent propagation
    return torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)


def body_pos_rel_to_root(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, body_name: str
) -> torch.Tensor:
    """Get the position of a body relative to the root/base frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        body_name: Name of the body/link to get position for (regex pattern supported).
    
    Returns:
        Position of the body relative to root in base frame [num_envs, 3].
        This is calculated as: body_pos_world - root_pos_world
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Find body index
    body_indices = asset.find_bodies(body_name)[0]
    if len(body_indices) == 0:
        raise ValueError(f"Body '{body_name}' not found in {asset_cfg.name}")
    body_idx = body_indices[0]
    
    # Get body position in world frame
    body_pos_w = asset.data.body_pos_w[:, body_idx, :3]
    
    # Get root position in world frame (first 3 elements are x, y, z)
    root_pos_w = asset.data.root_pos_w[:, :3]
    
    # Calculate relative position: body_pos_world - root_pos_world
    pos_rel = body_pos_w - root_pos_w
    
    # Replace NaN/inf with zeros
    return torch.nan_to_num(pos_rel, nan=0.0, posinf=0.0, neginf=0.0)

