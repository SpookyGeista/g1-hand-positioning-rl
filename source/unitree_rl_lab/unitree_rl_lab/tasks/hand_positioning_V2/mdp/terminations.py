from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from unitree_rl_lab.tasks.hand_positioning_V2.mdp.commands import HandPositionCommand


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when episode length exceeds maximum."""
    return env.episode_length_buf >= env.max_episode_length


def hand_position_success(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.03
) -> torch.Tensor:
    """Terminate (success) when hand is within threshold of target position."""
    command: HandPositionCommand = env.command_manager.get_term(command_name)
    error_norm = torch.norm(command.position_error, dim=-1)
    
    # Check if within threshold
    within_threshold = error_norm < threshold
    return within_threshold


