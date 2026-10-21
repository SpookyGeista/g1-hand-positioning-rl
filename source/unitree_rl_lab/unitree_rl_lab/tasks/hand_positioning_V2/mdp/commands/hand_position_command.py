from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HandPositionCommand(CommandTerm):
    """Command for generating target hand positions."""

    cfg: HandPositionCommandCfg

    def __init__(self, cfg: HandPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        # Find the hand body index
        hand_body_indices = self.robot.find_bodies(cfg.hand_body_name)[0]
        if len(hand_body_indices) == 0:
            raise ValueError(f"Hand body '{cfg.hand_body_name}' not found in robot")
        self.hand_body_index = hand_body_indices[0]

        # Initialize target position relative to robot base
        self.target_pos_rel = torch.zeros(self.num_envs, 3, device=self.device)
        # Store base position for relative calculations
        self.base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Track time for resampling
        self.time_since_resample = torch.zeros(self.num_envs, device=self.device)

        # Visualizer for target position
        if cfg.debug_vis:
            self.visualizer = VisualizationMarkers(cfg.visualizer_cfg.replace(prim_path="/Visuals/Command/hand_target"))

        # Initialize metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample target hand positions for specified environments."""
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Sample target position relative to base
        ranges = torch.tensor(
            [
                self.cfg.position_range["x"],
                self.cfg.position_range["y"],
                self.cfg.position_range["z"],
            ],
            device=self.device,
        )
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device
        )
        self.target_pos_rel[env_ids_tensor] = rand_samples
        self.time_since_resample[env_ids_tensor] = 0.0

    def _update_command(self):
        """Update command state."""
        # Update base position for relative calculations
        base_pos = self.robot.data.root_pos_w[:, :3]
        # Replace NaN/inf with zeros to prevent propagation
        self.base_pos_w = torch.nan_to_num(base_pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update time since resample
        self.time_since_resample += self._env.step_dt
        
        # Check if we need to resample
        resample_time = sample_uniform(
            torch.tensor(self.cfg.resampling_time_range[0], device=self.device),
            torch.tensor(self.cfg.resampling_time_range[1], device=self.device),
            (self.num_envs,),
            device=self.device,
        )
        env_ids_to_resample = torch.where(self.time_since_resample >= resample_time)[0].cpu().tolist()
        if len(env_ids_to_resample) > 0:
            self._resample_command(env_ids_to_resample)

    def reset(self, env_ids: Sequence[int]):
        """Reset command for specified environments."""
        self._resample_command(env_ids)
        # Update metrics and return them
        self._update_metrics()
        # Return metrics dictionary
        return self.metrics

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Get target position in world frame."""
        return self.base_pos_w + self.target_pos_rel

    @property
    def current_hand_pos_w(self) -> torch.Tensor:
        """Get current hand position in world frame."""
        pos = self.robot.data.body_pos_w[:, self.hand_body_index, :3]
        # Replace NaN/inf with zeros to prevent propagation
        return torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def current_hand_pos_rel(self) -> torch.Tensor:
        """Get current hand position relative to base."""
        rel_pos = self.current_hand_pos_w - self.base_pos_w
        # Replace NaN/inf with zeros to prevent propagation
        return torch.nan_to_num(rel_pos, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def position_error(self) -> torch.Tensor:
        """Get position error (target - current) in relative frame."""
        error = self.target_pos_rel - self.current_hand_pos_rel
        # Replace NaN/inf with zeros to prevent propagation
        return torch.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def command(self) -> torch.Tensor:
        """Get the command tensor (target position relative to base)."""
        return self.target_pos_rel

    def _update_metrics(self):
        """Update command metrics."""
        error_norm = torch.norm(self.position_error, dim=-1)
        # Replace NaN/inf with zeros to prevent propagation
        self.metrics["position_error"] = torch.nan_to_num(error_norm, nan=0.0, posinf=0.0, neginf=0.0)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        if debug_vis and self.cfg.debug_vis:
            if hasattr(self, 'visualizer'):
                self.visualizer.set_visibility(True)
        else:
            if hasattr(self, 'visualizer'):
                self.visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if not hasattr(self, 'robot') or not self.robot.is_initialized:
            return
        
        if hasattr(self, 'visualizer'):
            # Update base position first
            self.base_pos_w = self.robot.data.root_pos_w[:, :3]
            # Visualize target position
            target_pos_world = self.target_pos_w
            # Use identity quaternion for visualization
            target_quat = torch.zeros(self.num_envs, 4, device=self.device)
            target_quat[:, 0] = 1.0
            self.visualizer.visualize(target_pos_world, target_quat)


@configclass
class HandPositionCommandCfg(CommandTermCfg):
    """Configuration for hand position command."""

    class_type: type = HandPositionCommand

    asset_name: str = MISSING
    hand_body_name: str = MISSING  # e.g., "left_wrist_yaw_link" or "right_wrist_yaw_link"

    position_range: dict[str, tuple[float, float]] = {
        "x": (0.2, 0.6),  # forward
        "y": (-0.3, 0.3),  # left/right
        "z": (0.3, 0.8),  # up/down
    }

    resampling_time_range: tuple[float, float] = (10.0, 10.0)  # resample every 10 seconds

    debug_vis: bool = True
    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/hand_target")
    visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

