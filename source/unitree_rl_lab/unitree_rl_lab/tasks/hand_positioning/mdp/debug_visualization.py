"""Debug visualization utilities for hand positioning task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LinkPositionVisualizer:
    """Visualize a link's position relative to root frame in Isaac Sim."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        body_name: str,
        prim_path: str = "/Visuals/Debug/link_pos_rel_root",
    ):
        """Initialize the visualizer.

        Args:
            env: The environment instance.
            asset_cfg: Configuration for the robot asset.
            body_name: Name of the body/link to visualize (regex pattern supported).
            prim_path: Prim path for the visualizer in Isaac Sim.
        """
        self.env = env
        self.asset: Articulation = env.scene[asset_cfg.name]
        
        # Find body index
        body_indices = self.asset.find_bodies(body_name)[0]
        if len(body_indices) == 0:
            raise ValueError(f"Body '{body_name}' not found in {asset_cfg.name}")
        self.body_idx = body_indices[0]
        
        # Create visualizer
        self.visualizer = VisualizationMarkers(
            FRAME_MARKER_CFG.replace(prim_path=prim_path)
        )
        self.visualizer.set_visibility(True)

    def visualize(self):
        """Update the visualization with current link position relative to root."""
        if not self.asset.is_initialized:
            return
        
        # Get body position in world frame
        body_pos_w = self.asset.data.body_pos_w[:, self.body_idx, :3]
        
        # Get root position in world frame
        root_pos_w = self.asset.data.root_pos_w[:, :3]
        
        # Calculate relative position
        pos_rel_w = body_pos_w - root_pos_w
        
        # Visualize at root position with offset showing relative position
        # This shows a marker at: root_pos + relative_pos = body_pos
        # So you can see both the root and the relative offset
        visualize_pos = root_pos_w + pos_rel_w
        
        # Use identity quaternion for visualization
        quat = torch.zeros(self.env.num_envs, 4, device=self.env.device)
        quat[:, 0] = 1.0
        
        # Replace NaN/inf
        visualize_pos = torch.nan_to_num(visualize_pos, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.visualizer.visualize(visualize_pos, quat)
    
    def get_position_rel_root(self) -> torch.Tensor:
        """Get the current position relative to root (without visualization).
        
        Returns:
            Position relative to root [num_envs, 3]
        """
        if not self.asset.is_initialized:
            return torch.zeros(self.env.num_envs, 3, device=self.env.device)
        
        body_pos_w = self.asset.data.body_pos_w[:, self.body_idx, :3]
        root_pos_w = self.asset.data.root_pos_w[:, :3]
        pos_rel = body_pos_w - root_pos_w
        return torch.nan_to_num(pos_rel, nan=0.0, posinf=0.0, neginf=0.0)













