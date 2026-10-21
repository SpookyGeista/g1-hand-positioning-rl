# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Einfache Wurf-Szene für den G1-Roboter.

Diese Version ist von den alten ``common_config``-Hilfsmodulen entkoppelt und
verwendet direkt die im aktuellen Repository verfügbare G1-Konfiguration.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG


@configclass
class SimpleThrowSceneCfg(InteractiveSceneCfg):
    """Simple Throwing Scene Configuration - Unitree Style."""

    # einfacher Boden
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Umgebungslicht
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    # G1-Roboter (29 DOF, mit Händen), Basis fixiert
    # Wir überschreiben die init_state, um right_wrist_yaw_joint auf 90 Grad zu setzen
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ROBOT_CFG.init_state.replace(
            joint_pos={
                **ROBOT_CFG.init_state.joint_pos,
                "right_wrist_yaw_joint": math.pi / 2,  # 90 Grad für trainierte Policy
            }
        ),
    )
    robot.spawn.fix_base = True

    # Ball zum Werfen
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.8, 0.0),  # gelb
                roughness=0.8,
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.6,
                friction_combine_mode="multiply",
                restitution_combine_mode="average",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.0005, 0.23872, 0.84455),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
