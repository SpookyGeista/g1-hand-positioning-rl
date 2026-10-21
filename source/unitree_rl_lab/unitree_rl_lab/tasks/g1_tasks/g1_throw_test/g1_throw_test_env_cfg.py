# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from . import mdp

from unitree_rl_lab.tasks.common_scene.simple_throw_scene_cfg import SimpleThrowSceneCfg

##
# Scene definition
##

@configclass
class G1ThrowTestSceneCfg(SimpleThrowSceneCfg):
    """Einfache Wurf-Szene: Boden, Licht, Kamera, G1(Dex3)."""
    env_spacing: float = 2.5

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action configuration - only 3 joints for throwing."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
        ],
        clip={
            "right_shoulder_pitch_joint": (-0.6, 0.6),
            "right_elbow_joint": (-1.0, 1.0),
            "right_wrist_yaw_joint": (-0.5, 0.5),
        },
        scale={
            "right_.*": 0.35,
        },
        offset={
            "right_elbow_joint": 2.0944,
            "right_wrist_yaw_joint": math.pi / 2,  # 90 Grad für trainierte Policy
        },
        preserve_order=False,
        use_default_offset=False,
    )

@configclass
class ObservationsCfg:
    """Observations for the throwing policy."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group - body joints + dex3 hand joints."""

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_dex3_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Termination conditions for the throwing task - only timeout after 10 seconds."""
    time_out = DoneTerm(func=mdp.step_limit, time_out=True)
    # Deaktiviert: Nur nach 10 Sekunden terminieren
    # ball_rest = DoneTerm(func=mdp.ball_at_rest_on_ground, time_out=False)
    # ball_ground = DoneTerm(func=mdp.ball_ground_contact, time_out=False)

@configclass
class EventCfg:
    """Events for resetting ball and robot, and releasing ball."""
    reset_root_state_uniform = EventTermCfg(
        func="isaaclab.envs.mdp.events:reset_root_state_uniform",
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    reset_ball_and_robot = EventTermCfg(
        func=mdp.reset_ball_and_robot,
        mode="reset",
    )
    init_release_buffers = EventTermCfg(
        func=mdp.release_reset_buffers,
        mode="reset",
    )
    step_release_on_hand_height = EventTermCfg(
        func=mdp.step_release_on_hand_height,
        mode="interval",
        params={
            "z_thresh": 0.65,
            "v_min": 3.0,
            "min_steps_since_reset": 1,
            "max_hand_ball_dist": 0.15,
        },
        interval_range_s=(0.03, 0.03),
    )

##
# Environment configuration
##

@configclass
class G1ThrowTestEnvCfg(ManagerBasedRLEnvCfg):
    """G1 Throwing Test Environment Configuration"""
    
    # scene configuration
    scene: G1ThrowTestSceneCfg = G1ThrowTestSceneCfg()
    
    # MDP configuration
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rewards = None  # reward manager
    
    # environment settings
    env_spacing: float = 2.5
    episode_length_s: float = 10.0
    num_envs: int = 1  # just one environment for testing
    decimation: int = 4  # control frequency decimation
    
    # physics settings
    physics_dt: float = 0.005
    rendering_dt: float = 0.02
    control_frequency_inv: int = 1
    
    # debug settings
    debug_vis: bool = True
    enable_camera_sensors: bool = True


@configclass
class G1ThrowTestPlayEnvCfg(G1ThrowTestEnvCfg):
    """Play configuration - optimized for single environment playback."""
    
    def __post_init__(self):
        super().__post_init__()
        # Ensure only 1 environment for play
        self.scene.num_envs = 1
        self.num_envs = 1


