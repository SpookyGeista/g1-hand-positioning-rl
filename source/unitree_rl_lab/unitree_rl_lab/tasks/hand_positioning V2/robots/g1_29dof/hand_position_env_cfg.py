import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.hand_positioning import mdp

##
# Scene definition
##


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    # Fix base to prevent robot from falling
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.fix_base = True

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    hand_position = mdp.HandPositionCommandCfg(
        asset_name="robot",
        hand_body_name="left_wrist_yaw_link",  # left_hand_palm_link is merged into left_wrist_yaw_link by Isaac Sim
        position_range={
            "x": (0.2, 0.5),  # forward from base
            "y": (0.05, 0.3),  # left/right
            "z": (0.095, 0.3),  # up/down
        },
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Only control LEFT arm joints for hand positioning
    # Right arm and legs will maintain default position through high stiffness/damping
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_.*_joint",  # left shoulder joints only
            "left_elbow_joint",  # left elbow joint only
            "left_wrist_.*",  # left wrist joints only (wrist_roll, wrist_pitch, wrist_yaw)
        ],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Nur das Ziel der Handposition als Input (kein expliziter Fehler / Ist-Position),
        # damit das später im C++-Deploy leicht gesetzt werden kann.
        hand_target_position = ObsTerm(
            func=mdp.hand_target_position,
            params={"command_name": "hand_position"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_.*_joint",  # left arm only
                        "left_elbow_joint",
                        "left_wrist_.*",
                    ],
                )
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_.*_joint",  # left arm only
                        "left_elbow_joint",
                        "left_wrist_.*",
                    ],
                )
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        hand_position_error = ObsTerm(func=mdp.hand_position_error, params={"command_name": "hand_position"})
        hand_target_position = ObsTerm(func=mdp.hand_target_position, params={"command_name": "hand_position"})
        hand_current_position = ObsTerm(func=mdp.hand_current_position, params={"command_name": "hand_position"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_.*_joint",  # left arm only
                        "left_elbow_joint",
                        "left_wrist_.*",
                    ],
                )
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_.*_joint",  # left arm only
                        "left_elbow_joint",
                        "left_wrist_.*",
                    ],
                )
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_hand_position = RewTerm(
        func=mdp.track_hand_position_exp,
        weight=2.0,
        params={"command_name": "hand_position", "std": math.sqrt(0.25)},  # Same as velocity task vorhrt sqrt 0.1
    )

    hand_position_hold = RewTerm(
        func=mdp.hand_position_hold_reward,
        weight=1.0,
        params={"command_name": "hand_position", "threshold": 0.05, "std": 0.05},  # std Increased from 0.02
    )

    hand_palm_horizontal = RewTerm(
        func=mdp.hand_palm_horizontal_reward,
        weight=1.5,
        params={"command_name": "hand_position", "std": math.sqrt(0.25)},  # Increased from 0.1 to match velocity task
    )

    # alive reward removed - not useful when episode always runs for fixed time

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_.*_joint",  # left arm only
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            )
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-6,  # 10x stärker: von -2.5e-7 auf -2.5e-6
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_.*_joint",  # left arm only
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            )
        },
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,  # Penalty for high joint torques
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_.*_joint",  # left arm only
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            )
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_.*_joint",  # left arm only
                    "left_elbow_joint",
                    "left_wrist_.*_joint",  # Fixed: should end with _joint
                ],
            )
        },
    )
    energy = RewTerm(
        func=mdp.energy,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_.*_joint",  # left arm only
                    "left_elbow_joint",
                    "left_wrist_.*",
                ],
            )
        },
    )

    # -- robot
    # flat_orientation_l2 and base_height removed - robot stands stiffly, won't tip over or fall


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height removed - robot stands stiffly, won't fall
    # bad_orientation removed - robot stands stiffly, won't tip over


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the hand positioning environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs= 4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32

