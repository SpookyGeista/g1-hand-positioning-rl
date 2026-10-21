# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import csv
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    
    # Get frequency information
    physics_dt = env.unwrapped.cfg.sim.dt
    decimation = env.unwrapped.cfg.decimation
    policy_frequency = 1.0 / (physics_dt * decimation)
    physics_frequency = 1.0 / physics_dt
    
    print(f"[INFO] Physics timestep (dt): {physics_dt} s")
    print(f"[INFO] Decimation: {decimation}")
    print(f"[INFO] Policy step timestep (step_dt): {dt} s")
    print(f"[INFO] Physics frequency: {physics_frequency:.1f} Hz")
    print(f"[INFO] Policy frequency: {policy_frequency:.1f} Hz")
    print(f"[INFO] Policy publishes every {decimation} physics steps")

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    
    # Get robot from environment for torque and jerk measurements
    robot = env.unwrapped.scene["robot"]
    
    # Get indices of controlled joints (the ones the policy controls)
    controlled_joint_names = [
        "right_shoulder_pitch_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
    ]
    
    # Find joint indices
    controlled_joint_indices = []
    joint_names = robot.data.joint_names
    for joint_name in controlled_joint_names:
        try:
            idx = robot.get_dof_index(joint_name)
            controlled_joint_indices.append(idx)
        except (AttributeError, KeyError, ValueError):
            # Fallback: search in joint_names
            if joint_name in joint_names:
                idx = joint_names.index(joint_name)
                controlled_joint_indices.append(idx)
            else:
                print(f"[WARNING] Joint {joint_name} not found in robot joints!")
    
    controlled_joint_indices = torch.tensor(controlled_joint_indices, device=robot.device, dtype=torch.long)
    print(f"[INFO] Controlled joints: {controlled_joint_names}")
    print(f"[INFO] Controlled joint indices: {controlled_joint_indices.cpu().tolist()}")
    
    # Setup CSV file for logging
    csv_log_dir = os.path.join(log_dir, "play_logs")
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_filename = os.path.join(csv_log_dir, f"policy_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Open CSV file for writing
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = None  # Will be initialized after first data row
    
    print(f"[INFO] Logging data to CSV: {csv_filename}")
    
    # Initialize previous values for jerk calculation (only for controlled joints)
    prev_joint_vel = None
    prev_joint_acc = None
    
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # Convert actions to numpy for printing
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = np.array(actions)
            
            # env stepping (this updates robot data)
            obs, _, _, _ = env.step(actions)
            
            # Get robot data AFTER step (so data is current)
            # Filter to only controlled joints
            # Torques (Tau) - only for controlled joints
            if hasattr(robot.data, 'applied_torque'):
                all_torques = robot.data.applied_torque
                torques = all_torques[:, controlled_joint_indices] if all_torques is not None else None
            elif hasattr(robot.data, 'joint_torques'):
                all_torques = robot.data.joint_torques
                torques = all_torques[:, controlled_joint_indices] if all_torques is not None else None
            else:
                torques = None
            
            # Current joint velocities and positions - only for controlled joints
            all_joint_vel = robot.data.joint_vel
            all_joint_pos = robot.data.joint_pos
            joint_vel = all_joint_vel[:, controlled_joint_indices]
            joint_pos = all_joint_pos[:, controlled_joint_indices]
            
            # Calculate acceleration (change in velocity) - only for controlled joints
            if prev_joint_vel is not None:
                joint_acc = (joint_vel - prev_joint_vel) / dt
            else:
                joint_acc = torch.zeros_like(joint_vel)
            
            # Calculate jerk (change in acceleration) - only for controlled joints
            if prev_joint_acc is not None:
                jerk = (joint_acc - prev_joint_acc) / dt
            else:
                jerk = torch.zeros_like(joint_acc)
            
            # Update previous values for next iteration
            prev_joint_vel = joint_vel.clone()
            prev_joint_acc = joint_acc.clone()
            
            # Convert to numpy for printing
            if torques is not None:
                if isinstance(torques, torch.Tensor):
                    torques_np = torques.cpu().numpy()
                else:
                    torques_np = np.array(torques)
            else:
                torques_np = None
            
            if isinstance(joint_acc, torch.Tensor):
                acc_np = joint_acc.cpu().numpy()
            else:
                acc_np = np.array(joint_acc)
            
            if isinstance(jerk, torch.Tensor):
                jerk_np = jerk.cpu().numpy()
            else:
                jerk_np = np.array(jerk)
            
            # Get current timestamp
            timestamp = time.time()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            timestamp_ms = f"{timestamp:.3f}"
            
            # Format output nicely with timestamp
            if actions_np.ndim == 1:
                # Single environment
                print(f"[{time_str} ({timestamp_ms})] Step {timestep}:")
                print(f"  Actions = {actions_np}")
                if torques_np is not None:
                    print(f"  Tau (Torques) = {torques_np}")
                print(f"  Acceleration = {acc_np}")
                print(f"  Jerk = {jerk_np}")
                
                # Write to CSV
                if csv_writer is None:
                    # Initialize CSV writer with header
                    num_action_dims = len(actions_np)
                    header = ["timestamp", "step", "sim_time"]
                    header.extend([f"action_{controlled_joint_names[i]}" for i in range(num_action_dims)])
                    if torques_np is not None:
                        header.extend([f"tau_{controlled_joint_names[i]}" for i in range(len(torques_np))])
                    header.extend([f"acceleration_{controlled_joint_names[i]}" for i in range(len(acc_np))])
                    header.extend([f"jerk_{controlled_joint_names[i]}" for i in range(len(jerk_np))])
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(header)
                
                # Write data row
                row = [timestamp_ms, timestep, timestep * dt]
                row.extend(actions_np.tolist())
                if torques_np is not None:
                    row.extend(torques_np.tolist())
                row.extend(acc_np.tolist())
                row.extend(jerk_np.tolist())
                csv_writer.writerow(row)
            else:
                # Multiple environments
                for env_idx in range(actions_np.shape[0]):
                    print(f"[{time_str} ({timestamp_ms})] Step {timestep}, Env {env_idx}:")
                    print(f"  Actions = {actions_np[env_idx]}")
                    if torques_np is not None:
                        print(f"  Tau (Torques) = {torques_np[env_idx]}")
                    print(f"  Acceleration = {acc_np[env_idx]}")
                    print(f"  Jerk = {jerk_np[env_idx]}")
                    
                    # Write to CSV
                    if csv_writer is None:
                        # Initialize CSV writer with header
                        num_action_dims = actions_np.shape[1]
                        header = ["timestamp", "step", "env_idx", "sim_time"]
                        header.extend([f"action_{controlled_joint_names[i]}" for i in range(num_action_dims)])
                        if torques_np is not None:
                            header.extend([f"tau_{controlled_joint_names[i]}" for i in range(torques_np.shape[1])])
                        header.extend([f"acceleration_{controlled_joint_names[i]}" for i in range(acc_np.shape[1])])
                        header.extend([f"jerk_{controlled_joint_names[i]}" for i in range(jerk_np.shape[1])])
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(header)
                    
                    # Write data row
                    row = [timestamp_ms, timestep, env_idx, timestep * dt]
                    row.extend(actions_np[env_idx].tolist())
                    if torques_np is not None:
                        row.extend(torques_np[env_idx].tolist())
                    row.extend(acc_np[env_idx].tolist())
                    row.extend(jerk_np[env_idx].tolist())
                    csv_writer.writerow(row)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    # Close CSV file
    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] CSV file saved: {csv_filename}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
