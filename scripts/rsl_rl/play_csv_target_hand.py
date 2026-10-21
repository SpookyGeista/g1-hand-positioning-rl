# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play hand-target policy with interactive terminal targets and left-arm CSV logging.

- Zielpositionen während der Sim per Terminal eingeben (x y z)
- CSV im gleichen Format wie Real-Logs (track_left_arm_with_policy_commands):
  time, tick, policy_target_x, policy_target_y, policy_target_z,
  LeftShoulderPitch_q, LeftShoulderPitch_dq, LeftShoulderPitch_ddq_calc, ...
"""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play hand-target policy with interactive targets.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os
import sys
import threading
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

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
    """Play with RSL-RL hand-target agent, interactive targets, left-arm CSV logging."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    physics_dt = env.unwrapped.cfg.sim.dt
    decimation = env.unwrapped.cfg.decimation
    policy_frequency = 1.0 / (physics_dt * decimation)
    physics_frequency = 1.0 / physics_dt

    print(f"[INFO] Physics timestep (dt): {physics_dt} s")
    print(f"[INFO] Decimation: {decimation}")
    print(f"[INFO] Policy step timestep (step_dt): {dt} s")
    print(f"[INFO] Physics frequency: {physics_frequency:.1f} Hz")
    print(f"[INFO] Policy frequency: {policy_frequency:.1f} Hz")

    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    joint_names = robot.data.joint_names

    # Left-arm joints (same format as real logs: track_left_arm_with_policy_commands)
    left_arm_joint_names = [
        ("left_shoulder_pitch_joint", "LeftShoulderPitch"),
        ("left_shoulder_roll_joint", "LeftShoulderRoll"),
        ("left_shoulder_yaw_joint", "LeftShoulderYaw"),
        ("left_elbow_joint", "LeftElbow"),
        ("left_wrist_roll_joint", "LeftWristRoll"),
        ("left_wrist_pitch_joint", "LeftWristPitch"),
        ("left_wrist_yaw_joint", "LeftWristYaw"),
    ]

    left_arm_indices_list = []
    for joint_name, _ in left_arm_joint_names:
        try:
            idx = robot.get_dof_index(joint_name)
        except (AttributeError, KeyError, ValueError):
            if joint_name in joint_names:
                idx = joint_names.index(joint_name)
            else:
                print(f"[WARNING] Left arm joint {joint_name} not found!")
                idx = None
        left_arm_indices_list.append(idx)

    left_arm_indices_list = [i for i in left_arm_indices_list if i is not None]
    left_arm_indices = torch.tensor(left_arm_indices_list, device=robot.device, dtype=torch.long)
    print(f"[INFO] Left arm joints: {[b for _, b in left_arm_joint_names]}")

    # CSV im Real-Log-Format
    csv_log_dir = os.path.join(log_dir, "play_logs")
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_filename = os.path.join(
        csv_log_dir, f"left_arm_policy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    csv_file = open(csv_filename, "w", newline="")
    header = ["time", "tick", "policy_target_x", "policy_target_y", "policy_target_z"]
    for _, base in left_arm_joint_names:
        header.extend([f"{base}_q", f"{base}_dq", f"{base}_ddq_calc", f"{base}_jerk", f"{base}_tau_est"])
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    print(f"[INFO] Logging left arm CSV to: {csv_filename}")

    prev_left_dq = None
    prev_left_ddq = None
    prev_left_time = None

    # Interaktives Target: x y z aus Terminal
    target_pos_queue = []
    target_pos_lock = threading.Lock()
    stop_input_thread = threading.Event()

    def input_thread_func():
        print("[INFO] Interactive mode: Enter target positions as 'x y z' (e.g. '0.25 0.15 0.2')")
        while not stop_input_thread.is_set():
            try:
                line = sys.stdin.readline()
                if not line or stop_input_thread.is_set():
                    break
                line = line.strip()
                if line.lower() == "q":
                    break
                if line:
                    parts = line.split()
                    if len(parts) == 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        with target_pos_lock:
                            target_pos_queue.append((x, y, z))
                        print(f"[INFO] Queued target: x={x}, y={y}, z={z}")
                    else:
                        print(f"[WARNING] Expected 3 values (x y z), got {len(parts)}")
            except (ValueError, Exception) as e:
                if not stop_input_thread.is_set():
                    print(f"[WARNING] Invalid input: {e}")

    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()

        # Neue Targets aus Queue setzen
        with target_pos_lock:
            if target_pos_queue:
                new_target = target_pos_queue.pop(0)
                try:
                    cm = env.unwrapped.command_manager
                    if hasattr(cm, "_terms") and "hand_position" in cm._terms:
                        cmd = cm._terms["hand_position"]
                        if hasattr(cmd, "target_pos_rel"):
                            cmd.target_pos_rel[:, 0] = new_target[0]
                            cmd.target_pos_rel[:, 1] = new_target[1]
                            cmd.target_pos_rel[:, 2] = new_target[2]
                            print(f"[INFO] Target updated: x={new_target[0]}, y={new_target[1]}, z={new_target[2]}")
                except Exception as e:
                    print(f"[WARNING] Could not update target: {e}")

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Left-arm Zeile schreiben (Real-Log-Format)
        current_time = time.time()
        tick = timestep

        policy_target_x = policy_target_y = policy_target_z = ""
        try:
            cm = env.unwrapped.command_manager
            if hasattr(cm, "_terms") and "hand_position" in cm._terms:
                cmd = cm._terms["hand_position"]
                if hasattr(cmd, "target_pos_rel"):
                    t = cmd.target_pos_rel[0].detach().cpu().numpy()
                    if t.size >= 3:
                        policy_target_x, policy_target_y, policy_target_z = float(t[0]), float(t[1]), float(t[2])
        except Exception:
            pass

        row = [f"{current_time:.6f}", tick, policy_target_x, policy_target_y, policy_target_z]

        if left_arm_indices.numel() > 0:
            q_left = robot.data.joint_pos[0, left_arm_indices].detach().cpu().numpy()
            dq_left = robot.data.joint_vel[0, left_arm_indices].detach().cpu().numpy()

            ddq_left = None
            jerk_left = None
            if prev_left_time is not None and prev_left_dq is not None:
                dt_left = current_time - prev_left_time
                if dt_left > 0:
                    ddq_left = (dq_left - prev_left_dq) / dt_left
                    if prev_left_ddq is not None:
                        jerk_left = (ddq_left - prev_left_ddq) / dt_left

            tau_left = None
            if hasattr(robot.data, "applied_torque") and robot.data.applied_torque is not None:
                tau_left = robot.data.applied_torque[0, left_arm_indices].detach().cpu().numpy()
            elif hasattr(robot.data, "joint_torques") and robot.data.joint_torques is not None:
                tau_left = robot.data.joint_torques[0, left_arm_indices].detach().cpu().numpy()

            for i in range(len(left_arm_joint_names)):
                q_val = float(q_left[i])
                dq_val = float(dq_left[i])
                ddq_val = float(ddq_left[i]) if ddq_left is not None and i < len(ddq_left) else ""
                jerk_val = float(jerk_left[i]) if jerk_left is not None and i < len(jerk_left) else ""
                tau_val = float(tau_left[i]) if tau_left is not None and i < len(tau_left) else ""
                row.extend([q_val, dq_val, ddq_val, jerk_val, tau_val])

            prev_left_time = current_time
            prev_left_dq = dq_left.copy()
            if ddq_left is not None:
                prev_left_ddq = ddq_left.copy()
        else:
            for _ in left_arm_joint_names:
                row.extend(["", "", "", "", ""])

        csv_writer.writerow(row)

        if args_cli.video and timestep >= args_cli.video_length:
            break
        timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    stop_input_thread.set()
    env.close()
    csv_file.close()
    print(f"[INFO] CSV saved: {csv_filename}")


if __name__ == "__main__":
    main()
    simulation_app.close()
