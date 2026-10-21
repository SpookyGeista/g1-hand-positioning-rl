# Hand-positioning tools

Apply the overlay and configure robot assets as described in the [README](../README.md). Run these commands from the patched **Unitree RL Lab checkout**, using its Isaac environment.

## Discover tasks and train

```bash
python scripts/list_envs.py
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-HandPosition --headless
```

The alternate task is `Unitree-G1-29dof-HandPosition-V2`. The trainer creates checkpoints under its configured log directory.

## Playback with interactive hand targets

```bash
python scripts/rsl_rl/play_csv_target_hand.py \
  --task Unitree-G1-29dof-HandPosition --checkpoint /path/to/model.pt
```

This loads a trained policy. When prompted in the terminal, enter a target as three space-separated values, for example `0.25 0.15 0.20`. The script updates the task's relative hand-position command and records left-arm feedback/targets to CSV. Inspect `scripts/rsl_rl/play_csv.py` for the separate checkpoint-playback/logging entry point.

## Compute hand positions from recorded joints

```bash
python scripts/compute_hand_pose_from_real_csv.py \
  --input_file /path/to/joint_recording.csv \
  --output_csv /path/to/hand_positions.csv --headless
```

This starts Isaac Lab to evaluate the configured robot kinematics. The calculation uses `left_wrist_yaw_link` relative to the torso. The input must follow the joint-column layout expected by the script; the filename alone does not establish that a recording came from hardware.

## Export deployment configuration

```bash
python scripts/export_deploy_cfg.py \
  --task Unitree-G1-29dof-HandPosition \
  --log_dir /path/to/training_run --device cpu --num_envs 1
```

The helper exports the task's deployment configuration. Checkpoints, asset paths, joint order, and deployment configuration must belong to the same trained task. The additional hand-position FSM is an integration component, rather than a standalone launcher.
