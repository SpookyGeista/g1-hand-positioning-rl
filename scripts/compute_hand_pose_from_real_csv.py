#!/usr/bin/env python3
"""
Berechne die Handposition (left_wrist_yaw_link) relativ zum Torso-Frame
aus einer realen G1-CSV, die mit `track_left_arm_with_policy_commands` geloggt wurde.

Die Vorwärtskinematik läuft in Isaac Lab mit dem gleichen G1-29dof-URDF
wie dein HandPosition-V2-Task. Für jede Zeile der CSV werden die linken
Arm-Gelenkwinkel gesetzt und daraus die Handposition im Torso-Koordinatensystem
berechnet.

Beispiel-Aufruf (von deinem Isaac-Lab-Python / Omniverse-Terminal aus):

    python scripts/compute_hand_pose_from_real_csv.py \\
        -f "live logs g1/live log dance/left_arm_policy_log_20260227_102752.csv" \\
        --headless
"""

import argparse
import pathlib

import pandas as pd

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Berechne Handposition relativ zum Torso aus G1-Real-Logs (CSV)."
    )
    parser.add_argument(
        "--input_file",
        "-f",
        type=str,
        required=True,
        help="Pfad zur Eingabe-CSV (von track_left_arm_with_policy_commands).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Pfad zur Output-CSV (Standard: gleicher Name + '_with_hand_pos.csv').",
    )

    # Isaac Lab / AppLauncher Args (z.B. --headless, --device, ...)
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    if not args.output_csv:
        in_path = pathlib.Path(args.input_file)
        args.output_csv = str(in_path.with_name(in_path.stem + "_with_hand_pos.csv"))

    return args


def run(simulation_app, args_cli):
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from unitree_rl_lab.tasks.hand_positioning_V2.robots.g1_29dof.hand_position_env_cfg import (
        RobotSceneCfg,
    )

    # 1) Simulation & Szene aufsetzen (wie in den anderen Skripten)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = RobotSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]

    # Mapping von SDK-Joint-Namen -> Articulation-Joint-Indizes
    joint_sdk_names = scene.cfg.robot.joint_sdk_names
    robot_joint_indexes = robot.find_joints(joint_sdk_names, preserve_order=True)[0]

    name_to_sdk_index = {name: i for i, name in enumerate(joint_sdk_names)}

    # Link, dessen Position wir auswerten wollen (wie im HandPosition-Command)
    hand_body_name = "left_wrist_yaw_link"
    hand_body_indices = robot.find_bodies(hand_body_name)[0]
    if len(hand_body_indices) == 0:
        raise ValueError(f"Body '{hand_body_name}' nicht im G1-URDF gefunden.")
    hand_body_index = hand_body_indices[0]

    # CSV laden
    df = pd.read_csv(args_cli.input_file)

    # Spaltennamen aus deinem Logging-Skript
    csv_joint_cols = {
        "left_shoulder_pitch_joint": "LeftShoulderPitch_q",
        "left_shoulder_roll_joint": "LeftShoulderRoll_q",
        "left_shoulder_yaw_joint": "LeftShoulderYaw_q",
        "left_elbow_joint": "LeftElbow_q",
        "left_wrist_roll_joint": "LeftWristRoll_q",
        "left_wrist_pitch_joint": "LeftWristPitch_q",
        "left_wrist_yaw_joint": "LeftWristYaw_q",
    }

    for col in csv_joint_cols.values():
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in CSV {args_cli.input_file}")

    # Ergebnis-Listen für die Handposition relativ zum Torso
    hand_pos_x = []
    hand_pos_y = []
    hand_pos_z = []

    num_rows = len(df)
    row_idx = 0

    print(
        f"Starte FK-Auswertung für {num_rows} Zeilen aus '{args_cli.input_file}'. "
        f"Output wird geschrieben nach '{args_cli.output_csv}'."
    )

    # 2) Hauptschleife: durch CSV laufen, solange die App läuft
    while simulation_app.is_running() and row_idx < num_rows:
        row = df.iloc[row_idx]

        # Joint-States vom Default weg überschreiben
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        for joint_name, col_name in csv_joint_cols.items():
            sdk_idx = name_to_sdk_index[joint_name]
            art_idx = robot_joint_indexes[sdk_idx]
            q_val = float(row[col_name])
            joint_pos[0, art_idx] = q_val

        # Gelenkzustände in die Simulation schreiben
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Einen Render-/Update-Schritt, damit body_pos_w / root_pos_w aktuell sind
        sim.render()
        scene.update(sim.get_physics_dt())

        # Handposition relativ zum Torso (Root-Frame)
        body_pos_w = robot.data.body_pos_w[0, hand_body_index, :3]
        root_pos_w = robot.data.root_pos_w[0, :3]
        pos_rel = body_pos_w - root_pos_w

        pos_rel = pos_rel.detach().cpu()
        hand_pos_x.append(float(pos_rel[0].item()))
        hand_pos_y.append(float(pos_rel[1].item()))
        hand_pos_z.append(float(pos_rel[2].item()))

        row_idx += 1

    if row_idx < num_rows:
        print(
            f"\nHinweis: Simulation wurde vorzeitig beendet "
            f"({row_idx}/{num_rows} Zeilen verarbeitet)."
        )

    # 3) Ergebnisse zurück in die CSV schreiben
    df = df.iloc[:row_idx].copy()
    df["hand_pos_x_torso"] = hand_pos_x
    df["hand_pos_y_torso"] = hand_pos_y
    df["hand_pos_z_torso"] = hand_pos_z

    out_path = pathlib.Path(args_cli.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nFertig. Datei mit Handpositionen gespeichert unter:\n  {out_path}")


def main():
    args_cli = parse_args()

    # Omniverse / Isaac Lab App starten (wie in den anderen Skripten)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        run(simulation_app, args_cli)
    finally:
        # App sauber beenden
        simulation_app.close()


if __name__ == "__main__":
    main()

