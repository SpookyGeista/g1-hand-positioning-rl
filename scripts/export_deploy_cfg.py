#!/usr/bin/env python

"""
Kleines CLI-Skript zum Exportieren einer deploy.yaml für einen trainierten IsaacLab-Task.

Beispielaufruf (im Projekt-Root, in env_isaaclab):

    python scripts/export_deploy_cfg.py \
        --task Unitree-G1-29dof-HandPosition-Lift \
        --log_dir logs/rsl_rl/Unitree-G1-29dof-HandPosition-Lift/<dein_run_name>

Die Datei wird dann unter:

    <log_dir>/params/deploy.yaml

angelegt (Ordner wird bei Bedarf erstellt).
"""

import argparse
import os

import gymnasium as gym

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exportiere deploy.yaml für einen trainierten Unitree IsaacLab-Task."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Gym-Task-Name, z.B. 'Unitree-G1-29dof-HandPosition-Lift'.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Log-Verzeichnis des Trainingslaufs (z.B. logs/rsl_rl/<task>/<run_name>).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Gerät zum Laden der Env (default: cpu). Für Export reicht cpu.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Anzahl Envs beim Laden (hat nur Einfluss auf Ladezeit, nicht auf deploy.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_name = args.task
    log_dir = os.path.abspath(args.log_dir)

    print(f"[INFO] Task:    {task_name}")
    print(f"[INFO] Logdir:  {log_dir}")

    # Env-Konfiguration laden (Play-Config)
    env_cfg = parse_env_cfg(
        task_name,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=False,
        entry_point_key="play_env_cfg_entry_point",
    )

    # Gym-Env erzeugen
    env = gym.make(task_name, cfg=env_cfg).unwrapped

    try:
        # deploy.yaml exportieren
        export_deploy_cfg(env, log_dir)
    finally:
        env.close()

    out_file = os.path.join(log_dir, "params", "deploy.yaml")
    print(f"[INFO] deploy.yaml exportiert nach: {out_file}")


if __name__ == "__main__":
    main()









