# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import g1_throw_test_env_cfg

gym.register(
    id="Isaac-G1-Throw-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g1_throw_test_env_cfg.G1ThrowTestEnvCfg,
        "play_env_cfg_entry_point": g1_throw_test_env_cfg.G1ThrowTestPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.g1_tasks.g1_throw_test.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
    disable_env_checker=True,
)
