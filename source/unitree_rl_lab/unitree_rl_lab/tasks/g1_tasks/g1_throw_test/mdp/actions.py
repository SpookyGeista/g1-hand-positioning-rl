# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Action functions for G1 Throwing Test Task"""

import isaaclab.envs.mdp as base_mdp

class JointPositionActionCfg(base_mdp.JointPositionActionCfg):
    """Joint position action configuration for G1 robot - only 3 joints for throwing."""
    pass
