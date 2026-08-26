# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


##
# Stability environments.
##

gym.register(
    id="Twip-Stability-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twip_v2_stability_env_cfg:TwipStabilityEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stability_cfg.yaml",
    },
)

gym.register(
    id="Twip-Stability-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twip_v2_stability_env_cfg:TwipStabilityEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_stability_cfg.yaml",
    },
)


##
# Locomotion environments.
##

gym.register(
    id="Twip-Locomotion-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twip_v2_locomotion_env_cfg:TwipLocomotionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_locomotion_cfg.yaml",
    },
)

gym.register(
    id="Twip-Locomotion-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.twip_v2_locomotion_env_cfg:TwipLocomotionEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_locomotion_cfg.yaml",
    },
)
