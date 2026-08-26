# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils import math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def velocity_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    actual_velocity = asset.data.root_lin_vel_b[:, :2]
    commanded_velocity = env.command_manager.get_command(command_name)[:, :2]

    return torch.sum(torch.square(actual_velocity - commanded_velocity), dim=1)


def velocity_magnitude_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    # ------------------------------------------------------------
    # Actual velocity in BODY frame
    # ------------------------------------------------------------
    actual_velocity_b = asset.data.root_lin_vel_b

    actual_y_b = actual_velocity_b[:, 1]
    actual_z_b = actual_velocity_b[:, 2]

    # ------------------------------------------------------------
    # Commanded velocity in WORLD frame
    # ------------------------------------------------------------
    command = env.command_manager.get_command(command_name)

    command_w = torch.zeros((command.shape[0], 3), device=env.device)

    command_w[:, 0] = command[:, 0]  # world X
    command_w[:, 1] = command[:, 1]  # world Y

    # ------------------------------------------------------------
    # Convert WORLD command -> BODY command
    # ------------------------------------------------------------
    command_b = math.quat_apply_inverse(asset.data.root_quat_w.torch, command_w)

    commanded_y_b = command_b[:, 1]
    commanded_z_b = command_b[:, 2]

    # ------------------------------------------------------------
    # L2 velocity error in BODY Y-Z plane
    # ------------------------------------------------------------
    velocity_error = torch.stack((actual_y_b - commanded_y_b, actual_z_b - commanded_z_b), dim=1)

    return torch.sum(torch.square(velocity_error), dim=1)
