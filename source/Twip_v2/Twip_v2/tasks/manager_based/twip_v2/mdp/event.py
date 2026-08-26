from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def push_robot(env: ManagerBasedRLEnv, env_ids, asset_cfg: SceneEntityCfg, velocity_range: dict ) -> torch.Tensor :

    asset: Articulation = env.scene[asset_cfg.name]

    # Sample horizontal velocity in WORLD frame
    num_envs = len(env_ids)

    vx = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["x"][0],
        velocity_range["x"][1],
    )

    vy = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["y"][0],
        velocity_range["y"][1],
    )

    vz = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["z"][0],
        velocity_range["z"][1],
    )

    wx = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["roll"][0],
        velocity_range["roll"][1],
    )

    wy = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["pitch"][0],
        velocity_range["pitch"][1],
    )

    wz = torch.empty(num_envs, device=env.device).uniform_(
        velocity_range["yaw"][0],
        velocity_range["yaw"][1],
    )

    velocity = torch.zeros(
        (num_envs, 6),
        device=env.device,
    )

    # Replace only the world-frame horizontal velocity
    velocity[:, 0] = vx
    velocity[:, 1] = vy
    velocity[:, 2] = vz
    velocity[:, 3] = wx
    velocity[:, 4] = wy
    velocity[:, 5] = wz

    asset.write_root_com_velocity_to_sim(
        root_velocity=velocity,
        env_ids=env_ids,
    )
