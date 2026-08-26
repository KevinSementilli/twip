# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Pre-defined configs
##

from .twip_articu_conf import TWIP_CFG  # isort:skip
from .twip_v2_env_cfg import TwipEnvCfg


##
# MDP settings
##


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_wheel_joint", "right_wheel_joint"]),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_body_orientation = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "roll": (0.0, 0.0),
                "pitch": (-math.radians(5.0), math.radians(5.0)),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # initial disturbance
    initial_push = EventTerm(
        func=mdp.push_robot,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Stay alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Penalty for falling
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # Keep the chassis upright
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Penalize body angular velocity
    body_ang_vel = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Penalize excessive wheel velocity
    wheel_vel = RewTerm(func=mdp.joint_vel_l2,weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot",
                joint_names=["left_wheel_joint","right_wheel_joint"],
            ),
        }
    )

    # Penalize rapid changes in wheel torque
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # ============================================================
    # Stage 2: Light -> moderate disturbance
    # ============================================================

    disturbance_stage_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.initial_push.params.velocity_range",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "start_step": 2_500,
            },
        },
    )

    # ============================================================
    # Stage 3: Moderate -> heavy disturbance
    # ============================================================

    disturbance_stage_3 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.initial_push.params.velocity_range",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": {
                    "x": (-0.75, 0.75),
                    "y": (-0.75, 0.75),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "start_step": 5_000,
            },
        },
    )

    # ============================================================
    # Stage 3: Moderate -> heavy disturbance
    # ============================================================

    disturbance_stage_4 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.initial_push.params.velocity_range",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "start_step": 7_500,
            },
        },
    )


##
# Environment configuration
##


@configclass
class TwipStabilityEnvCfg(TwipEnvCfg):
    """Configuration for the Twip stability environment."""
    
    events = EventCfg()
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()


@configclass
class TwipStabilityEnvCfg_PLAY(TwipStabilityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        