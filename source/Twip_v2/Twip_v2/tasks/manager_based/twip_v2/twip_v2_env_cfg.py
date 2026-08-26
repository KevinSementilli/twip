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


##
# Scene definition
##


@configclass
class TwipSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robot
    robot = TWIP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        joint_names=["left_wheel_joint", "right_wheel_joint"], 
        scale=6.0
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    base_velocity = mdp.WorldVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 5.0),
        debug_vis=True,
        ranges = mdp.WorldVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Wheel positions
        wheel_pos = ObsTerm(func=mdp.joint_pos_rel)

        # Wheel velocities
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Body orientation relative to gravity
        base_orientation = ObsTerm(func=mdp.projected_gravity)

        # Body linear velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # Desired velocity command
        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Previous action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False # change to True if you want to add noise to the observations
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Robot falls over
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": math.radians(30.0),
        },
    )


##
# Environment configuration
##


@configclass
class TwipEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = TwipSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands =CommandsCfg()
    # MDP settings
    terminations = TerminationsCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (5.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 60.0


@configclass
class TwipEnvCfg_PLAY(TwipEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        