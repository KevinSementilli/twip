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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 2.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
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
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
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
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # Keep the chassis upright
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0,
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
    action_rate = RewTerm(func=mdp.action_rate_l2,weight=-0.001)

    # Track commanded forward velocity
    velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"std": 0.5, "command_name": "base_velocity"}
    )

    # yaw_tracking = RewTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=0.5,
    #     params={
    #         "std": 0.5,
    #         "command_name": "base_velocity",
    #     },
    # )


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
            "modify_fn": mdp.set_disturbance,
            "modify_params": {
                "x_range": (-2.0, 2.0),
                "y_range": (-2.0, 2.0),
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
            "modify_fn": mdp.set_disturbance,
            "modify_params": {
                "x_range": (-4.0, 4.0),
                "y_range": (-4.0, 4.0),
                "start_step": 5_000,
            },
        },
    )

    # ============================================================
    # Stage 4: velocity tracking at +/- 1 m/s
    # ============================================================

    disturbance_stage_4 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.initial_push.params.velocity_range",
            "modify_fn": mdp.set_disturbance,
            "modify_params": {
                "x_range": (0.0, 0.0),
                "y_range": (0.0, 0.0),
                "start_step": 7_500,
            },
        },
    )

    command_stage_4 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": (-1.0, 1.0),
                "start_step": 7_500,
            },
        },
    )

    upright_stage_4 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "upright",
            "weight": -0.2,
            "num_steps": 7_500,
        },
    )

    # ============================================================
    # Stage 5: 15,000 steps
    # Increase velocity command to +/- 2 m/s
    # ============================================================

    command_stage_5 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": (-2.0, 2.0),
                "start_step": 10_000,
            },
        },
    )

    # ============================================================
    # Stage 6: 20,000 steps
    # Increase velocity command to +/- 4 m/s
    # ============================================================

    command_stage_6 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": mdp.set_curr_param,
            "modify_params": {
                "value": (-4.0, 4.0),
                "start_step": 12_500,
            },
        },
    )


##
# Environment configuration
##


@configclass
class TwipEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = TwipSceneCfg(num_envs=8192, env_spacing=2.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands =CommandsCfg()
    # MDP settings
    terminations = TerminationsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()

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
        