
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.
Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TWIP_CFG = ArticulationCfg(

# Where is the USD file for this robot?
spawn=sim_utils.UsdFileCfg(       
    usd_path=f"/home/legion5/Projects/Twip/Twip/source/Twip/assets/twip.usd", 
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
# What is its initial position of the robot, and its joints?
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
# What parts of the robot move, and how stiff / damped are they?
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=6.0,
            stiffness=0.0,
            damping=0.0,
        ),
    }
)
