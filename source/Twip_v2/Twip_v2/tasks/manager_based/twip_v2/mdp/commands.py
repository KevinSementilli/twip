from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)
from isaaclab.utils.configclass import configclass


class WorldVelocityCommand(CommandTerm):
    """Velocity command expressed in the WORLD frame."""

    cfg: WorldVelocityCommandCfg

    def __init__(
        self,
        cfg: WorldVelocityCommandCfg,
        env,
    ):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]

        # [vx_world, vy_world, wz_world]
        self._command = torch.zeros(
            (env.num_envs, 3),
            device=env.device,
        )

    # ==============================================================
    # Command generation
    # ==============================================================

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new WORLD-frame velocity command."""

        num_commands = len(env_ids)

        self._command[env_ids, 0] = torch.empty(
            num_commands,
            device=self.device,
        ).uniform_(*self.cfg.ranges.lin_vel_x)

        self._command[env_ids, 1] = torch.empty(
            num_commands,
            device=self.device,
        ).uniform_(*self.cfg.ranges.lin_vel_y)

        self._command[env_ids, 2] = torch.empty(
            num_commands,
            device=self.device,
        ).uniform_(*self.cfg.ranges.ang_vel_z)

    def _update_command(self):
        """No post-processing required for world-frame commands."""
        pass

    def _update_metrics(self):
        """No additional command metrics."""
        pass

    # ==============================================================
    # Debug visualization
    # ==============================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create and toggle the velocity visualization markers."""

        if debug_vis:

            if not hasattr(self, "goal_vel_visualizer"):

                # Desired velocity
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )

                # Actual velocity
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )

            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)

        else:

            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize desired and actual WORLD-frame velocity."""

        # Make sure robot data is valid
        if not self.robot.is_initialized:
            return

        # ----------------------------------------------------------
        # Marker position
        # ----------------------------------------------------------

        base_pos_w = self.robot.data.root_pos_w.clone()

        # Move arrows above robot
        base_pos_w[:, 2] += 0.5

        # ----------------------------------------------------------
        # Desired velocity
        # ----------------------------------------------------------

        vel_des_arrow_scale, vel_des_arrow_quat = (
            self._resolve_xy_velocity_to_arrow(
                self.command[:, :2]
            )
        )

        # ----------------------------------------------------------
        # Actual velocity
        # ----------------------------------------------------------

        vel_arrow_scale, vel_arrow_quat = (
            self._resolve_xy_velocity_to_arrow(
                self.robot.data.root_lin_vel_w[:, :2]
            )
        )

        # ----------------------------------------------------------
        # Display
        # ----------------------------------------------------------

        self.goal_vel_visualizer.visualize(
            base_pos_w,
            vel_des_arrow_quat,
            vel_des_arrow_scale,
        )

        self.current_vel_visualizer.visualize(
            base_pos_w,
            vel_arrow_quat,
            vel_arrow_scale,
        )

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert an XY world-frame velocity into arrow scale and orientation."""

        # Get default marker scale
        default_scale = self.cfg.goal_vel_visualizer_cfg.markers["arrow"].scale

        arrow_scale = torch.tensor(
            default_scale,
            device=self.device,
        ).repeat(xy_velocity.shape[0], 1)

        # ----------------------------------------------------------
        # Arrow length = velocity magnitude
        # ----------------------------------------------------------

        arrow_scale[:, 0] *= (
            torch.linalg.norm(xy_velocity, dim=1) * 3.0
        )

        # ----------------------------------------------------------
        # Arrow direction
        # ----------------------------------------------------------

        heading_angle = torch.atan2(
            xy_velocity[:, 1],
            xy_velocity[:, 0],
        )

        zeros = torch.zeros_like(heading_angle)

        arrow_quat = math_utils.quat_from_euler_xyz(
            zeros,
            zeros,
            heading_angle,
        )

        # IMPORTANT:
        # No multiplication by root_quat_w here.
        #
        # The command and actual velocity are already expressed
        # in the WORLD frame.
        #
        # Therefore this quaternion is already a world-frame
        # orientation.

        return arrow_scale, arrow_quat

    # ==============================================================
    # Command property
    # ==============================================================

    @property
    def command(self) -> torch.Tensor:
        return self._command


@configclass
class WorldVelocityCommandCfg(CommandTermCfg):
    """Configuration for a velocity command expressed in the WORLD frame."""

    # This is now the actual default value of every config instance.
    class_type: type[CommandTerm] = WorldVelocityCommand

    asset_name: str = "robot"

    resampling_time_range: tuple[float, float] = (2.0, 5.0)

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()

    # Visualization configuration
    goal_vel_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal")

    current_vel_visualizer_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current")

    # Same size as Isaac Lab's built-in velocity visualization
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
