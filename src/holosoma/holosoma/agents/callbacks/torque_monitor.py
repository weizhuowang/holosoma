"""Real-time torque and thermal monitoring callback for evaluation.

Ported from FAR-FALCON agents/callbacks/torque_monitor.py.
MonitorServer and HTML template are bundled in this directory
(originally from toy_thermal_ig/sim2sim_monitor).

Requires ``flask-socketio`` at runtime (``pip install flask-socketio``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from holosoma.agents.callbacks.base_callback import RLEvalCallback


class TorqueMonitor(RLEvalCallback):
    """Real-time motor torque and thermal monitoring callback for robot evaluation.

    Launches a web server (default port 5001) that displays per-joint
    motor torque utilization, thermal data, and velocity tracking.

    Enable via experiment config::

        algo=replace(algo.ppo, config=replace(algo.ppo.config,
            eval_callbacks={"torque_monitor": TorqueMonitorCfg()}))

    Requires ``flask-socketio`` (``pip install flask-socketio``).
    """

    def __init__(self, config: Any, training_loop: Any):
        super().__init__(config, training_loop)
        self.env = self.training_loop.env
        self.num_envs = self.env.num_envs

        self.use_weblogger = getattr(config, "use_weblogger", True)
        self.log_single_robot = getattr(config, "log_single_robot", True)
        sim_dt = getattr(config, "sim_dt", 0.02)
        port = getattr(config, "port", 5001)

        self.monitor = None
        if self.use_weblogger:
            from holosoma.agents.callbacks.monitor_server import MonitorServer

            self.monitor = MonitorServer(dt=sim_dt, port=port)

    def on_pre_evaluate_policy(self):
        self.robot_num_dofs = self.env.num_dofs
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()

        # Camera follow
        self.camera_follow = True
        self.camera_distance = 1.5
        self.camera_height = 0.0
        self.camera_angle = 0.0
        self.orbit_speed = 0.2

        if hasattr(self.env, "dof_names"):
            self.dof_names = self.env.dof_names
        else:
            self.dof_names = [f"Joint_{i}" for i in range(self.robot_num_dofs)]

        if self.monitor is not None:
            self.monitor.set_robot_config(
                num_dofs=self.robot_num_dofs,
                torque_limits=self.log_dof_torque_limits,
                dof_names=self.dof_names,
            )

    def on_pre_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        if self.camera_follow and hasattr(self.env, "simulator"):
            self._update_camera_follow()

        if self.monitor is None:
            return actor_state

        # Torque data
        motor_torques = getattr(self.env, "motor_torques_AB", None)
        if motor_torques is not None:
            current_torques = motor_torques.cpu().numpy()
        else:
            return actor_state

        if self.log_single_robot:
            torque_data = current_torques[0]
        else:
            torque_data = current_torques.mean(axis=0)

        torque_utilization = np.abs(torque_data) / self.log_dof_torque_limits * 100.0

        # Velocity data
        velocity_data = {}
        commands = getattr(self.env, "command_manager", None)
        if commands is not None:
            cmds = commands.commands
            root_states = self.env.simulator.robot_root_states
            if self.log_single_robot:
                velocity_data = {
                    "command_vel": cmds[0, :2].cpu().numpy(),
                    "actual_vel": root_states[0, 7:9].cpu().numpy(),
                }
            else:
                velocity_data = {
                    "command_vel": cmds[:, :2].mean(dim=0).cpu().numpy(),
                    "actual_vel": root_states[:, 7:9].mean(dim=0).cpu().numpy(),
                }

        # Thermal data
        thermal_data = {}
        if hasattr(self.env, "get_thermal_info"):
            from holosoma.agents.callbacks.monitor_server import process_thermal_data

            thermal_info = self.env.get_thermal_info()
            if thermal_info:
                thermal_data = process_thermal_data(thermal_info, self.log_single_robot)

        self.monitor.log_data(
            {
                "current_torques": torque_data,
                "torque_utilization": torque_utilization,
                "max_torques": self.log_dof_torque_limits,
                "thermal_data": thermal_data,
                "velocity_data": velocity_data,
            }
        )

        return actor_state

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        step = actor_state.get("step", 0)
        plot_interval = getattr(self.config, "plot_update_interval", 2)
        if self.monitor is not None and ((step + 1) % plot_interval == 0):
            self.monitor.update()
        return actor_state

    def on_post_evaluate_policy(self):
        pass

    def _update_camera_follow(self):
        try:
            import math

            import isaacgym.gymapi as gymapi

            robot_pos = self.env.simulator.robot_root_states[0, :3].cpu().numpy()
            self.camera_angle += self.orbit_speed * self.env.dt

            cam_x = robot_pos[0] + self.camera_distance * math.cos(self.camera_angle)
            cam_y = robot_pos[1] + self.camera_distance * math.sin(self.camera_angle)
            cam_z = robot_pos[2] + self.camera_height

            cam_pos = gymapi.Vec3(cam_x, cam_y, cam_z)
            cam_target = gymapi.Vec3(robot_pos[0], robot_pos[1], robot_pos[2])

            self.env.simulator.gym.viewer_camera_look_at(self.env.simulator.viewer, None, cam_pos, cam_target)
        except Exception:
            pass
