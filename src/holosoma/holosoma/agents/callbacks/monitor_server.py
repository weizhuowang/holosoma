"""Standalone real-time monitoring server for robot data visualization."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def process_thermal_data(thermal_info: Dict, log_single_robot: bool = True) -> Dict:
    """Process thermal data from environment.

    Args:
        thermal_info: Dictionary with joint thermal data containing winding_temp and case_temp tensors
        log_single_robot: If True, extract data for first robot only. If False, average across all robots.

    Returns:
        Processed thermal data dictionary with float values
    """
    thermal_data = {}

    if log_single_robot:
        # Extract thermal data for first robot
        for joint_name, joint_thermal in thermal_info.items():
            thermal_data[joint_name] = {
                "winding_temp": float(joint_thermal["winding_temp"][0].cpu().numpy()),
                "case_temp": float(joint_thermal["case_temp"][0].cpu().numpy()),
            }
    else:
        # Average thermal data across all robots
        for joint_name, joint_thermal in thermal_info.items():
            thermal_data[joint_name] = {
                "winding_temp": float(joint_thermal["winding_temp"].mean().cpu().numpy()),
                "case_temp": float(joint_thermal["case_temp"].mean().cpu().numpy()),
            }

    return thermal_data


class MonitorServer:
    """Web-based real-time visualization server for robot monitoring data."""

    # Configuration constants
    TORQUE_WINDOW_SIZE = 250  # 5 seconds at 50Hz
    TORQUE_WINDOW_SECONDS = 5.0
    THERMAL_DOWNSAMPLE_RATE = 10  # Store every 10th sample
    THERMAL_WINDOW_SIZE = 250  # 250 downsampled points = 50 seconds
    THERMAL_WINDOW_SECONDS = 50.0
    PEAK_SAMPLE_COUNT = 50  # Samples for peak calculation

    def __init__(
        self,
        dt: float,
        port: int = 5001,
        template_path: Optional[str] = None,
        torque_window_size: int = 250,
        thermal_downsample_rate: int = 10,
    ):
        """Initialize the monitoring server.

        Args:
            dt: Simulation timestep in seconds
            port: Port number for the web server
            template_path: Path to HTML template file (defaults to same directory)
            torque_window_size: Number of samples to keep in torque buffer
            thermal_downsample_rate: Downsampling rate for thermal data
        """
        self.dt = dt
        self.port = port
        self.simulation_time = 0.0
        self.total_steps = 0
        self.thermal_sample_counter = 0

        # Allow customization of buffer sizes
        self.TORQUE_WINDOW_SIZE = torque_window_size
        self.THERMAL_DOWNSAMPLE_RATE = thermal_downsample_rate

        # Set template path
        if template_path is None:
            # Default to template in same directory as this module
            self.template_path = Path(__file__).parent / "torque_monitor_template.html"
        else:
            self.template_path = Path(template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")

        # Initialize data buffers
        self._init_buffers()

        # Web server setup
        self._init_web_server()

        # Track connection state
        self.connected_clients = {}
        self.data_counters = {"torque": 0, "thermal": 0}

        # Robot configuration
        self.robot_config = None

        # Velocity data
        self.velocity_data = None

    def _init_buffers(self):
        """Initialize all data buffers."""
        self.torque_buffer = {
            "utilization": deque(maxlen=self.TORQUE_WINDOW_SIZE),
            "torques": deque(maxlen=self.TORQUE_WINDOW_SIZE),
            "timestamps": deque(maxlen=self.TORQUE_WINDOW_SIZE),
        }

        self.thermal_buffer = {
            "data": deque(maxlen=self.THERMAL_WINDOW_SIZE),
            "timestamps": deque(maxlen=self.THERMAL_WINDOW_SIZE),
        }

        # Cache for peak values
        self.peak_cache = {"values": None, "step": -1}

    def _init_web_server(self):
        """Initialize Flask web server."""
        from flask import Flask, send_file  # noqa: F811
        from flask_socketio import SocketIO

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, logger=False, engineio_logger=False, cors_allowed_origins="*")
        self.server_thread = None

        # Suppress Flask logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        # Register routes
        @self.app.route("/")
        def index():
            return send_file(str(self.template_path))

        @self.socketio.on("connect")
        def handle_connect(auth=None):
            from flask import request

            client_id = request.sid
            self.connected_clients[client_id] = {
                "last_counters": dict(self.data_counters),
                "connection_time": time.time(),
            }
            print(f"Client connected: {client_id}")
            self._send_full_update(client_id)

        @self.socketio.on("disconnect")
        def handle_disconnect():
            from flask import request

            self.connected_clients.pop(request.sid, None)
            print(f"Client disconnected: {request.sid}")

    def set_robot_config(self, num_dofs: int, torque_limits: np.ndarray, dof_names: List[str]):
        """Configure robot-specific parameters.

        Args:
            num_dofs: Number of degrees of freedom
            torque_limits: Array of torque limits for each DOF
            dof_names: List of joint names
        """
        self.robot_config = {"num_dofs": num_dofs, "torque_limits": torque_limits, "dof_names": dof_names}

    def log_data(self, data_dict: Dict[str, Any]):
        """Log torque and thermal data.

        Args:
            data_dict: Dictionary containing:
                - current_torques: Current torque values
                - torque_utilization: Torque utilization percentages
                - max_torques: Maximum torque limits
                - thermal_data: Optional thermal data dictionary
                - velocity_data: Optional velocity data dictionary
        """
        self.simulation_time += self.dt
        self.total_steps += 1

        # Log torque data
        self.torque_buffer["utilization"].append(data_dict["torque_utilization"].copy())
        self.torque_buffer["torques"].append(data_dict["current_torques"].copy())
        self.torque_buffer["timestamps"].append(self.simulation_time)
        self.data_counters["torque"] += 1

        # Store velocity data if available
        if data_dict.get("velocity_data"):
            self.velocity_data = data_dict["velocity_data"]

        # Log thermal data (downsampled)
        if data_dict.get("thermal_data"):
            self.thermal_sample_counter += 1
            if self.thermal_sample_counter % self.THERMAL_DOWNSAMPLE_RATE == 0:
                self.thermal_buffer["data"].append(data_dict["thermal_data"])
                self.thermal_buffer["timestamps"].append(self.simulation_time)
                self.data_counters["thermal"] += 1

    def update(self):
        """Start server if needed and send updates to connected clients."""
        if self.server_thread is None:
            self._start_server()

        if self.connected_clients:
            self._send_incremental_updates()

    def start(self):
        """Start the monitoring server."""
        if self.server_thread is None:
            self._start_server()

    def _start_server(self):
        """Start the Flask web server."""
        self.server_thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app, debug=False, use_reloader=False, port=self.port, log_output=False, allow_unsafe_werkzeug=True
            ),
            daemon=True,
        )
        self.server_thread.start()
        time.sleep(1.0)
        print(f"\n🔧 Monitor Server Web Interface: http://127.0.0.1:{self.port}\n")

    def _get_current_data(self) -> Dict:
        """Get current state data."""
        if not self.torque_buffer["utilization"]:
            return {}

        current = {
            "utilization": self.torque_buffer["utilization"][-1].tolist(),
            "torques": self.torque_buffer["torques"][-1].tolist(),
            "peak": self._get_peak_values(),
        }

        # Add velocity if available
        if self.velocity_data:
            current["velocity"] = {
                "command_x": float(self.velocity_data["command_vel"][0]),
                "command_y": float(self.velocity_data["command_vel"][1]),
                "actual_x": float(self.velocity_data["actual_vel"][0]),
                "actual_y": float(self.velocity_data["actual_vel"][1]),
            }

        # Add thermal if available
        if self.thermal_buffer["data"]:
            thermal_data = self.thermal_buffer["data"][-1]
            joint_names = list(thermal_data.keys())
            current["thermal"] = {
                "joints": joint_names,
                "winding": [thermal_data[j]["winding_temp"] for j in joint_names],
                "case": [thermal_data[j]["case_temp"] for j in joint_names],
            }

        return current

    def _get_peak_values(self) -> List[float]:
        """Calculate peak values with caching."""
        if self.peak_cache["step"] != self.total_steps:
            recent_data = list(self.torque_buffer["utilization"])
            if recent_data:
                sample_count = min(self.PEAK_SAMPLE_COUNT, len(recent_data))
                recent_array = np.array(recent_data[-sample_count:])
                self.peak_cache["values"] = np.max(recent_array, axis=0).tolist()
            else:
                self.peak_cache["values"] = [0] * self.robot_config["num_dofs"]
            self.peak_cache["step"] = self.total_steps

        return self.peak_cache["values"]

    def _send_full_update(self, client_id: str):
        """Send complete data to newly connected client."""
        if not self.torque_buffer["utilization"] or not self.robot_config:
            return

        data = {
            "message_type": "full_update",
            "metadata": {
                "joint_names": self.robot_config["dof_names"],
                "torque_limits": self.robot_config["torque_limits"].tolist(),
                "num_dofs": self.robot_config["num_dofs"],
            },
            "current": self._get_current_data(),
            "timeline": self._get_timeline_data(full=True),
        }

        # Update client state
        self.connected_clients[client_id]["last_counters"] = dict(self.data_counters)

        # Send data
        self.socketio.emit("torque_update", json.dumps(data, separators=(",", ":")), to=client_id)

    def _send_incremental_updates(self):
        """Send incremental updates to all connected clients."""
        current_data = self._get_current_data()
        if not current_data:
            return

        for client_id, client_state in list(self.connected_clients.items()):
            # Skip recently connected clients
            if time.time() - client_state["connection_time"] < 0.5:
                continue

            # Calculate new data points
            new_points = {
                "torque": self.data_counters["torque"] - client_state["last_counters"]["torque"],
                "thermal": self.data_counters["thermal"] - client_state["last_counters"]["thermal"],
            }

            # Build update
            update = {
                "message_type": "incremental_update",
                "current": current_data,
                "new_data": self._get_new_data_points(new_points),
            }

            # Update client state
            client_state["last_counters"] = dict(self.data_counters)

            # Send update
            self.socketio.emit("torque_update", json.dumps(update, separators=(",", ":")), to=client_id)

    def _get_timeline_data(self, full: bool = False) -> Dict:
        """Get timeline data for visualization."""
        timeline = {}

        # Torque timeline
        if self.torque_buffer["utilization"]:
            torque_data = list(self.torque_buffer["utilization"])
            timestamps = list(self.torque_buffer["timestamps"])

            timeline["torque"] = {"time": timestamps, "data": [d.tolist() for d in torque_data]}

        # Thermal timeline
        if self.thermal_buffer["data"]:
            thermal_data = list(self.thermal_buffer["data"])
            thermal_times = list(self.thermal_buffer["timestamps"])

            # Process thermal data by joint
            joints_data = {}
            for data_point in thermal_data:
                for joint, temps in data_point.items():
                    if joint not in joints_data:
                        joints_data[joint] = {"winding": [], "case": []}
                    joints_data[joint]["winding"].append(temps["winding_temp"])
                    joints_data[joint]["case"].append(temps["case_temp"])

            timeline["thermal"] = {"time": thermal_times, "joints": joints_data}

        return timeline

    def _get_new_data_points(self, new_points: Dict[str, int]) -> Dict:
        """Get only the new data points for incremental update."""
        new_data = {}

        # New torque points
        if new_points["torque"] > 0:
            torque_list = list(self.torque_buffer["utilization"])
            time_list = list(self.torque_buffer["timestamps"])
            n = min(new_points["torque"], len(torque_list))

            if n > 0:
                new_data["torque"] = {"time": time_list[-n:], "data": [d.tolist() for d in torque_list[-n:]]}

        # New thermal points
        if new_points["thermal"] > 0 and self.thermal_buffer["data"]:
            thermal_list = list(self.thermal_buffer["data"])
            thermal_times = list(self.thermal_buffer["timestamps"])
            n = min(new_points["thermal"], len(thermal_list))

            if n > 0:
                # Process new thermal data
                joints_data = {}
                for data_point in thermal_list[-n:]:
                    for joint, temps in data_point.items():
                        if joint not in joints_data:
                            joints_data[joint] = {"winding": [], "case": []}
                        joints_data[joint]["winding"].append(temps["winding_temp"])
                        joints_data[joint]["case"].append(temps["case_temp"])

                new_data["thermal"] = {"time": thermal_times[-n:], "joints": joints_data}

        return new_data