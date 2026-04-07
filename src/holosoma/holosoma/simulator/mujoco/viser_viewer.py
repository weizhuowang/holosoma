"""Browser-based Viser viewer for MuJoCo direct simulation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import numpy as np
import yourdfpy  # type: ignore[import-untyped]
from loguru import logger

from holosoma.utils.module_utils import get_holosoma_root

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


@dataclass(frozen=True)
class MuJoCoViewerCommand:
    """Command emitted by the browser viewer."""

    name: str
    value: Any | None = None


@dataclass(frozen=True)
class MuJoCoViewerSnapshot:
    """State snapshot consumed by the browser viewer."""

    sim_time: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_positions: np.ndarray
    commands: np.ndarray | None
    gantry_enabled: bool
    gantry_length: float | None
    gantry_force: float | None
    gantry_point: np.ndarray | None
    gantry_attachment_pos: np.ndarray | None
    camera_tracking: bool


class MuJoCoViserViewer:
    """Browser-based viewer for MuJoCo sim2sim workflows."""

    def __init__(self, simulator: MuJoCo) -> None:
        self.simulator = simulator
        self.config = simulator.simulator_config.viewer.viser

        self.server: Any | None = None
        self.robot_root: Any | None = None
        self.robot_viser: Any | None = None
        self.urdf_joint_order: list[str] = []
        self.urdf_to_sim_indices: np.ndarray | None = None

        self._pending_commands: Queue[MuJoCoViewerCommand] = Queue()
        self._ui_update_in_progress = False
        self._last_update_time: float | None = None
        self._smoothed_viewer_fps = 0.0
        self._last_tracking_target: np.ndarray | None = None
        self._server_kwargs_supported = True

        self._show_meshes_cb: Any | None = None
        self._camera_tracking_cb: Any | None = None
        self._status_handles: dict[str, Any] = {}
        self._gantry_anchor_handle: Any | None = None
        self._gantry_line_handle: Any | None = None

    def start(self) -> None:
        """Start the browser viewer and initialize its scene."""
        try:
            import viser  # type: ignore[import-not-found]
            from viser.extras import ViserUrdf  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Viser viewer requested but 'viser' is not installed in the MuJoCo environment. "
                "Install it with 'pip install viser' or rerun the MuJoCo setup."
            ) from exc

        self.server = self._create_server(viser)
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=False)

        if self.config.show_grid:
            self.server.scene.add_grid(
                "/grid",
                width=8.0,
                height=8.0,
                position=(0.0, 0.0, 0.0),
            )

        robot_urdf = yourdfpy.URDF.load(
            self._resolve_robot_urdf_path(),
            load_meshes=True,
            build_scene_graph=True,
        )
        self.robot_viser = ViserUrdf(self.server, urdf_or_path=robot_urdf, root_node_name="/robot")
        self.robot_viser.show_visual = self.config.show_meshes
        self._build_joint_mapping()

        if self.config.show_gantry:
            self._gantry_anchor_handle = self.server.scene.add_point_cloud(
                "/gantry_anchor",
                points=np.zeros((1, 3), dtype=np.float32),
                colors=np.array([[255.0, 80.0, 80.0]], dtype=np.float32),
                point_size=0.03,
                point_shape="circle",
            )
            self._gantry_line_handle = self.server.scene.add_line_segments(
                "/gantry_line",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=np.array([[[255.0, 80.0, 80.0], [255.0, 80.0, 80.0]]], dtype=np.float32),
                line_width=2.0,
            )

        self._setup_gui()
        self.update(self.simulator.get_viewer_snapshot())

        url = f"http://{self.config.host}:{self.config.port}"
        if self._server_kwargs_supported:
            logger.info(f"Viser viewer ready at {url}")
        else:
            logger.info(
                "Viser viewer started. This installed viser version does not accept explicit host/port kwargs; "
                "check the startup logs above for the actual URL."
            )

    def enqueue_command(self, name: str, value: Any | None = None) -> None:
        """Queue a simulator command for execution on the sim thread."""
        self._pending_commands.put(MuJoCoViewerCommand(name=name, value=value))

    def drain_pending_commands(self) -> list[MuJoCoViewerCommand]:
        """Drain queued browser commands."""
        drained: list[MuJoCoViewerCommand] = []
        while True:
            try:
                drained.append(self._pending_commands.get_nowait())
            except Empty:
                break
        return drained

    def update(self, snapshot: MuJoCoViewerSnapshot) -> None:
        """Push the latest simulator state into the browser scene."""
        if self.server is None or self.robot_root is None or self.robot_viser is None:
            return

        sim_joint_positions = snapshot.joint_positions
        if self.urdf_to_sim_indices is None:
            raise RuntimeError("URDF to simulator joint mapping has not been initialized")

        self.robot_root.position = snapshot.root_pos
        self.robot_root.wxyz = snapshot.root_quat_wxyz
        self.robot_viser.update_cfg(sim_joint_positions[self.urdf_to_sim_indices])

        if self._show_meshes_cb is not None:
            self.robot_viser.show_visual = bool(self._show_meshes_cb.value)

        self._update_gantry_visuals(snapshot)
        self._update_tracking(snapshot)
        self._update_status(snapshot)

    def close(self) -> None:
        """Best-effort shutdown of browser viewer resources."""
        if self.server is None:
            return

        for method_name in ("stop", "close", "shutdown"):
            method = getattr(self.server, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning(f"Failed to stop Viser server via {method_name}(): {exc}")
                break

        self.server = None

    def _create_server(self, viser_module: Any) -> Any:
        """Create a Viser server with a compatibility fallback."""
        try:
            server = viser_module.ViserServer(host=self.config.host, port=self.config.port)
        except TypeError:
            self._server_kwargs_supported = False
            logger.warning(
                "Installed viser version does not support explicit host/port kwargs. "
                "Falling back to default constructor."
            )
            server = viser_module.ViserServer()
        return server

    def _resolve_robot_urdf_path(self) -> str:
        """Resolve robot URDF path from the existing robot config."""
        asset_root = self.simulator.robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return os.path.join(asset_root, self.simulator.robot_config.asset.urdf_file)

    def _build_joint_mapping(self) -> None:
        """Build explicit URDF joint order -> simulator DOF mapping."""
        assert self.robot_viser is not None

        joint_limits = self.robot_viser.get_actuated_joint_limits()
        self.urdf_joint_order = list(joint_limits.keys())

        name_to_sim_index = {name: i for i, name in enumerate(self.simulator.dof_names)}
        mapped_indices: list[int] = []
        missing_names: list[str] = []
        for joint_name in self.urdf_joint_order:
            if joint_name not in name_to_sim_index:
                missing_names.append(joint_name)
                continue
            mapped_indices.append(name_to_sim_index[joint_name])

        if missing_names:
            if len(self.urdf_joint_order) == len(self.simulator.dof_names):
                logger.warning(
                    "Viser URDF joint names do not fully match simulator DOF names. "
                    "Falling back to positional mapping. Missing names: {}",
                    missing_names,
                )
                mapped_indices = list(range(len(self.simulator.dof_names)))
            else:
                raise KeyError(
                    "Unable to map Viser URDF joints to simulator DOFs. "
                    f"Missing joints: {missing_names}. Simulator DOFs: {self.simulator.dof_names}"
                )

        self.urdf_to_sim_indices = np.asarray(mapped_indices, dtype=int)
        logger.info(f"Viser joint mapping initialized for {len(self.urdf_to_sim_indices)} joints")

    def _setup_gui(self) -> None:
        """Create the browser GUI."""
        assert self.server is not None

        with self.server.gui.add_folder("Display"):
            self._show_meshes_cb = self.server.gui.add_checkbox(
                "Show meshes",
                initial_value=self.config.show_meshes,
            )
            self._camera_tracking_cb = self.server.gui.add_checkbox(
                "Camera tracking",
                initial_value=self.simulator.simulator_config.viewer.enable_tracking,
            )

        with self.server.gui.add_folder("Controls"):
            reset_btn = self.server.gui.add_button("Reset sim")
            gantry_raise_btn = self.server.gui.add_button("Gantry raise")
            gantry_lower_btn = self.server.gui.add_button("Gantry lower")
            gantry_toggle_btn = self.server.gui.add_button("Gantry toggle")
            gantry_force_btn = self.server.gui.add_button("Gantry force +")
            gantry_sign_btn = self.server.gui.add_button("Gantry sign")
            zero_cmd_btn = self.server.gui.add_button("Zero command")

        with self.server.gui.add_folder("Status"):
            self._status_handles["sim_time"] = self.server.gui.add_number(
                "Sim time",
                initial_value=0.0,
                min=0.0,
                max=1e9,
                step=0.001,
            )
            self._status_handles["viewer_fps"] = self.server.gui.add_number(
                "Viewer FPS",
                initial_value=0.0,
                min=0.0,
                max=1e6,
                step=0.1,
            )
            self._status_handles["target_fps"] = self.server.gui.add_number(
                "Target sim FPS",
                initial_value=float(self.simulator.simulator_config.sim.fps),
                min=0.0,
                max=1e6,
                step=1.0,
            )
            self._status_handles["gantry_length"] = self.server.gui.add_number(
                "Gantry length",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.01,
            )
            self._status_handles["gantry_force"] = self.server.gui.add_number(
                "Gantry force",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.1,
            )
            self._status_handles["cmd_vx"] = self.server.gui.add_number(
                "Cmd vx",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.01,
            )
            self._status_handles["cmd_vy"] = self.server.gui.add_number(
                "Cmd vy",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.01,
            )
            self._status_handles["cmd_yaw"] = self.server.gui.add_number(
                "Cmd yaw",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.01,
            )
            self._status_handles["cmd_walk"] = self.server.gui.add_number(
                "Cmd walk/stand",
                initial_value=0.0,
                min=-1e6,
                max=1e6,
                step=0.01,
            )

        @self._show_meshes_cb.on_update
        def _(_event: Any) -> None:
            if self.robot_viser is not None:
                self.robot_viser.show_visual = bool(self._show_meshes_cb.value)

        @self._camera_tracking_cb.on_update
        def _(_event: Any) -> None:
            if self._ui_update_in_progress:
                return
            if self.config.allow_controls:
                self.enqueue_command("set_camera_tracking", bool(self._camera_tracking_cb.value))

        @reset_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("reset")

        @gantry_raise_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("gantry_raise")

        @gantry_lower_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("gantry_lower")

        @gantry_toggle_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("gantry_toggle")

        @gantry_force_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("gantry_force_adjust")

        @gantry_sign_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("gantry_force_sign_toggle")

        @zero_cmd_btn.on_click
        def _(_event: Any) -> None:
            if self.config.allow_controls:
                self.enqueue_command("zero_commands")

    def _update_gantry_visuals(self, snapshot: MuJoCoViewerSnapshot) -> None:
        """Update simple browser-side gantry visuals."""
        if self._gantry_anchor_handle is None or self._gantry_line_handle is None:
            return

        if snapshot.gantry_point is None or snapshot.gantry_attachment_pos is None:
            self._gantry_anchor_handle.points = np.zeros((1, 3), dtype=np.float32)
            self._gantry_line_handle.points = np.zeros((1, 2, 3), dtype=np.float32)
            return

        self._gantry_anchor_handle.points = np.asarray(snapshot.gantry_point, dtype=np.float32).reshape(1, 3)
        self._gantry_line_handle.points = np.stack(
            [
                np.asarray(snapshot.gantry_point, dtype=np.float32),
                np.asarray(snapshot.gantry_attachment_pos, dtype=np.float32),
            ],
            axis=0,
        ).reshape(1, 2, 3)

    def _update_tracking(self, snapshot: MuJoCoViewerSnapshot) -> None:
        """Best-effort browser camera tracking by translating client cameras."""
        if self.server is None:
            return

        if self._camera_tracking_cb is not None:
            self._ui_update_in_progress = True
            self._camera_tracking_cb.value = snapshot.camera_tracking
            self._ui_update_in_progress = False

        if not snapshot.camera_tracking:
            self._last_tracking_target = None
            return

        if self._last_tracking_target is None:
            self._last_tracking_target = snapshot.root_pos.copy()
            return

        delta = snapshot.root_pos - self._last_tracking_target
        if np.linalg.norm(delta) < 1e-6:
            return

        get_clients = getattr(self.server, "get_clients", None)
        if not callable(get_clients):
            self._last_tracking_target = snapshot.root_pos.copy()
            return

        try:
            clients = get_clients()
        except Exception:  # pragma: no cover - best effort compatibility
            self._last_tracking_target = snapshot.root_pos.copy()
            return

        client_iterable = clients.values() if hasattr(clients, "values") else clients
        for client in client_iterable:
            camera = getattr(client, "camera", None)
            if camera is None or not hasattr(camera, "position") or not hasattr(camera, "look_at"):
                continue
            try:
                camera.position = tuple(np.asarray(camera.position, dtype=np.float64) + delta)
                camera.look_at = tuple(np.asarray(camera.look_at, dtype=np.float64) + delta)
            except Exception:  # pragma: no cover - best effort compatibility
                continue

        self._last_tracking_target = snapshot.root_pos.copy()

    def _update_status(self, snapshot: MuJoCoViewerSnapshot) -> None:
        """Update status widgets from the latest simulator snapshot."""
        now = time.time()
        if self._last_update_time is not None:
            dt = max(1e-6, now - self._last_update_time)
            instant_fps = 1.0 / dt
            if self._smoothed_viewer_fps <= 0.0:
                self._smoothed_viewer_fps = instant_fps
            else:
                self._smoothed_viewer_fps = 0.9 * self._smoothed_viewer_fps + 0.1 * instant_fps
        self._last_update_time = now

        self._status_handles["sim_time"].value = float(snapshot.sim_time)
        self._status_handles["viewer_fps"].value = float(self._smoothed_viewer_fps)
        self._status_handles["gantry_length"].value = float(snapshot.gantry_length or 0.0)
        self._status_handles["gantry_force"].value = float(snapshot.gantry_force or 0.0)

        if snapshot.commands is None or snapshot.commands.size == 0:
            self._status_handles["cmd_vx"].value = 0.0
            self._status_handles["cmd_vy"].value = 0.0
            self._status_handles["cmd_yaw"].value = 0.0
            self._status_handles["cmd_walk"].value = 0.0
            return

        self._status_handles["cmd_vx"].value = float(snapshot.commands[0])
        self._status_handles["cmd_vy"].value = float(snapshot.commands[1])
        self._status_handles["cmd_yaw"].value = float(snapshot.commands[3])
        self._status_handles["cmd_walk"].value = float(snapshot.commands[4])
