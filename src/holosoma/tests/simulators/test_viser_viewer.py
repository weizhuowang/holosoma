"""Tests for MuJoCo Viser viewer integration helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from holosoma.config_types.viewer import ViewerConfig
from holosoma.simulator.mujoco.viser_viewer import MuJoCoViserViewer
from holosoma.utils.sim_utils import DirectSimulation


class TestViewerConfig:
    """Smoke tests for viewer backend selection."""

    @pytest.mark.parametrize(
        ("backend", "uses_native", "uses_viser"),
        [
            ("native", True, False),
            ("viser", False, True),
            ("both", True, True),
            ("none", False, False),
        ],
    )
    def test_backend_helpers(self, backend: str, uses_native: bool, uses_viser: bool) -> None:
        """Backend helper methods should match the configured backend."""
        config = ViewerConfig(backend=backend)
        assert config.uses_native() is uses_native
        assert config.uses_viser() is uses_viser


class TestMuJoCoViserViewer:
    """Tests for the browser viewer command queue."""

    def test_command_queue_preserves_order(self) -> None:
        """Queued commands should drain in FIFO order."""
        simulator = SimpleNamespace(
            simulator_config=SimpleNamespace(
                viewer=SimpleNamespace(
                    viser=SimpleNamespace(
                        host="127.0.0.1",
                        port=8080,
                        show_grid=True,
                        show_meshes=True,
                        show_gantry=True,
                        allow_controls=True,
                    )
                )
            )
        )

        viewer = MuJoCoViserViewer(simulator)
        viewer.enqueue_command("reset")
        viewer.enqueue_command("set_camera_tracking", True)

        drained = viewer.drain_pending_commands()

        assert [command.name for command in drained] == ["reset", "set_camera_tracking"]
        assert drained[1].value is True
        assert viewer.drain_pending_commands() == []


class TestDirectSimulationViewerSetup:
    """Tests for headless-aware direct simulation initialization."""

    def test_initialize_respects_training_headless_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DirectSimulation should forward the configured headless value."""
        simulator = Mock()
        simulator.should_setup_viewer.return_value = True
        simulator.video_recorder = None
        env = SimpleNamespace(sim=simulator)
        config = SimpleNamespace(
            training=SimpleNamespace(headless=True),
            robot=SimpleNamespace(
                init_state=SimpleNamespace(
                    pos=[0.0, 0.0, 0.0],
                    rot=[1.0, 0.0, 0.0, 0.0],
                    lin_vel=[0.0, 0.0, 0.0],
                    ang_vel=[0.0, 0.0, 0.0],
                )
            ),
        )

        direct_sim = DirectSimulation(config=config, env=env, device="cpu", simulation_app=None)
        monkeypatch.setattr(direct_sim, "_create_base_init_state", lambda: torch.zeros(13))

        direct_sim.initialize()

        simulator.set_headless.assert_called_once_with(True)
        simulator.should_setup_viewer.assert_called_once_with()
        simulator.setup_viewer.assert_called_once_with()

    def test_initialize_skips_viewer_when_backend_not_requested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DirectSimulation should not call setup_viewer when the simulator opts out."""
        simulator = Mock()
        simulator.should_setup_viewer.return_value = False
        simulator.video_recorder = None
        env = SimpleNamespace(sim=simulator)
        config = SimpleNamespace(
            training=SimpleNamespace(headless=True),
            robot=SimpleNamespace(
                init_state=SimpleNamespace(
                    pos=[0.0, 0.0, 0.0],
                    rot=[1.0, 0.0, 0.0, 0.0],
                    lin_vel=[0.0, 0.0, 0.0],
                    ang_vel=[0.0, 0.0, 0.0],
                )
            ),
        )

        direct_sim = DirectSimulation(config=config, env=env, device="cpu", simulation_app=None)
        monkeypatch.setattr(direct_sim, "_create_base_init_state", lambda: torch.zeros(13))

        direct_sim.initialize()

        simulator.set_headless.assert_called_once_with(True)
        simulator.should_setup_viewer.assert_called_once_with()
        simulator.setup_viewer.assert_not_called()
