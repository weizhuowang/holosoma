"""Viewer configuration types for holosoma simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from holosoma.config_types.video import CameraConfig


@dataclass(frozen=True)
class ViserViewerConfig:
    """Configuration for browser-based Viser visualization."""

    host: str = "127.0.0.1"
    """Host interface to bind the Viser server to."""

    port: int = 8080
    """TCP port for the Viser server."""

    show_grid: bool = True
    """Whether to show a ground grid in the browser viewer."""

    show_meshes: bool = True
    """Whether robot meshes should be visible on first load."""

    show_gantry: bool = True
    """Whether to render simple gantry debug visuals in the browser viewer."""

    allow_controls: bool = True
    """Whether browser clients can issue simulator control commands."""


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for interactive viewer camera behavior.

    This is separate from video recording camera config to allow independent
    configuration of viewer and recording cameras.

    Parameters
    ----------
    enable_tracking : bool, default=False
        Enable camera tracking in the interactive viewer.
    camera : CameraConfig | None, default=None
        Camera configuration (FixedCameraConfig | SphericalCameraConfig | CartesianCameraConfig).
        Must be provided if enable_tracking=True.

    Examples
    --------
    Enable viewer camera with default spherical tracking:

    >>> from holosoma.config_types.video import SphericalCameraConfig
    >>> viewer = ViewerConfig(
    ...     enable_tracking=True,
    ...     camera=SphericalCameraConfig(),  # Uses all defaults
    ... )

    Enable viewer camera with custom settings:

    >>> viewer = ViewerConfig(
    ...     enable_tracking=True,
    ...     camera=SphericalCameraConfig(
    ...         distance=5.0,
    ...         azimuth=90.0,
    ...         elevation=30.0,
    ...         smoothing=0.9,
    ...         tracking_body_name="Trunk",
    ...     ),
    ... )
    """

    backend: Literal["native", "viser", "both", "none"] = "native"
    """Viewer backend selection.

    - ``native``: simulator-native desktop viewer only
    - ``viser``: browser-based Viser viewer only
    - ``both``: native desktop viewer + browser viewer
    - ``none``: no viewer
    """

    enable_tracking: bool = False
    """Enable camera tracking in the interactive viewer."""

    camera: CameraConfig | None = None
    """Camera configuration. Must be provided if enable_tracking=True."""

    viser: ViserViewerConfig = field(default_factory=ViserViewerConfig)
    """Browser-based viewer configuration."""

    def uses_native(self) -> bool:
        """Whether the native simulator viewer should be active."""
        return self.backend in ("native", "both")

    def uses_viser(self) -> bool:
        """Whether the browser-based Viser viewer should be active."""
        return self.backend in ("viser", "both")
