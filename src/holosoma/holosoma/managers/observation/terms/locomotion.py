"""Basic locomotion observation terms.

These functions compute individual observation components for legged locomotion tasks.
Each function mirrors the manager-based observation pipeline that replaced the legacy direct `_get_obs_*()` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from holosoma.utils.rotations import quat_rotate_inverse
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
    from holosoma.managers.command.terms.locomotion import LocomotionGait


def _base_quat(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: LeggedRobotLocomotionManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


def command_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded linear velocity (x, y).

    Returns:
        Tensor of shape [num_envs, 2]

    Equivalent to:
        env.command_manager.commands[:, :2]
    """
    return env.command_manager.commands[:, :2]


def command_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded angular velocity (yaw).

    Returns:
        Tensor of shape [num_envs, 1]

    Equivalent to:
        env.command_manager.commands[:, 2:3]
    """
    return env.command_manager.commands[:, 2:3]


def sin_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Sine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.sin(phase)


def cos_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Cosine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.cos(phase)


# ================================================================================================
# Thermal Observations
# ================================================================================================


def joint_temperature(env, ambient_temp: float = 30.0, clip_max: float = 300.0) -> torch.Tensor:
    """Per-DOF winding temperature.

    For joints without thermal models, returns ambient_temp.
    Requires env to have ``winding_temps`` and ``thermal_joint_indices``
    (set by ``LeggedRobotLocomotionThermalManager``).

    Returns:
        Tensor of shape [num_envs, num_dof]
    """
    output = torch.full((env.num_envs, env.num_dof), ambient_temp, device=env.device, dtype=torch.float32)
    winding_temps = getattr(env, "winding_temps", None)
    thermal_joint_indices = getattr(env, "thermal_joint_indices", None)
    if winding_temps is None or thermal_joint_indices is None or len(thermal_joint_indices) == 0:
        return output
    output[:, thermal_joint_indices] = torch.clamp(winding_temps, max=clip_max)
    return output


def joint_case_temperature(env, ambient_temp: float = 30.0, clip_max: float = 300.0) -> torch.Tensor:
    """Per-DOF case (housing) temperature.

    Returns:
        Tensor of shape [num_envs, num_dof]
    """
    output = torch.full((env.num_envs, env.num_dof), ambient_temp, device=env.device, dtype=torch.float32)
    case_temps = getattr(env, "case_temps", None)
    thermal_joint_indices = getattr(env, "thermal_joint_indices", None)
    if case_temps is None or thermal_joint_indices is None or len(thermal_joint_indices) == 0:
        return output
    output[:, thermal_joint_indices] = torch.clamp(case_temps, max=clip_max)
    return output


def joint_temperature_rate(env, clip_range: float = 10.0) -> torch.Tensor:
    """Per-DOF winding temperature rate of change (deg C/s).

    Clamped to ``[-clip_range, clip_range]`` to prevent observation explosion.

    Returns:
        Tensor of shape [num_envs, num_dof]
    """
    output = torch.zeros((env.num_envs, env.num_dof), device=env.device, dtype=torch.float32)
    winding_temps = getattr(env, "winding_temps", None)
    prev_winding_temps = getattr(env, "prev_winding_temps", None)
    thermal_joint_indices = getattr(env, "thermal_joint_indices", None)
    if winding_temps is None or prev_winding_temps is None or thermal_joint_indices is None:
        return output
    rate = torch.clamp((winding_temps - prev_winding_temps) / env.dt, min=-clip_range, max=clip_range)
    output[:, thermal_joint_indices] = rate
    return output


def joint_temperature_headroom(env, max_temp: float = 110.0, ambient_temp: float = 30.0) -> torch.Tensor:
    """Per-DOF distance to maximum temperature limit.

    Returns:
        Tensor of shape [num_envs, num_dof]
    """
    output = torch.full(
        (env.num_envs, env.num_dof), max_temp - ambient_temp, device=env.device, dtype=torch.float32
    )
    winding_temps = getattr(env, "winding_temps", None)
    thermal_joint_indices = getattr(env, "thermal_joint_indices", None)
    if winding_temps is None or thermal_joint_indices is None or len(thermal_joint_indices) == 0:
        return output
    output[:, thermal_joint_indices] = max_temp - winding_temps
    return output


def joint_temperature_initial(env, ambient_temp: float = 30.0) -> torch.Tensor:
    """Per-DOF initial winding temperature at episode start.

    Helps the policy understand its thermal budget for the episode.

    Returns:
        Tensor of shape [num_envs, num_dof]
    """
    output = torch.full((env.num_envs, env.num_dof), ambient_temp, device=env.device, dtype=torch.float32)
    initial_winding_temps = getattr(env, "initial_winding_temps", None)
    thermal_joint_indices = getattr(env, "thermal_joint_indices", None)
    if initial_winding_temps is None or thermal_joint_indices is None or len(thermal_joint_indices) == 0:
        return output
    output[:, thermal_joint_indices] = initial_winding_temps
    return output
