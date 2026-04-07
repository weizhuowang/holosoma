from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.utils.safe_torch_import import torch

_DEFAULT_THERMAL_REPO = Path(
    os.environ.get("HOLOSOMA_THERMAL_REPO", str(Path.home() / "Documents" / "gits" / "toy_thermal_ig"))
).expanduser()
if _DEFAULT_THERMAL_REPO.exists() and str(_DEFAULT_THERMAL_REPO) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_THERMAL_REPO))

_THERMAL_IMPORT_ERROR: Exception | None = None
try:
    from simulator import ThermalSimulator
except Exception as exc:  # pragma: no cover - defensive fallback
    ThermalSimulator = None
    _THERMAL_IMPORT_ERROR = exc


class LeggedRobotLocomotionThermalManager(LeggedRobotLocomotionManager):
    """Locomotion environment with thermal motor simulation.

    Responsibilities (kept thin):
    - Initialize ``ThermalSimulator`` from ``toy_thermal_ig``
    - Update thermal state each physics step
    - Expose ``apply_thermal_reset()`` for the randomization manager
    - Log thermal metrics

    Observation, reward, and termination terms live in
    ``managers/{observation,reward,termination}/terms/locomotion.py``.
    """

    DEFAULT_THERMAL_PARAMS_PATH = (
        Path.home()
        / "Documents"
        / "gits"
        / "toy_thermal_ig"
        / "data"
        / "symmetric_batch_sysid_multifile_20250807_200807"
        / "thermal_params.yaml"
    )

    def __init__(self, tyro_config, *, device):
        self.thermal_ambient_temp = 30.0
        super().__init__(tyro_config, device=device)
        self._init_thermal_simulator()

    def _resolve_thermal_params_path(self) -> Path:
        value = os.environ.get("HOLOSOMA_THERMAL_PARAMS_PATH", str(self.DEFAULT_THERMAL_PARAMS_PATH))
        return Path(value).expanduser()

    def _init_thermal_simulator(self) -> None:
        if ThermalSimulator is None:
            raise ImportError(
                "ThermalSimulator import failed. Set HOLOSOMA_THERMAL_REPO to your toy_thermal_ig path."
            ) from _THERMAL_IMPORT_ERROR

        thermal_params_path = self._resolve_thermal_params_path()
        self.thermal_simulator = ThermalSimulator(
            thermal_params_path=thermal_params_path,
            num_envs=self.num_envs,
            dof_names=self.dof_names,
            device=self.device,
            ambient_temp=self.thermal_ambient_temp,
            enable_parallel_mechanisms=True,
            fast_dynamics=True,
        )

        self.thermal_joint_names = self.thermal_simulator.thermal_joint_names
        self.thermal_joint_indices = self.thermal_simulator.thermal_joint_indices

        self.winding_temps = self.thermal_simulator.winding_temps
        self.case_temps = self.thermal_simulator.case_temps
        self.prev_winding_temps = self.winding_temps.clone()
        self.initial_winding_temps = self.winding_temps.clone()
        self.torque_history = self.thermal_simulator.torque_history
        self.motor_torques_AB = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        self._sync_torque_reference()

    def _sync_torque_reference(self) -> None:
        try:
            joint_term = self.action_manager.get_term("joint_control")
            self.torques = joint_term.torques
        except Exception:
            self.torques = torch.zeros((self.num_envs, self.num_dof), device=self.device)

    # ------------------------------------------------------------------
    # Fallback thermal reset on episode boundaries
    # ------------------------------------------------------------------

    def _reset_tasks_callback(self, env_ids):
        """Reset thermal state to ambient for environments that are resetting.

        This guarantees temperatures are reset even when the ``thermal_reset``
        randomization term is not present in the config.  When the term *is*
        present it will run afterwards (in ``randomization_manager.reset``)
        and overwrite with its own randomized temperatures.
        """
        super()._reset_tasks_callback(env_ids)
        if not hasattr(self, "thermal_simulator"):
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return
        self.thermal_simulator.reset(env_ids=env_ids_tensor)
        self.winding_temps = self.thermal_simulator.winding_temps
        self.case_temps = self.thermal_simulator.case_temps
        self.torque_history = self.thermal_simulator.torque_history
        self.prev_winding_temps[env_ids_tensor] = self.winding_temps[env_ids_tensor].clone()
        self.initial_winding_temps[env_ids_tensor] = self.winding_temps[env_ids_tensor].clone()

    # ------------------------------------------------------------------
    # Per-step thermal update
    # ------------------------------------------------------------------

    def _pre_compute_observations_callback(self):
        self._update_thermal_simulation()
        super()._pre_compute_observations_callback()

    def _update_thermal_simulation(self) -> None:
        if not hasattr(self, "thermal_simulator"):
            return

        self._sync_torque_reference()
        if self.torques.shape[1] != self.num_dof:
            return

        self.prev_winding_temps.copy_(self.winding_temps)
        winding_temps, case_temps = self.thermal_simulator.step(
            dt=self.dt,
            joint_torques=self.torques,
            joint_positions=self.simulator.dof_pos,
        )
        self.winding_temps = winding_temps
        self.case_temps = case_temps
        self.motor_torques_AB = self.thermal_simulator.compute_motor_torques(self.torques, self.simulator.dof_pos)

    # ------------------------------------------------------------------
    # Thermal reset (called by randomization term ``thermal_reset``)
    # ------------------------------------------------------------------

    def apply_thermal_reset(
        self,
        env_ids,
        *,
        randomize_temp: bool = True,
        winding_temp_range: Sequence[float] = (30.0, 50.0),
        case_temp_range: Sequence[float] = (30.0, 50.0),
        hot_knee_prob: float = 0.4,
        hot_hip_prob: float = 0.4,
        hot_joint_temp: float = 90.0,
    ) -> None:
        if not hasattr(self, "thermal_simulator"):
            return

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        initial_winding_temps = None
        initial_case_temps = None
        if randomize_temp:
            initial_winding_temps, initial_case_temps = self._sample_initial_thermal_temps(
                num_envs=env_ids_tensor.numel(),
                winding_temp_range=winding_temp_range,
                case_temp_range=case_temp_range,
                hot_knee_prob=hot_knee_prob,
                hot_hip_prob=hot_hip_prob,
                hot_joint_temp=hot_joint_temp,
            )

        self.thermal_simulator.reset(
            env_ids=env_ids_tensor,
            initial_winding_temps=initial_winding_temps,
            initial_case_temps=initial_case_temps,
        )

        self.winding_temps = self.thermal_simulator.winding_temps
        self.case_temps = self.thermal_simulator.case_temps
        self.torque_history = self.thermal_simulator.torque_history
        self.prev_winding_temps[env_ids_tensor] = self.winding_temps[env_ids_tensor].clone()
        self.initial_winding_temps[env_ids_tensor] = self.winding_temps[env_ids_tensor].clone()

    def _sample_initial_thermal_temps(
        self,
        *,
        num_envs: int,
        winding_temp_range: Sequence[float],
        case_temp_range: Sequence[float],
        hot_knee_prob: float,
        hot_hip_prob: float,
        hot_joint_temp: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_thermal_joints = len(self.thermal_joint_names)
        case_low, case_high = float(case_temp_range[0]), float(case_temp_range[1])
        wind_low, wind_high = float(winding_temp_range[0]), float(winding_temp_range[1])

        initial_case_temps = torch.rand(num_envs, num_thermal_joints, device=self.device)
        initial_case_temps = initial_case_temps * (case_high - case_low) + case_low

        initial_winding_temps = torch.rand(num_envs, num_thermal_joints, device=self.device)
        initial_winding_temps = initial_winding_temps * (wind_high - wind_low) + wind_low

        thermal_joint_names = self.thermal_joint_names
        knee_indices = [i for i, name in enumerate(thermal_joint_names) if "knee" in name]
        hip_pitch_indices = [i for i, name in enumerate(thermal_joint_names) if "hip_pitch" in name]

        rand_group = torch.rand(num_envs, device=self.device)
        hot_knee_mask = rand_group < hot_knee_prob
        hot_hip_mask = (rand_group >= hot_knee_prob) & (rand_group < hot_knee_prob + hot_hip_prob)

        if hot_knee_mask.any():
            initial_winding_temps[hot_knee_mask, :] = self.thermal_ambient_temp
            initial_case_temps[hot_knee_mask, :] = self.thermal_ambient_temp
            for idx in knee_indices:
                initial_winding_temps[hot_knee_mask, idx] = hot_joint_temp
                initial_case_temps[hot_knee_mask, idx] = hot_joint_temp

        if hot_hip_mask.any():
            initial_winding_temps[hot_hip_mask, :] = self.thermal_ambient_temp
            initial_case_temps[hot_hip_mask, :] = self.thermal_ambient_temp
            for idx in hip_pitch_indices:
                initial_winding_temps[hot_hip_mask, idx] = hot_joint_temp
                initial_case_temps[hot_hip_mask, idx] = hot_joint_temp

        initial_winding_temps = torch.max(initial_winding_temps, initial_case_temps)
        return initial_winding_temps, initial_case_temps

    # ------------------------------------------------------------------
    # Logging & monitoring
    # ------------------------------------------------------------------

    def get_thermal_info(self):
        return self.thermal_simulator.get_thermal_info()

    def _update_log_dict(self):
        super()._update_log_dict()

        if not hasattr(self, "winding_temps"):
            return

        per_joint_penalty = torch.clamp((self.winding_temps - 60.0) / 50.0, min=0.0, max=1.5)
        max_penalty = per_joint_penalty.max(dim=1).values
        mean_penalty = per_joint_penalty.mean(dim=1)
        sum_penalty = torch.clamp(per_joint_penalty.sum(dim=1), max=7.0)

        self.log_dict["temp_penalty_max"] = max_penalty.mean()
        self.log_dict["temp_penalty_mean"] = mean_penalty.mean()
        self.log_dict["temp_penalty_sum7"] = sum_penalty.mean()
        self.log_dict["max_winding_temp"] = self.winding_temps.max(dim=1).values.mean()
        self.log_dict["mean_winding_temp"] = self.winding_temps.mean()
