from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

_OBS_TERMS = "holosoma.managers.observation.terms.locomotion"
_REW_TERMS = "holosoma.managers.reward.terms.locomotion"
_TERM_TERMS = "holosoma.managers.termination.terms.locomotion"
_RAND_TERMS = "holosoma.managers.randomization.terms.locomotion"

g1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_thermal_manager.LeggedRobotLocomotionThermalManager",
    training=TrainingConfig(project="g1-manager-thermal", name="g1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=2000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=replace(
        randomization.g1_29dof_randomization,
        reset_terms={
            **randomization.g1_29dof_randomization.reset_terms,
            "thermal_reset": RandomizationTermCfg(
                func=f"{_RAND_TERMS}:thermal_reset",
                params={
                    "winding_temp_range": [30.0, 50.0],
                    "case_temp_range": [30.0, 50.0],
                    "hot_knee_prob": 0.4,
                    "hot_hip_prob": 0.4,
                    "hot_joint_temp": 90.0,
                },
            ),
        },
    ),
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=reward.g1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

g1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="g1-manager-thermal-baseline", name="g1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.g1_29dof_loco_single_wolinvel,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_termination,
    randomization=randomization.g1_29dof_randomization,
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum_fast_sac,
    reward=reward.g1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

g1_29dof_thermal = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_thermal_manager.LeggedRobotLocomotionThermalManager",
    training=TrainingConfig(project="g1-manager-thermal", name="g1_29dof_thermal_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=2000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=replace(
        observation.g1_29dof_loco_single_wolinvel,
        groups={
            **observation.g1_29dof_loco_single_wolinvel.groups,
            "actor_obs": replace(
                observation.g1_29dof_loco_single_wolinvel.groups["actor_obs"],
                terms={
                    **observation.g1_29dof_loco_single_wolinvel.groups["actor_obs"].terms,
                    "joint_temperature": ObsTermCfg(
                        func=f"{_OBS_TERMS}:joint_temperature",
                        scale=0.003,
                        noise=0.0,
                    ),
                },
            ),
            "critic_obs": replace(
                observation.g1_29dof_loco_single_wolinvel.groups["critic_obs"],
                terms={
                    **observation.g1_29dof_loco_single_wolinvel.groups["critic_obs"].terms,
                    "joint_temperature": ObsTermCfg(
                        func=f"{_OBS_TERMS}:joint_temperature",
                        scale=0.003,
                        noise=0.0,
                    ),
                },
            ),
        },
    ),
    action=action.g1_29dof_joint_pos,
    termination=replace(
        termination.g1_29dof_termination,
        terms={
            **termination.g1_29dof_termination.terms,
            "temperature": TerminationTermCfg(
                func=f"{_TERM_TERMS}:temperature_exceeded",
                params={"threshold": 110.0},
            ),
        },
    ),
    randomization=replace(
        randomization.g1_29dof_randomization,
        reset_terms={
            **randomization.g1_29dof_randomization.reset_terms,
            "thermal_reset": RandomizationTermCfg(
                func=f"{_RAND_TERMS}:thermal_reset",
                params={
                    "winding_temp_range": [30.0, 50.0],
                    "case_temp_range": [30.0, 50.0],
                    "hot_knee_prob": 0.4,
                    "hot_hip_prob": 0.4,
                    "hot_joint_temp": 90.0,
                },
            ),
        },
    ),
    command=command.g1_29dof_command,
    curriculum=curriculum.g1_29dof_curriculum,
    reward=replace(
        reward.g1_29dof_loco,
        terms={
            **reward.g1_29dof_loco.terms,
            "penalty_joint_temperature": RewardTermCfg(
                func=f"{_REW_TERMS}:penalty_joint_temperature",
                weight=-1.0,
                params={"start_temp": 60.0, "ramp_temp": 50.0, "overall_scale": 1.5},
                tags=["penalty_curriculum"],
            ),
        },
    ),
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

__all__ = ["g1_29dof", "g1_29dof_fast_sac", "g1_29dof_thermal"]
