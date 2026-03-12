# Holosoma 项目笔记

## 架构

- Manager-Based 架构：ObservationManager, RewardManager, TerminationManager, RandomizationManager, CommandManager, CurriculumManager, TerrainManager, ActionManager
- Term 函数模式：`def term_func(env, **params) -> Tensor`，通过字符串路径 `"module.path:function_name"` 注册
- 配置系统：Tyro（Python dataclass + CLI），`ExperimentConfig`，用 `replace()` 创建变体
- 模拟器：isaacgym, isaacsim, mujoco, mjwarp

## 关键路径

- 训练入口：`src/holosoma/holosoma/train_agent.py`
- 环境类：`src/holosoma/holosoma/envs/locomotion/`
- Manager terms：`src/holosoma/holosoma/managers/{observation,reward,termination,randomization}/terms/locomotion.py`
- 算法：`src/holosoma/holosoma/agents/{ppo,fast_sac}/`
- 实验配置：`src/holosoma/holosoma/config_values/loco/g1/experiment.py`
- 对称性/镜像：`src/holosoma/holosoma/agents/modules/augmentation_utils.py`

## 训练命令

```bash
source scripts/source_isaacgym_setup.sh

# 单卡
python src/holosoma/holosoma/train_agent.py exp:g1-29dof-thermal simulator:isaacsim logger:wandb --training.seed 1

# 多卡（num-envs 是总数，自动按 GPU 数量分）
torchrun --nproc_per_node=4 --master_port=29501 src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-thermal simulator:isaacsim logger:wandb --training.num-envs 16384

# 覆盖迭代次数
--algo.config.num-learning-iterations 5000
```

## Conda 环境

- 路径：`~/.holosoma_deps/miniconda3/envs/hssim`
- Python：`~/.holosoma_deps/miniconda3/envs/hssim/bin/python`

## Thermal 系统

- 环境类：`envs/locomotion/locomotion_thermal_manager.py`（LeggedRobotLocomotionThermalManager）
- 依赖 `toy_thermal_ig`（通过 sys.path 加载，非 pip install）
  - 默认路径：`~/Documents/gits/toy_thermal_ig`
  - 覆盖：`HOLOSOMA_THERMAL_REPO` 环境变量
  - 参数文件：`HOLOSOMA_THERMAL_PARAMS_PATH` 环境变量
- 5 个观测 term、1 个奖励 term、1 个终止 term、1 个 randomization reset term
- 所有 thermal term 用 `getattr(env, ...)` 安全访问 — 非 thermal 环境调用时无副作用
- 设计文档：`thermal_port_plan.md`

## WandB 查询

用 hssim 环境的 Python 查询训练指标：

```bash
~/.holosoma_deps/miniconda3/envs/hssim/bin/python -c "
import wandb, os
os.environ['WANDB_BASE_URL'] = 'https://far.wandb.io'
api = wandb.Api()
# 查询代码
"
```

### 常用查询

- 列出 runs：`api.runs('项目名', per_page=20)` → `r.id, r.name, r.state`
- 查指标：`api.run('项目名/RUN_ID').history(keys=[...], samples=30)`
- 查所有 key：`api.run('项目名/RUN_ID').history(samples=1).columns`

### 项目名

- `g1-manager-thermal` — thermal 实验
- `g1-manager-thermal-baseline` — baseline 对照

### 常用指标

- 训练：`Train/mean_reward`, `Train/mean_episode_length`
- Thermal：`Env/max_winding_temp`, `Env/mean_winding_temp`, `Env/temp_penalty_max`, `Episode/rew_penalty_joint_temperature`
- Loss：`Loss/value_loss`, `Loss/surrogate_loss`, `Loss/entropy`
- 性能：`Perf/total_fps`

## Remote Experiment Manager

- Snapshot 目录：`~/Documents/gits/remote_exp_manager/data/snapshots/{job_id}/src/`
- 训练 log 在 snapshot 目录里，不在原 repo：`snapshots/{job_id}/src/logs/`
- 同时跑多个 torchrun 需要不同的 `--master_port` 和 `CUDA_VISIBLE_DEVICES`
