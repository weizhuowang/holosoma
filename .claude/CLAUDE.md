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
- 状态查询：`python3 ~/Documents/gits/remote_exp_manager/remexp_cli.py status --json`
- 排队功能：`--wait-for-job <job_id>` 让新 job 等指定 job 完成后再启动（snapshot 会立即创建）
- 只支持 1 卡或 2 卡实验，8 卡机器用并行跑更多实验而非单实验多卡

## 实验工作流

可用的 skill 和 agent：
- `/launch-exp` — Skill：查 GPU、规划分配、确认后批量起实验（spawn run-exp agents）
- `/run-exp` — Agent：提交单个实验到 rem + 监控到完成，返回原始状态（不分析）
- `/check-exp` — Skill：快速查当前状态（GPU + 跑着的 job + WandB 最新指标）
- `/report-exp` — Agent：实验跑完后生成详细报告（读 TensorBoard、画图、统计摘要）

标准流程：
1. `/launch-exp` 起实验 → 自动 spawn 后台 run-exp agent
2. 跑的过程中用 `/check-exp` 查进度（或 `/loop 10m /check-exp` 自动巡检）
3. run-exp agent 完成后通知你 → 更新 experiments_log.md
4. 需要深度分析时 spawn `/report-exp`（读图理解训练曲线）

## 实验记录

**每次启动或完成实验时，必须更新 `.claude/experiments_log.md`。**
- 启动时追加 `[LAUNCHED]` 条目（config、GPU、seed、目标）
- 完成时追加 `[DONE]` 条目（status、reward、takeaway）
- 即使不走 `/launch-exp` skill，手动跑实验也要记录
