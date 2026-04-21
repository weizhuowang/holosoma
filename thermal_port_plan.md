# Thermal Port 计划：FAR-FALCON → Holosoma（v2）

> 上次更新：2026-04-07
>
> - **Phase 1（架构搬迁）**：✅ 完成
> - **Phase 2（行为对齐）**：✅ 不需要（Holosoma 有意偏离 FAR-FALCON 默认值）
> - **Phase 3（功能补全）**：⚠️ 部分完成（3.1 ✅, 3.2 ⬜, 3.3 ✅, 3.4 ⬜）
> - **Phase 4（生态扩展）**：⚠️ 部分完成（4.1 ✅, 其余 ⬜）

## 背景

FAR-FALCON (HumanoidVerse) 中的 thermal 功能需要 port 到 Holosoma 的 Manager-Based 架构。
`dev_thermal` 分支从 commit `7ee380f`（初步 port）经过多次迭代到 `e1e7992`（当前 HEAD），
架构搬迁已完成，但行为与 FAR-FALCON 存在多处分歧，且部分功能尚未迁移。

---

## Phase 1：架构搬迁（✅ 已完成）

### 完成的工作

将 thermal term 函数从环境类搬到 `managers/xxx/terms/locomotion.py`，符合 Holosoma 的 Manager-Based 架构。

| 文件 | 变更 | 状态 |
|------|------|------|
| `managers/observation/terms/locomotion.py` | +5 thermal 观测函数 | ✅ |
| `managers/reward/terms/locomotion.py` | +1 温度惩罚函数 | ✅ |
| `managers/termination/terms/locomotion.py` | +1 温度超限终止函数 | ✅ |
| `managers/randomization/terms/locomotion.py` | +1 thermal reset term | ✅ |
| `envs/locomotion/locomotion_thermal_manager.py` | 精简环境类，保留 simulator 初始化 + 更新 + reset + logging | ✅ |
| `config_values/loco/g1/experiment.py` | `g1_29dof_thermal` 实验配置 | ✅ |
| `agents/modules/augmentation_utils.py` | 已有 `mirror_obs_joint_temperature`，不用改 | ✅ |

### 设计原则

- term 函数放 `managers/xxx/terms/`，不放环境类文件
- 环境类只负责 ThermalSimulator 初始化 + 每步温度更新 + reset 底层方法 + logging
- thermal reset 走 RandomizationManager 的 `reset_terms`（环境类 `_reset_tasks_callback` 保留 fallback 到 ambient）
- 所有 magic number 通过 `params` 配置传入
- 所有 thermal obs/reward/termination 用 `getattr(env, ...)` 安全访问，非 thermal 环境调用时返回默认值

### 相对 FAR-FALCON 的有意改动

1. **Logging 公式修复**：FAR-FALCON `_update_log_dict` 用指数公式计算 `temp_penalty_*`，与 reward 实际用的线性公式不一致。Holosoma port 时统一改为线性公式 `clamp((T - 60) / 50, 0, 1.5)`。
2. **`_reset_tasks_callback` fallback**：即使不配 `thermal_reset` randomization term，环境类也会在 reset 时把温度归零到 ambient，避免 episode 间漏 reset。
3. **`fast_dynamics=True`**：commit `e1e7992` 启用了 `ThermalSimulator` 的快速动力学模式（FAR-FALCON 默认 `False`）。温度上升更快，训练中 thermal 信号更强。

---

## Phase 2：行为对齐（✅ 不需要）

Holosoma 的 thermal 训练参数有意偏离 FAR-FALCON 默认值，形成一套自洽的配置：

| 差异点 | FAR-FALCON | Holosoma | 结论 |
|---|---|---|---|
| `max_episode_length_s` | 100s | 20s（默认） | `fast_dynamics=True` 加速温度累积，20s 够 |
| `penalty_curriculum.degree` | 0.0001 | 0.00025 | Holosoma 自己的调参，不改 |
| `penalty_curriculum.level_up_threshold` | 850 | 750 | 同上 |
| `terminate_by_temperature` | 关 | 开（threshold=110） | Holosoma 有意加强，正确 |
| `overall_scale` | 0.3 | 1.5 | 配合 fast_dynamics + 短 episode，合理 |
| `thermal_params_path` | YAML 配置 | 环境变量 | 只有一套 params，够用 |

---

## Phase 3：功能补全（⬜ 待开始）

### 3.1 Command Curriculum

**来源**：FAR-FALCON `locomotion_thermal.py:87-126`

FAR-FALCON 在训练过程中将速度命令范围从 ±1.0 m/s 线性 ramp 到 ±4.0 m/s（迭代 2000-10000）。
高速命令才能真正暴露电机过热问题——±1 m/s 下温度上不去。

**Holosoma 现状**：
- `config_values/loco/g1/command.py:21-25`：`lin_vel_x: [-1.0, 1.0]` 写死
- `managers/command/terms/locomotion.py` 的 `LocomotionCommand` 类无 ramp 机制

**实现方案**：
在 `managers/curriculum/terms/locomotion.py` 加一个 `CommandRangeCurriculum` class（参考 `PenaltyCurriculum:105` 写法），在 `step()` 阶段修改 `command_manager.get_term("locomotion_command").command_ranges`。

```python
class CommandRangeCurriculum(CurriculumTermBase):
    """基于迭代次数线性 ramp 速度命令范围。"""
    # params: start_iteration, end_iteration, final_lin_vel_x_range
    # 每步根据 common_step_counter 计算 factor，更新 command_ranges
```

配置：
```python
"command_curriculum": CurriculumTermCfg(
    func="holosoma.managers.curriculum.terms.locomotion:CommandRangeCurriculum",
    params={
        "start_iteration": 2000,
        "end_iteration": 10000,
        "final_lin_vel_x_range": 4.0,
    },
),
```

### 3.2 启用其余 thermal observations

当前 `g1_29dof_thermal` 只接入了 `joint_temperature`（1/5）。FAR-FALCON 的 `static_squat_thermal` 已经启用了 4 个（winding, case, rate, initial）并调好了 obs_scales。

5 个 obs 函数已全部实现，需要在配置里按需启用：

| 观测 | 函数 | scale | 说明 |
|---|---|---|---|
| `joint_temperature` | `...:joint_temperature` | 0.003 | ✅ 已启用 |
| `joint_case_temperature` | `...:joint_case_temperature` | 0.004 | ⬜ 函数已实现 |
| `joint_temperature_rate` | `...:joint_temperature_rate` | 0.1 | ⬜ 函数已实现 |
| `joint_temperature_headroom` | `...:joint_temperature_headroom` | 0.01 | ⬜ 函数已实现 |
| `joint_temperature_initial` | `...:joint_temperature_initial` | 0.004 | ⬜ 函数已实现 |

obs_scales 来自 FAR-FALCON `static_squat_g1_29dof_thermal.yaml:47-51`。可以先在 critic_obs 加所有 thermal obs（给 value function 更多信息），actor_obs 保持只用 joint_temperature。

注意：启用更多 obs 会增大 obs 维度，需要同步更新 `augmentation_utils.py` 的 mirror mapping。当前 `mirror_obs_joint_temperature` 只处理单个 joint_temperature obs。

### 3.3 Termination 分类统计

**来源**：FAR-FALCON `locomotion.py:29-39` + `locomotion_thermal.py:49-50,525-527`

FAR-FALCON 维护 `termination_counts` 字典，按原因（timeout/contact/gravity/low_height/temperature 等）分别统计终止次数，训练时打印分布。帮助诊断"robot 是摔了还是过热了"。

Holosoma 没有这个基础设施。

**实现方案**：在 `TerminationManager` 中加一个 per-term counter，每次 reset 时统计哪个 term 触发了终止。或者在 `_update_log_dict` 中添加 per-term 终止率到 WandB。

### 3.4 多阶段训练配置

**来源**：FAR-FALCON `static_squat_g1_29dof_stage1.yaml` / `_stage2.yaml`

```python
# Stage 1: 学基础行走，关闭温度惩罚和终止
g1_29dof_thermal_stage1 = replace(g1_29dof_thermal,
    training=replace(..., name="g1_29dof_thermal_stage1"),
    reward=replace(..., "penalty_joint_temperature": RewardTermCfg(weight=0.0, ...)),
    termination=replace(...),  # 去掉 temperature_exceeded term
)

# Stage 2: 打开温度惩罚和终止，加载 stage1 checkpoint
g1_29dof_thermal_stage2 = replace(g1_29dof_thermal,
    training=replace(..., name="g1_29dof_thermal_stage2"),
    reward=replace(..., "penalty_joint_temperature": RewardTermCfg(weight=-2.0, ...)),
)
```

---

## Phase 4：生态扩展（⬜ 待开始）

以下功能不直接影响 locomotion thermal 训练，优先级较低。

### 4.1 TorqueMonitor Eval Callback

**来源**：FAR-FALCON `agents/callbacks/torque_monitor.py`（162 行）

实时 web UI（端口 5001），用 `toy_thermal_ig.sim2sim_monitor.MonitorServer`。显示 per-joint motor torque utilization（彩色 bar）+ thermal data + 速度命令/实际速度。包含 camera follow 模式。

Holosoma 现状：
- `agents/callbacks/base_callback.py` 有空的 `RLEvalCallback` 基类
- 无任何具体 callback 实现
- 启用方式需要适配 Tyro（FAR-FALCON 用 Hydra `+opt=eval_torque_monitor`）

### 4.2 StaticSquat 环境

**来源**：FAR-FALCON `envs/squat/static_squat.py`（继承 `LeggedRobotLocomotionThermal`）

静态深蹲任务，添加 `arm_symmetry` 和 `both_feet_contact` reward。Thermal config 启用 4 个 obs，penalty 强度 -2.0。

需要：新环境类 + reward terms + 实验配置。

### 4.3 UpperBodyForceResistance 环境

**来源**：FAR-FALCON `envs/upper_body/force_resistance.py`（继承 `LeggedRobotLocomotionThermal`）

在机器人手上施加向后力（默认 5N），保持平衡和行走。有 hand_position / force_resistance / hand_height / arm_symmetry 等自定义 reward。

需要：新环境类 + reward terms + body index 查找 + 实验配置。

### 4.4 G1 12dof 机器人配置

FAR-FALCON 有 `locomotion_g1_12dof_thermal.yaml`（obs_scale=0.008 vs 29dof 的 0.003）。
Holosoma 当前只有 g1_29dof，没有 g1_12dof robot config。不是 thermal 专属问题。

### 4.5 T1 robot thermal 变体

依赖 `toy_thermal_ig` 是否有 T1 的 `thermal_params.yaml`。Holosoma 已有 `config_values/loco/t1/`，robot 本身支持。

### 4.6 Sound system 集成

**来源**：FAR-FALCON `locomotion_thermal.py:21-34,223-253,362-377`

`sound_sim` 集成（VelocitySynthesizer / DirectionChangeSynthesizer / TorqueDeltaSynthesizer / FootStompSynthesizer），eval 时给机器人配音效。纯 demo 功能。

### 4.7 Sim2Real 真机温度反馈

FAR-FALCON `sim2real/foxglove_layouts/` 有温度可视化面板。bridge 中 MotorState `temperature` 字段被 TODO 了（未启用）。Holosoma `bridge/unitree/unitree_sdk2py_bridge.py` 没有温度处理。

如果要用 thermal policy 部署到真机（读真机电机温度作为 obs input），需要打通这条链路。

---

## 训练命令

```bash
# 1. 设置环境
source scripts/source_isaacgym_setup.sh

# 2. 训练 thermal 实验
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-thermal \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# 3. 对照组（非 thermal baseline）
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# 4. Fast SAC 变体
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# 多卡
torchrun --nproc_per_node=4 --master_port=29501 src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-thermal simulator:isaacsim logger:wandb --training.num-envs 16384
```

## 依赖

- `toy_thermal_ig`：thermal simulator 核心包
  - 默认路径：`~/Documents/gits/toy_thermal_ig`
  - 覆盖：`HOLOSOMA_THERMAL_REPO` 环境变量
  - thermal params 默认路径：`toy_thermal_ig/data/symmetric_batch_sysid_multifile_20250807_200807/thermal_params.yaml`
  - 覆盖：`HOLOSOMA_THERMAL_PARAMS_PATH` 环境变量
- `sound_sim`（Phase 4.6 only）：`~/Documents/gits/sound_sim`
- `sim2sim_monitor`（Phase 4.1 only）：`toy_thermal_ig` 子模块

## 当前参数快照（`g1_29dof_thermal`）

| 参数 | 当前值 | FAR-FALCON 等效值 | 备注 |
|---|---|---|---|
| `fast_dynamics` | `True` | `False` | Holosoma 温度上升更快 |
| `overall_scale` | `1.5` | `0.3` | Holosoma 惩罚 5x 强 |
| `temperature_exceeded threshold` | `110.0` | N/A（默认关闭） | Holosoma 独有 |
| `max_episode_length_s` | `20.0`（默认） | `100.0` | ⚠️ 差距 5x |
| `penalty_curriculum.degree` | `0.00025` | `0.0001` | Holosoma ramp 更快 |
| `penalty_curriculum.level_up_threshold` | `750` | `850` | 小差异 |
| `obs 启用数` | 1/5 | 1/5（locomotion）；4/5（squat） | 一致（locomotion 场景） |
| `command_ranges.lin_vel_x` | `[-1, 1]` 固定 | `[-1, 1]` → `[-4, 4]` ramp | ⚠️ 缺 curriculum |
