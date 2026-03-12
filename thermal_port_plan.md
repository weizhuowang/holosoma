# Thermal Port 计划：FAR-FALCON → Holosoma

> **状态：✅ 实现完成** — 2026-03-10
> 6 个文件修改，+265/-55 行。语法验证通过，待训练验证。

## 背景

FAR-FALCON (HumanoidVerse) 中的 thermal 功能需要 port 到 Holosoma 的 Manager-Based 架构。
当前 `dev_thermal` 分支已有一个初步 port（commit `7ee380f`），功能可用但不符合 Holosoma 的架构惯例。

## 已修复的问题（原 port 的问题）

1. ~~obs/reward/termination 函数全写在 `envs/locomotion/locomotion_thermal_manager.py` 里~~ → 已搬到 `managers/xxx/terms/locomotion.py`
2. ~~thermal reset 硬写在 `_reset_tasks_callback`，参数 hardcode~~ → 已改为 RandomizationManager 的 `reset_terms` 驱动
3. ~~只 port 了 1 个观测（winding temp）~~ → 已加齐 5 个观测（winding, case, rate, headroom, initial）
4. commit 混入了无关改动（device 字段）— 保留不动
5. ~~reward 配置里 `params={}` 为空~~ → 已显式暴露 `start_temp`, `ramp_temp`, `overall_scale`

## 设计方案

### 原则

- term 函数放 `managers/xxx/terms/`，不放环境类文件
- 环境类只负责 ThermalSimulator 初始化 + 每步温度更新
- thermal reset 走 RandomizationManager 的 reset_terms
- 所有 magic number 通过 `params` 配置传入
- 不新建文件夹或文件（thermal terms 直接加到现有 locomotion terms 文件里）

### 文件变更

#### 1. `managers/observation/terms/locomotion.py` — 加 5 个观测函数

```python
def joint_temperature(env, ambient_temp=30.0, clip_max=300.0) -> Tensor:
    """绕组温度，(num_envs, num_dof)。非 thermal 关节填 ambient_temp。"""

def joint_case_temperature(env, ambient_temp=30.0, clip_max=300.0) -> Tensor:
    """壳体温度，(num_envs, num_dof)。"""

def joint_temperature_rate(env, clip_range=10.0) -> Tensor:
    """温度变化率 (°C/s)，clamp 到 ±clip_range。"""

def joint_temperature_headroom(env, max_temp=110.0, ambient_temp=30.0) -> Tensor:
    """距最高温限制的余量。"""

def joint_temperature_initial(env, ambient_temp=30.0) -> Tensor:
    """episode 开始时的初始温度（帮助 policy 理解 thermal budget）。"""
```

所有函数签名统一 `(env, **params)`，通过 `getattr` 安全访问 thermal 属性，
非 thermal 环境调用时返回默认值（ambient_temp 或 zero）。

#### 2. `managers/reward/terms/locomotion.py` — 加 1 个奖励函数

```python
def penalty_joint_temperature(env, start_temp=60.0, ramp_temp=50.0,
                               max_penalty=1.5, max_weight=0.8,
                               mean_weight=0.2, overall_scale=0.3) -> Tensor:
    """温度惩罚。从 start_temp 开始线性增长，0.8*max + 0.2*mean。"""
```

从现有 `locomotion_thermal_manager.py` 底部搬过来，逻辑不变。

#### 3. `managers/termination/terms/locomotion.py` — 加 1 个终止函数

```python
def temperature_exceeded(env, threshold=90.0) -> Tensor:
    """任意关节 winding > threshold 则终止。"""
```

从现有 `locomotion_thermal_manager.py` 底部搬过来，逻辑不变。

#### 4. `managers/randomization/terms/locomotion.py` — 加 thermal reset term

```python
def thermal_reset(env, randomize_temp=True, winding_temp_range=(30.0, 50.0),
                  case_temp_range=(30.0, 50.0), hot_knee_prob=0.4,
                  hot_hip_prob=0.4, hot_joint_temp=90.0):
    """reset 时随机化温度初始状态。委托给 env.apply_thermal_reset()。"""
```

环境类保留 `apply_thermal_reset()` 和 `_sample_initial_thermal_temps()` 作为底层方法，
randomization term 只是包装调用。

#### 5. `envs/locomotion/locomotion_thermal_manager.py` — 精简环境类

保留：
- `__init__` + `_init_thermal_simulator()`：初始化 ThermalSimulator
- `_pre_compute_observations_callback()`：每步更新温度
- `_update_thermal_simulation()`：调用 thermal_simulator.step()
- `apply_thermal_reset()` + `_sample_initial_thermal_temps()`：被 randomization term 调用
- `_update_log_dict()`：thermal logging
- `get_thermal_info()`：监控接口

删除：
- `_reset_tasks_callback()` 中的 hardcode reset（改由 randomization term 驱动）
- 文件底部的 3 个独立函数（搬到 managers/terms/）

#### 6. `config_values/loco/g1/experiment.py` — 更新配置

更新 `g1_29dof_thermal` 的 func 路径：
- `holosoma.envs.locomotion.locomotion_thermal_manager:obs_joint_temperature`
  → `holosoma.managers.observation.terms.locomotion:joint_temperature`
- `holosoma.envs.locomotion.locomotion_thermal_manager:reward_penalty_joint_temperature`
  → `holosoma.managers.reward.terms.locomotion:penalty_joint_temperature`
- `holosoma.envs.locomotion.locomotion_thermal_manager:termination_temperature_exceeded`
  → `holosoma.managers.termination.terms.locomotion:temperature_exceeded`

补充 reward params：
```python
"penalty_joint_temperature": RewardTermCfg(
    func="holosoma.managers.reward.terms.locomotion:penalty_joint_temperature",
    weight=-1.0,
    params={"start_temp": 60.0, "ramp_temp": 50.0, "overall_scale": 0.3},
    tags=["penalty_curriculum"],
),
```

加 randomization reset term：
```python
randomization=replace(randomization.g1_29dof_randomization, reset_terms={
    **randomization.g1_29dof_randomization.reset_terms,
    "thermal_reset": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:thermal_reset",
        params={
            "winding_temp_range": [30.0, 50.0],
            "case_temp_range": [30.0, 50.0],
            "hot_knee_prob": 0.4,
            "hot_hip_prob": 0.4,
            "hot_joint_temp": 90.0,
        },
    ),
}),
```

#### 7. `agents/modules/augmentation_utils.py` — 不用动

`mirror_obs_joint_temperature` 已经实现，适用于所有 5 个 thermal 观测
（都是 per-DOF scalar，只需 index remap，不需 sign flip）。

### 可选：多阶段训练配置

```python
# Stage 1: 学基础行走，关闭温度惩罚和终止
g1_29dof_thermal_stage1 = replace(g1_29dof_thermal,
    training=replace(..., name="g1_29dof_thermal_stage1"),
    reward=replace(..., "penalty_joint_temperature": RewardTermCfg(weight=0.0, ...)),
    termination=replace(...),  # 去掉 temperature term
)

# Stage 2: 打开温度惩罚和终止
g1_29dof_thermal_stage2 = replace(g1_29dof_thermal,
    training=replace(..., name="g1_29dof_thermal_stage2"),
    reward=replace(..., "penalty_joint_temperature": RewardTermCfg(weight=-2.0, ...)),
)
```

### 不 port 的内容

以下 FAR-FALCON 功能暂不 port（与当前 locomotion thermal 无关或优先级低）：
- Sound system（`sound_sim` 集成）
- Torque monitor callback（独立功能，不影响训练）
- Static squat 环境
- Upper body force resistance 环境
- Command curriculum（FAR-FALCON 写在环境类里，Holosoma 应该在 CurriculumManager 里实现）
- T1 机器人 thermal 变体（等 G1 稳定后再加）

### 无关改动处理

commit `7ee380f` 中的 `device` 字段和 `_normalize_device` 改动与 thermal 无关，
应该拆成独立 commit 或保留在主分支。重构时不动这部分。

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
```

**依赖**: thermal 实验需要 `toy_thermal_ig` 包：
- 默认路径：`~/Documents/gits/toy_thermal_ig`
- 可通过 `HOLOSOMA_THERMAL_REPO` 环境变量指定
- thermal params 默认路径：`toy_thermal_ig/data/symmetric_batch_sysid_multifile_20250807_200807/thermal_params.yaml`
- 可通过 `HOLOSOMA_THERMAL_PARAMS_PATH` 环境变量覆盖

## 验证计划

1. ⬜ 基础验证：`exp:g1-29dof-thermal` 能正常启动训练
2. ⬜ 观测验证：检查 obs 维度匹配（当前只在 actor_obs 和 critic_obs 中加了 `joint_temperature`）
3. ⬜ 奖励验证：WandB 中确认 `rew_penalty_joint_temperature` 曲线正常
4. ⬜ Reset 验证：检查 randomization term 是否正确调用 `apply_thermal_reset`
5. ⬜ 对称性验证：`use_symmetry=True` 下检查 thermal obs mirror 正确
6. ⬜ 非 thermal 回归：确认 `exp:g1-29dof` baseline 不受影响
7. ⬜ Thermal logging：确认 `max_winding_temp`, `temp_penalty_*` 等指标正常上报

## 变更文件清单

| 文件 | 变更 | 状态 |
|------|------|------|
| `managers/observation/terms/locomotion.py` | +5 thermal 观测函数 | ✅ |
| `managers/reward/terms/locomotion.py` | +1 温度惩罚函数 | ✅ |
| `managers/termination/terms/locomotion.py` | +1 温度超限终止函数 | ✅ |
| `managers/randomization/terms/locomotion.py` | +1 thermal reset term | ✅ |
| `envs/locomotion/locomotion_thermal_manager.py` | 精简：删除冗余函数和 hardcode reset | ✅ |
| `config_values/loco/g1/experiment.py` | 更新 func 路径 + 加 randomization term | ✅ |
| `agents/modules/augmentation_utils.py` | 不用改（已有 mirror 支持） | ✅ |
