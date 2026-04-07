# Holosoma Viser Viewer 设计

## 目标

为 `holosoma` 的 sim2sim 增加一个基于 `viser` 的浏览器可视化方案，使它在无头服务器场景下可以基本替代当前 MuJoCo 桌面 viewer。

## 当前实现状态

截至目前，这份设计里最核心的一条链已经落地：

- `run_sim.py` 支持 `viewer.backend = native | viser | both | none`
- `training.headless=True` 时不再被 direct sim 硬编码覆盖
- MuJoCo 支持在无头模式下启动 `viser` 浏览器 viewer
- 浏览器侧已支持：
  - 机器人 URDF mesh 可视化
  - 基础状态面板
  - camera tracking 开关
  - reset
  - gantry raise / lower / toggle / force sign / zero command
- 原生 MuJoCo viewer 和 `viser` 可以独立启用，也可以双开

当前还没有完全覆盖的项：

- terrain / static object 的完整 scene 可视化
- 多客户端单控制者仲裁
- 更丰富的命令面板

所以它现在属于“可用的第一版”，不是最终完整版。

主要目标场景：

- `run_sim.py` + MuJoCo sim2sim
- 仿真跑在无头 Linux 服务器上
- 用户从另一台机器用浏览器查看
- 覆盖 sim2sim bring-up 过程中最常用的可视化和控制能力

## 为什么要做

目前 MuJoCo sim2sim 依赖本地桌面 viewer：

- `holosoma/simulator/mujoco/mujoco.py` 里使用的是 `mujoco.viewer.launch_passive(...)`
- `run_sim.py` 默认是本地窗口工作流
- 一旦 headless，就会直接跳过 viewer 初始化

这带来几个问题：

- 没有浏览器可访问的远程 viewer
- 无头服务器上不方便排查 sim2sim 问题
- MuJoCo 窗口里的常用控制，例如 `7/8/9`、reset、camera tracking，没有远程替代方案

## 当前调研结论

### 已有基础

1. `holosoma_retargeting` 已经成功使用了 `viser`
2. retargeting 代码已经验证了以下能力是可复用的：
   - `viser.ViserServer`
   - `ViserUrdf`
   - 浏览器 3D scene 搭建
   - MuJoCo `qpos` 下 base pose 的 `wxyz` 约定
   - 基本 GUI 结构，例如 folder、button、slider

### 当前没有的能力

1. sim2sim 主流程里没有接入 `viser`
2. `viser` 目前只出现在 `holosoma_retargeting`，不在主 MuJoCo sim2sim 路径里
3. 当前 MuJoCo viewer 的逻辑是典型本地窗口 + 键盘回调模式
4. direct sim 里 MuJoCo 当前主要是 robot-only 的 live state，可视化 scene/object 还没有完全打通

### 一个很关键的现有问题

`DirectSimulation.initialize()` 里当前有一行：

```python
self.simulator.set_headless(False)
```

这是硬编码，和后面 `config.training.headless` 的逻辑不一致。

如果不先修这个问题，后面即使接了 `viser`，真正的“headless + remote viewer”工作流也会不稳定。

## 对 retargeting 的复用策略

### 可以直接借鉴的部分

retargeting 的 `viser` 实现很值得参考，尤其是 scene 层：

- `ViserServer` 生命周期
- `ViserUrdf` 机器人加载方式
- 基于 `yourdfpy` / URDF 的 mesh 展示
- GUI 面板组织方式

### 不能直接照搬的部分

retargeting 本质上是离线 `npz` 播放器，而 sim2sim 是在线 live simulator，这两者差别很大：

- retargeting 不需要 live simulator integration
- retargeting 不需要线程安全的控制回写
- retargeting 没有 gantry / reset / tracking 控件
- retargeting 不需要实时 telemetry
- retargeting 不需要 multi-client 策略

结论：

- retargeting 对浏览器 scene 层非常有借鉴意义
- sim2sim 的控制架构、线程模型、状态同步仍然需要重新设计

## 产品定义

这个功能的目标不是“在浏览器里看个机器人姿态”，而是做一个对 sim2sim 真正有用的远程 viewer。

### 必须具备的能力

1. 浏览器里实时显示机器人状态
2. 仿真机不需要 X server
3. 通过 URDF mesh 进行机器人可视化
4. 支持 camera tracking 开关
5. 支持 virtual gantry 常用控制：
   - 抬高
   - 放低
   - 开关
   - 调整 force
   - force sign toggle
6. 支持 reset
7. 支持状态面板，至少包括：
   - sim time
   - sim FPS target
   - viewer update rate
   - gantry status
   - camera tracking status
   - current command vector
8. 浏览器回调必须线程安全
9. 适配 G1 和 T1 的 MuJoCo sim2sim

### 加分项

1. 显示 terrain / static scene
2. WBT 下显示 object asset
3. 多客户端只读 + 单控制者机制
4. 支持 native viewer + viser 双开
5. 提供更丰富的 direct sim 控制面板

## 预期使用方式

### 推荐命令行形式

```bash
source scripts/source_mujoco_setup.sh
export MUJOCO_GL=egl

python src/holosoma/holosoma/run_sim.py robot:g1-29dof \
  --training.headless=True \
  --simulator.config.viewer.backend viser \
  --simulator.config.viewer.viser.host 0.0.0.0 \
  --simulator.config.viewer.viser.port 8080
```

### 浏览器使用体验

启动后日志里应该打印类似：

```text
Viser viewer ready at http://0.0.0.0:8080
```

浏览器里至少应该有：

- 3D 机器人视图
- 状态面板
- gantry 控件
- reset 按钮
- camera tracking 开关
- 可选的高级控制区

## 总体架构方案

### 1. 扩展 viewer 配置

当前 `ViewerConfig` 太薄，无法表达“本地 viewer / 浏览器 viewer / 双开 / 不开”的区别。

建议扩成下面这种结构。

注：当前已实现版本里没有单独保留 `enabled` 字段，而是直接用 `backend` 控制启停。

```python
@dataclass(frozen=True)
class ViserViewerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    show_grid: bool = True
    show_meshes: bool = True
    show_gantry: bool = True
    allow_controls: bool = True


@dataclass(frozen=True)
class ViewerConfig:
    backend: Literal["native", "viser", "both", "none"] = "native"
    enable_tracking: bool = False
    camera: CameraConfig | None = None
    viser: ViserViewerConfig = field(default_factory=ViserViewerConfig)
```

这样设计的原因：

- `training.headless` 应该只表示“是否开本地桌面窗口”
- `viewer.backend` 决定有没有浏览器 viewer
- 这样可以自然支持本地、远程和双 viewer 工作流

### 2. 给 MuJoCo 增加专用的 Viser backend

建议新增：

```text
src/holosoma/holosoma/simulator/mujoco/viser_viewer.py
```

核心职责：

- 创建 `ViserServer`
- 加载机器人 URDF mesh
- 管理浏览器 GUI
- 以固定刷新率发布仿真状态
- 接收浏览器控制，并把控制请求放进线程安全队列

建议主类形态如下：

```python
class MuJoCoViserViewer:
    def start(self) -> None: ...
    def update(self) -> None: ...
    def drain_pending_commands(self) -> None: ...
    def close(self) -> None: ...
```

### 3. 所有控制都必须回到仿真线程执行

这是整个方案最重要的规则。

不能让浏览器回调线程直接改 MuJoCo 状态。

正确做法是：

1. GUI 回调只负责把命令放进线程安全队列
2. MuJoCo 主仿真线程每个 step 去 drain 这个队列
3. 真正的 simulator 状态修改在仿真线程内完成

好处：

- 避免 MuJoCo 状态竞争
- 命令执行更可预测
- 更符合物理仿真循环的语义

### 4. 复用现有控制逻辑，不要在 web 层重复发明一套

当前 MuJoCo viewer 里已经有不少控制逻辑：

- `CommandRegistry`
- `VirtualGantry.handle_command(...)`
- reset / episode start 相关逻辑

Viser viewer 不应该重新发明这些控制行为，而应该调用同一套底层逻辑。

建议做法：

- 把目前偏“按键驱动”的逻辑拆出命名 command 接口
- native viewer 仍然保留 `CommandRegistry`
- browser viewer 改为调用同一套 named command

理想形式例如：

```python
simulator.execute_named_command("gantry_raise")
simulator.execute_named_command("reset")
simulator.execute_named_command("toggle_camera_tracking")
```

这样 web 层就不用伪装成 GLFW keycode 了。

### 5. 增加结构化的 viewer snapshot

建议增加一个面向 UI 的状态快照对象：

```python
@dataclass
class ViewerSnapshot:
    sim_time: float
    robot_base_pos: np.ndarray
    robot_base_quat_wxyz: np.ndarray
    joint_positions: np.ndarray
    commands: np.ndarray | None
    gantry_enabled: bool
    gantry_length: float | None
    gantry_force: float | None
    camera_tracking: bool
    target_fps: float
```

由 simulator/backend 侧生成 snapshot，viser 侧只消费 snapshot。

好处：

- sim 逻辑和 UI 逻辑边界更清晰
- 更好测
- 未来要扩到 IsaacGym / IsaacSim 也更自然

### 6. 机器人资源路径必须走现有 config

机器人 URDF 路径不要写死。

必须基于现有：

- `robot_config.asset.asset_root`
- `robot_config.asset.urdf_file`

并复用项目里已有的 `@holosoma/...` 解析逻辑。

### 7. Joint mapping 必须显式做

不能假设 `ViserUrdf` 的关节顺序一定和 simulator 的 `self.dof_names` 完全一致。

实现时必须显式建立：

- simulator DOF name -> URDF actuated joint name
- `ViserUrdf.update_cfg(...)` 需要的关节顺序向量

这对 G1 / T1 的正确性都很重要。

## Scene 范围

### 第一阶段

先做这些内容：

1. robot URDF mesh
2. grid / plane
3. gantry anchor 的可视化 marker
4. gantry anchor 到机器人主体的连线

### 第二阶段

再补这些内容：

1. terrain mesh / static scene
2. WBT 的 object asset
3. 更多 debug visuals

为什么分阶段：

- 当前 direct sim 下机器人 live state 是最清晰、最好拿到的
- scene/object 的 live state 还没有同等成熟
- 先把 robot-first 做起来，已经能解决大部分 headless sim2sim 痛点

## 控制范围

### 第一阶段控制

优先覆盖 MuJoCo 窗口里 sim2sim 最常用的控制：

- gantry raise
- gantry lower
- gantry toggle
- gantry force adjust
- gantry force sign toggle
- reset sim
- toggle camera tracking
- show/hide status panel

### 第二阶段控制

如果希望达到更完整的 direct sim parity，再补：

- velocity command 按钮 / slider
- walk / stand toggle
- waist yaw / height 调整
- zero command

说明：

对纯 sim2sim 来说，policy 主要还是由 `run_policy.py` 控制。浏览器控制更适合 direct sim 调试和补充控制。

## 多客户端策略

推荐默认策略：

- 多个客户端都能连进来
- 都可以看
- 但同一时间只有一个客户端能控制

最简单的实现方式：

1. 第一个点击 “Take Control” 的客户端成为 controller
2. 其他客户端默认只读
3. controller 释放前，其他客户端不能执行 reset / gantry 等控制
4. 服务端日志记录控制权切换

这样可以避免多人误操作。

## 集成计划

### Milestone 0：前置清理

1. 修复 `DirectSimulation.initialize()`，让它真正尊重 `config.training.headless`
2. 解耦“是否有 native window”和“是否有 remote viewer”
3. 给 config 增加 viewer backend 选择

### Milestone 1：最小可用 remote Viser viewer

1. 给主 `holosoma` 包增加 `viser` 依赖
2. 实现 `MuJoCoViserViewer`
3. 浏览器里显示 live robot mesh
4. 增加状态面板
5. 增加 camera tracking 开关

验收标准：

- 用户可以在无头服务器上跑 MuJoCo sim2sim，并在浏览器里看到机器人

### Milestone 2：达到远程操作基本可用

1. 增加 gantry 控制
2. 增加 reset
3. 状态面板补齐 gantry / tracking 状态
4. 增加 command queue 和线程安全控制通路

验收标准：

- 用户在正常 sim2sim bring-up 中，不再依赖 MuJoCo 本地桌面窗口

### Milestone 3：补 scene 完整度

1. 增加 terrain / static scene 可视化
2. 增加 WBT object 可视化
3. 增加 gantry line / anchor debug visuals

验收标准：

- 浏览器视图不只适合平地 locomotion，也足够支撑 scene-heavy / WBT 工作流

### Milestone 4：打磨

1. 多客户端控制权机制
2. 支持 `backend=both`
3. 更好的 telemetry 布局
4. 浏览器内帮助文档 / control legend
5. 文档和 troubleshooting 更新

## 依赖方案

推荐方案：

- 直接把 `viser` 加到 `src/holosoma/pyproject.toml` 的主依赖里

原因：

- 这个功能属于主 `holosoma` 的 MuJoCo sim2sim
- 现有主包本身已经带了不少可视化依赖
- 这样最省心，安装和使用成本最低

备选方案：

- 做成 `holosoma[viser]` 这样的 optional extra

理论上更干净，但实际会增加 setup 复杂度，对当前 repo 价值不大。

## 测试计划

### 单元测试

1. 机器人 URDF 资源路径解析
2. simulator DOF 到 `ViserUrdf` joint order 的 mapping
3. browser command queue 的 enqueue / dequeue 行为
4. 控制权仲裁逻辑

### Smoke test

1. `viewer.backend=viser` 在没有 X server 的情况下能启动
2. 浏览器可以连上并收到 live 状态
3. 浏览器里的 gantry 控制能生效
4. 浏览器里的 reset 能生效

### 手工验证矩阵

1. G1 locomotion sim2sim
2. T1 locomotion sim2sim
3. G1 WBT sim2sim
4. headless server + EGL
5. 本地开发机 + `backend=both`

## 风险与约束

1. `ViserUrdf` 的 joint 顺序不一定和 simulator 一样，必须显式 mapping
2. scene / object 渲染会比 robot 渲染难，因为 direct MuJoCo 这部分 live state 目前还比较薄
3. 浏览器回调默认是并发的，所以 command queue 是必须的，不是可选项
4. 不能把整个设计过度绑定在 MuJoCo keycode 上，否则后续难维护

## 推荐的第一批实现范围

如果追求最快落地并且立即有价值，建议先做这一版：

1. 修 headless 逻辑
2. 加 `viewer.backend=viser`
3. 在 MuJoCo sim 中启动 `ViserServer`
4. 实时同步 robot pose + joints
5. 增加 reset / gantry / camera tracking 控制
6. 增加 grid + status panel

这版已经足够让 headless server 上的 sim2sim 真正能用了。

## 后续实现顺序建议

正式动手时，建议按这个顺序推进：

1. 先加 config types
2. 再加 `MuJoCoViserViewer`
3. 再补 command queue 和 simulator integration hook
4. 再把控制逻辑从 keycode-only 重构成 named command
5. 最后补文档和验证脚本

## 最终决策总结

建议路线如下：

- 要参考 retargeting 的 `viser` 实现
- 但不能直接把 retargeting player 改一改就塞进 sim2sim
- 应该为 MuJoCo sim2sim 单独做一个 `Viser` backend
- 核心控制逻辑要通过线程安全 command queue 回到仿真线程
- headless server 支持必须被当作一等工作流，而不是附带能力
