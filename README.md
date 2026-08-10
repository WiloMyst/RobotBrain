# RobotBrain

## 概述

RobotBrain 是一个 C++ 实现的端侧推理节点,把云端任务下发和本地 RL policy 推理 + 关节控制串成一条闭环链路。

通信分两层：控制面用 gRPC 承载低频任务和速度指令,数据面用 ROS2 (DDS) 承载高频本体感知和关节控制。推理用 ONNX Runtime 部署 unitree_rl_gym 预训练的 G1 locomotion policy,输入 47 维本体感知,输出 12 维关节目标。

> **当前状态**:已接入 MuJoCo 仿真闭环,部署 G1 RL walking policy,机器人可在仿真中稳定行走。控制循环跑在专用实时线程上,使用 `SCHED_FIFO + clock_nanosleep`。

## 架构与功能

### 1. 控制面 / 数据面分层通信

- **控制面 - gRPC:** 异步服务端基于 `ServerBuilder` + `CompletionQueue`,接收云端下发的任务和速度指令。任务携带 `TaskType` 枚举,端侧据此选择速度缩放和动作幅度限制。状态回传用服务端流式 RPC,定时上报 metrics。
- **数据面 - ROS2:** 订阅 `sensor_msgs::JointState`(关节位置/速度)+ `sensor_msgs::Imu`(角速度/姿态),合成 47 维本体感知;发布 `std_msgs::Float64MultiArray` 关节目标。
- **grpc::Alarm 定时:** 状态流式回传用 `grpc::Alarm` 在 CompletionQueue 上注册定时事件,避免阻塞式 sleep。

### 2. 推理管线

参考 unitree_rl_gym 的 G1 Actor 接口实现:

- **本体感知 (47 维):** `base_ang_vel(3)×0.25 + projected_gravity(3) + velocity_cmd(3)×[2,2,0.25] + dof_pos(12,相对 default 偏移)×1.0 + dof_vel(12)×0.05 + prev_actions(12) + sin_phase/cos_phase(2,步态时钟 period=0.8s)`
- **关节目标 (12 维):** `default_joint_positions + action * action_scale(0.25)`
- **projected_gravity:** 从 IMU 四元数计算重力在 base frame 的投影,参考 legged_gym `quat_rotate_inverse` 语义
- **步态时钟:** 用帧计数计算 phase(`total_frames * control_dt`),填入 `sin(2π·phase)` 和 `cos(2π·phase)`,参考 unitree_rl_gym deploy_mujoco 约定

### 3. 并发控制与安全防护

- **任务世代号:** 新指令到达时自增世代号,推理完成后比对,若已被抢占则丢弃陈旧结果
- **背压丢帧:** 用 `compare_exchange_strong` 原子地置位 `is_inferring` 标记,上一帧仍在推理时当前帧直接丢弃
- **软件看门狗:** 500ms 心跳超时判定,超时回 default 站立姿态
- **关节限位:** policy 输出相对 default 的偏移不超过 `joint_position_limit`

### 4. 实时性改造

控制循环从 ROS2 `WallTimer` 升级为专用实时线程:

- **SCHED_FIFO:** `pthread_setschedparam(SCHED_FIFO, priority=80)`。WSL2 非 root 下降级到 SCHED_OTHER
- **clock_nanosleep:** `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next)`,绝对时间唤醒,消除累积漂移
- **mlockall:** `mlockall(MCL_CURRENT | MCL_FUTURE)`,防止页被换出导致抖动
- **jitter 测量:** 每帧记录 `实际唤醒时间 - 预期唤醒时间`,环形缓冲 1000 帧统计 P50/P95/P99/max
- **降级策略:** SCHED_FIFO/mlockall 失败时 warning 但继续运行,保证非 root 环境可跑

## 模型与仿真

### Locomotion Policy

- **来源:** unitree_rl_gym 官方预训练 G1 walking policy(`deploy/pre_train/g1/motion.pt`,PyTorch JIT)
- **转换:** `tools/reexport_onnx.py` 提取 JIT 权重重建无状态 LSTM policy,导出 ONNX(3 输入:obs/hidden/cell,3 输出:action/new_hidden/new_cell),输出 `models/policy_g1.onnx`
- **架构:** LSTM(47→64) + MLP(64→32→12),输入 47 维本体感知含步态时钟,输出 12 维关节目标偏移
- **PD 参数:** 对齐 `deploy_mujoco/configs/g1.yaml`:`kp=[100,100,100,150,40,40,...]` `kd=[2,2,2,4,2,2,...]`

### MuJoCo 仿真闭环

- **仿真器:** MuJoCo + unitree_rl_gym 官方 G1 MJCF(`g1_12dof.xml`,12-DoF 双腿)
- **数据流:** MuJoCo 500Hz 物理步进 → 100Hz 发布 `/joint_states` + `/imu` → RobotBrain 50Hz 推理 → `/joint_commands` → MuJoCo PD 控制器转力矩
- **PD 控制器:** `tau = kp[i] * (target[i] - q[i]) - kd[i] * dq[i]`,逐关节独立增益
- **关节映射:** G1 标准顺序 `left_hip_pitch → left_hip_roll → left_hip_yaw → left_knee → left_ankle_pitch → left_ankle_roll → right 同理`
- **IMU 角速度:** 直接用 world frame 的 `qvel[3:6]`,不转 body frame,对齐 legged_gym `base_ang_vel` 约定
- **启动时序:** MuJoCo 启动后保持 standing reset 姿态(physics 不步进),等收到 RobotBrain 的第一条 joint_commands 才启用物理,避免 DDS 发现延迟期间机器人因无控制而摔倒

## 性能数据

WSL2 Ubuntu 22.04 + SCHED_OTHER(非 root) 环境:

| 指标 | 值 | 说明 |
|------|-----|------|
| 控制频率 | 50Hz | control_rate_ms=20 |
| ONNX 推理延迟 | 0.2ms | ONNX Runtime CPU EP |
| jitter P50 | 0.12ms | clock_nanosleep 绝对时间定时 |
| jitter P99 | 0.32ms | WSL2 SCHED_OTHER 下 |
| 丢帧率 | 0% | CAS 背压 + 世代号抢占 |

> WSL2 受 Hypervisor 调度限制只能做软实时。

## 运行方式

### 1. 准备模型与仿真资产

模型和仿真资产需下载。如需重新导出 ONNX:

```bash
python3 tools/reexport_onnx.py          # 读 models/pretrained/motion_g1.pt,输出 models/policy_g1.onnx
```

### 2. MuJoCo 仿真闭环

三终端分别启动(需先 source ROS2 与 colcon install 环境):

```bash
# T1: MuJoCo 仿真器
cd ~/robotbrain_ws/src/RobotBrain/tools
python3 mujoco_sim_node.py --ros-args \
  -p render_output:=/mnt/d/Triton/RobotBrain/sim_view.png \
  -p render_interval_ms:=500

# T2: 推理节点
ros2 run robot_brain robot_brain_node

# T3: 云端客户端
python3 tools/cloud_client.py
```

浏览器打开 `sim_view.html` 可看仿真画面,每 500ms 自动刷新。

### 3. 性能 benchmark

需先启动 T1 + T2:

```bash
python3 tools/benchmark_client.py      # 持续 30s 下发 NAVIGATE 并采集延迟/丢帧数据
```

## 依赖

- C++ 17 / CMake
- ROS 2 Humble / DDS
- gRPC / Protobuf
- ONNX Runtime
- MuJoCo (Python bindings)
- unitree_rl_gym 官方 G1 MJCF
