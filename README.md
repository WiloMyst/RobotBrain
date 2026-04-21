# RobotBrain-VLA (具身智能边缘大模型计算中枢)

## 概述

**RobotBrain-VLA** 是一个专为具身智能（机器人/自动驾驶）设计的企业级端侧大模型边缘计算节点。

系统严格遵循分布式架构中的**“控制面（Control Plane）与数据面（Data Plane）深度解耦”**原则。以 C++ 高性能异步引擎为核心，通过集成 gRPC 与 ROS2 (DDS)，完美打通了“云端高级语义指令下发”到“端侧视觉-动作 (Vision-Language-Action) 大模型闭环控制”的完整链路。

## 架构与功能特性

### 1. 云边协同的双轨通信引擎 (Dual-Track Comm Engine)

传统纯 RPC 架构无法满足机器人底层海量传感器的分发需求，本项目设计了双模态网络底座：

- **控制面 - gRPC 高可靠微服务：** 负责接收云端车队管理系统 (FMS) 或手机端下发的低频、高优语义指令（如：`"跟随红色目标"`）。跨语言友好，保障指令的绝对送达。
- **数据面 - ROS2 (DDS) 实时广播：** 采用去中心化的发布/订阅模式。引擎内部直连原生机器人底盘，实现对高频（30Hz+）高清摄像头点云的毫秒级订阅，以及控制扭矩的高速下发。

### 2. 极致榨取硬件的视觉推理管线 (Zero-Copy Vision Pipeline)

针对端侧算力板（如 Jetson Orin）显存与内存瓶颈，设计了极端的数据流：

- **原生 QoS 极限调优：** 摒弃 ROS2 默认的可靠传输，强制在视觉节点挂载 `SensorDataQoS (Best Effort)`，并将队列深度极化为 `1`。确保 AI 引擎在任何时刻抓取的都是绝对新鲜的“物理世界快照”，从网络底层根除延迟积压。
- **SIMD 预处理与零拷贝 (Zero-Copy)：** 基于自研 RAII 高性能物理内存池 (`BufferPool`)，巧妙利用 `cv::Mat` 包装器进行张量预处理。结合 OpenCV 的底层向量化指令集，一步完成图像解码、缩放、HWC->CHW 转换与归一化，并直接映射为 ONNX 张量，全程 **0 次 Heap Allocation (堆内存分配)**。

### 3. 工业级并发状态机与安全熔断 (Industrial Safety & Concurrency)

在复杂的实体物理世界，代码的 Bug 意味着机器人的硬损坏。本项目在并发控制上实施了最严苛的防护：

- **无锁世代调度机制 (Generation ID)：** 创新引入基于 `std::atomic<uint64_t>` 的任务世代号机制。云端新指令到达瞬间即可完成抢占，作废后台线程陈旧的推理结果，避免新旧运动指令冲突（Lock-Free Preemption）。
- **防洪闸门与背压 (Backpressure)：** 采用 CAS (Compare-And-Swap) 原语实现 `is_inferring` 绝对锁死。当算力不足引发帧率倒挂时，主动执行静默丢帧 (Drop Frame)，拒绝系统雪崩。
- **物理级软件看门狗 (Software Watchdog)：** 控制循环内建 500ms 致命阈值。一旦发生网络断联、进程死锁或推理卡死，心跳丢失瞬间触发底盘 `PublishControl(0, 0)` 强制刹车，保障具身硬件绝对安全。

## 版本与依赖

- **核心语言：** C++ 17
- **构建系统：** CMake (ament_cmake)
- **中间件架构：** ROS 2 (Humble) / DDS (FastRTPS/Cyclone)
- **网络与服务：** gRPC / Protobuf
- **异构算力引擎：** ONNX Runtime (C++ API), 兼容 CUDA Execution Provider
- **机器视觉底座：** OpenCV 4.x