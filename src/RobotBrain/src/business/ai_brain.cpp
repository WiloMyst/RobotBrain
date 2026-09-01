#include "engine/business/ai_brain.h"
#include "engine/infra/thread_pool.hpp"
#include "engine/core/ros_node.h"
#include "engine/business/models/policy_model.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace engine {
namespace business {

namespace {
// 组装 47 维本体感知观测向量 (unitree_rl_gym / G1 标准)
// [0:3]   ang_vel * ang_vel_scale      (IMU 角速度,缩放)
// [3:6]   projected_gravity            (重力在 base frame 投影,不缩放)
// [6:9]   velocity_command * cmd_scale (用户下发,逐维缩放)
// [9:21]  dof_pos * dof_pos_scale      (相对 default 偏移,缩放) -- 12 维
// [21:33] dof_vel * dof_vel_scale      -- 12 维
// [33:45] prev_actions                -- 12 维
// [45:47] sin_phase, cos_phase        (步态时钟,period=gait_phase_period)
void BuildObservation(const core::Proprioception& prop,
                      const float vel_cmd[3],
                      const std::vector<float>& default_dof,
                      const std::vector<float>& prev_actions,
                      int num_joints,
                      float ang_vel_scale,
                      float dof_pos_scale,
                      float dof_vel_scale,
                      const float cmd_scale[3],
                      float sin_phase,
                      float cos_phase,
                      float* obs_out) {
    float* p = obs_out;
    p[0] = prop.angular_velocity[0] * ang_vel_scale;
    p[1] = prop.angular_velocity[1] * ang_vel_scale;
    p[2] = prop.angular_velocity[2] * ang_vel_scale;
    p[3] = prop.projected_gravity[0];
    p[4] = prop.projected_gravity[1];
    p[5] = prop.projected_gravity[2];
    p[6] = vel_cmd[0] * cmd_scale[0];
    p[7] = vel_cmd[1] * cmd_scale[1];
    p[8] = vel_cmd[2] * cmd_scale[2];
    for (int i = 0; i < num_joints; ++i) {
        p[9 + i] = (prop.joint_positions[i] - default_dof[i]) * dof_pos_scale;
    }
    for (int i = 0; i < num_joints; ++i) {
        p[21 + i] = prop.joint_velocities[i] * dof_vel_scale;
    }
    for (int i = 0; i < num_joints; ++i) {
        p[33 + i] = prev_actions[i];
    }
    p[45] = sin_phase;
    p[46] = cos_phase;
}
} // namespace

AIBrain::AIBrain(const BrainConfig& config, core::RosNode* ros_body)
    : ros_body_(ros_body),
      num_joints_(config.num_joints),
      obs_dim_(config.obs_dim),
      action_scale_(config.action_scale),
      default_joint_positions_(config.default_joint_positions),
      max_joint_velocity_(config.max_joint_velocity),
      joint_position_limit_(config.joint_position_limit),
      ang_vel_scale_(config.ang_vel_scale),
      dof_pos_scale_(config.dof_pos_scale),
      dof_vel_scale_(config.dof_vel_scale),
      gait_phase_period_(config.gait_phase_period),
      control_dt_(config.control_rate_ms / 1000.0f),
      max_vel_ramp_(config.max_vel_ramp),
      fall_gravity_threshold_(config.fall_gravity_threshold),
      fall_debounce_frames_(config.fall_debounce_frames),
      watchdog_timeout_ms_(config.watchdog_timeout_ms) {
    spdlog::info("AIBrain: loading RL locomotion policy from {}", config.model_path);
    policy_model_ = std::make_unique<models::PolicyModel>(
        config.model_path, config.intra_op_threads, config.use_gpu);

    inference_pool_ = std::make_unique<infra::ThreadPool>(1, 10);
    tensor_pool_ = std::make_unique<infra::BufferPool<float>>(
        3, static_cast<size_t>(obs_dim_));

    prev_actions_.assign(num_joints_, 0.0f);
    last_joint_targets_ = default_joint_positions_;
    metrics_.current_state.store(RobotState::STANDING);

    cmd_scale_[0] = config.cmd_scale[0];
    cmd_scale_[1] = config.cmd_scale[1];
    cmd_scale_[2] = config.cmd_scale[2];

    FeedDog();
    last_control_ms_ = GetCurrentTimeMs();

    if (config.rt_enabled) {
        ros_body_->RegisterRealtimeTick(
            std::bind(&AIBrain::ControlLoopStep, this),
            config.control_rate_ms, config.rt_priority, config.rt_use_mlockall);
    } else {
        ros_body_->RegisterBrainTick(
            std::bind(&AIBrain::ControlLoopStep, this), config.control_rate_ms);
    }
}

AIBrain::~AIBrain() {
    metrics_.current_state.store(RobotState::IDLE);
}

TaskProfile AIBrain::GetTaskProfile(::Robot::TaskType type) {
    switch (type) {
        case ::Robot::NAVIGATE:
            return {1.0f, 1.0f};
        case ::Robot::FOLLOW:
            return {0.6f, 0.8f};
        case ::Robot::PICK:
            return {0.3f, 0.5f};
        case ::Robot::EMERGENCY_STOP:
        case ::Robot::TASK_UNKNOWN:
        default:
            return {0.0f, 0.0f};
    }
}

void AIBrain::ProcessRobotTask(const std::string& task_id,
                                ::Robot::TaskType task_type,
                                const std::string& command,
                                float vel_cmd_x, float vel_cmd_y, float vel_cmd_z) {
    {
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        current_task_id_ = task_id;
        current_command_ = command;
        current_task_type_ = task_type;
        velocity_cmd_[0] = vel_cmd_x;
        velocity_cmd_[1] = vel_cmd_y;
        velocity_cmd_[2] = vel_cmd_z;
    }

    uint64_t gen = ++task_generation_;
    metrics_.current_generation.store(gen);
    metrics_.current_task_type.store(static_cast<int>(task_type));

    if (task_type == ::Robot::EMERGENCY_STOP) {
        metrics_.current_state.store(RobotState::STANDING);
        std::fill(prev_actions_.begin(), prev_actions_.end(), 0.0f);
        policy_model_->ResetHiddenState();
        last_joint_targets_ = default_joint_positions_;
        ros_body_->PublishJointCommand(default_joint_positions_);
        spdlog::warn("[EmergencyStop] task [{}], generation -> {}", task_id, gen);
        FeedDog();
        return;
    }

    metrics_.current_state.store(RobotState::EXECUTING);
    FeedDog();

    spdlog::info("[TaskPreempt] task [{}] type={} gen={} vel=({:.2f},{:.2f},{:.2f})",
                 task_id, ::Robot::TaskType_Name(task_type), gen,
                 vel_cmd_x, vel_cmd_y, vel_cmd_z);
}

void AIBrain::ControlLoopStep() {
    int64_t now = GetCurrentTimeMs();
    if (now - last_heartbeat_ms_.load() > watchdog_timeout_ms_) {
        RobotState cur = metrics_.current_state.load();
        if (cur == RobotState::EXECUTING) {
            spdlog::critical("[WATCHDOG] heartbeat lost > {}ms, fallback to STANDING",
                             watchdog_timeout_ms_);
            metrics_.current_state.store(RobotState::STANDING);
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            velocity_cmd_[0] = 0.0f;
            velocity_cmd_[1] = 0.0f;
            velocity_cmd_[2] = 0.0f;
        }
    }

    RobotState state = metrics_.current_state.load();

    if (state == RobotState::EXECUTING || state == RobotState::STANDING) {
        bool expected = false;
        if (!is_inferring_.compare_exchange_strong(expected, true)) {
            metrics_.dropped_frames++;
            return;
        }

        auto prop = ros_body_->GetLatestProprioception();
        if (!prop.valid) {
            is_inferring_.store(false);
            // 不发布 joint_commands:default 姿态本身不稳定,纯 PD(target=default)会摔倒
            // 让 mujoco_sim 保持 physics_enabled=False,机器人保持 standing reset 姿态
            // 等 prop.valid 后 policy 主动控制平衡
            return;
        }

        // 仿真重置检测:mujoco_sim 倒地重置后,gravity_z 会从 ~0 突变回 ~-1
        // 此时必须清零 LSTM 状态,否则 policy 带着"倒地记忆"推理 → 输出异常
        static float last_gravity_z = -1.0f;
        float cur_gravity_z = prop.projected_gravity[2];
        if (last_gravity_z > -0.3f && cur_gravity_z < -0.7f) {
            spdlog::warn("[RESET] gravity_z {:.2f}->{:.2f}, resetting LSTM + prev_actions",
                         last_gravity_z, cur_gravity_z);
            policy_model_->ResetHiddenState();
            std::fill(prev_actions_.begin(), prev_actions_.end(), 0.0f);
            last_joint_targets_ = default_joint_positions_;
        }
        last_gravity_z = cur_gravity_z;

        // 速度渐变:current_vel_cmd_ 向 target 渐变,STAND->EXECUTING 不跳变
        float dt = (last_control_ms_ > 0)
            ? static_cast<float>(now - last_control_ms_) / 1000.0f
            : 0.02f;
        last_control_ms_ = now;
        if (dt <= 0.0f || dt > 0.5f) dt = 0.02f;
        RampVelocityCommand(dt);

        ::Robot::TaskType my_task_type;
        uint64_t my_generation;
        {
            std::lock_guard<std::mutex> lock(cmd_mtx_);
            my_task_type = current_task_type_;
            my_generation = task_generation_.load();
        }

        TaskProfile profile = (state == RobotState::STANDING)
            ? TaskProfile{1.0f, 1.0f}  // STAND: velocity_scale=1.0 (用 stand_vel), action 不缩放
            : GetTaskProfile(my_task_type);

        // G1 policy 训练时 cmd_init=[0.5, 0, 0],STAND 用同值保持 obs 分布一致
        float scaled_vel_cmd[3];
        if (state == RobotState::STANDING) {
            scaled_vel_cmd[0] = 0.5f;
            scaled_vel_cmd[1] = 0.0f;
            scaled_vel_cmd[2] = 0.0f;
        } else {
            scaled_vel_cmd[0] = current_vel_cmd_[0] * profile.velocity_scale;
            scaled_vel_cmd[1] = current_vel_cmd_[1] * profile.velocity_scale;
            scaled_vel_cmd[2] = current_vel_cmd_[2] * profile.velocity_scale;
        }

        auto future_opt = inference_pool_->enqueue(
            [this, prop, scaled_vel_cmd, my_generation, profile]() {
                auto t0 = std::chrono::steady_clock::now();

                try {
                    auto obs_buffer = tensor_pool_->Acquire();
                    float* obs_ptr = obs_buffer->data();

                    // phase 用 frame 计数,避免构造到首次推理的墙上时钟延迟导致 phase 偏移
                    uint64_t frame = metrics_.total_frames.load();
                    float elapsed = static_cast<float>(frame) * control_dt_;
                    float phase = std::fmod(elapsed, gait_phase_period_) / gait_phase_period_;
                    float sin_phase = std::sin(2.0f * M_PI * phase);
                    float cos_phase = std::cos(2.0f * M_PI * phase);

                    BuildObservation(prop, scaled_vel_cmd,
                                     default_joint_positions_, prev_actions_,
                                     num_joints_,
                                     ang_vel_scale_, dof_pos_scale_, dof_vel_scale_,
                                     cmd_scale_, sin_phase, cos_phase,
                                     obs_ptr);

                    std::vector<float> actions =
                        policy_model_->Forward(obs_ptr, static_cast<size_t>(obs_dim_));

                    // 诊断日志:对比 diagnose_v4.py
                    if (metrics_.total_frames % 50 == 0 || metrics_.total_frames < 3) {
                        spdlog::info("[DIAG] frame={} obs[:3]=[{:.3f},{:.3f},{:.3f}] "
                                     "grav=[{:.3f},{:.3f},{:.3f}] cmd=[{:.3f},{:.3f},{:.3f}] "
                                     "qj[:3]=[{:.3f},{:.3f},{:.3f}] dqj[:3]=[{:.3f},{:.3f},{:.3f}] "
                                     "prev[:3]=[{:.3f},{:.3f},{:.3f}] act[:3]=[{:.3f},{:.3f},{:.3f}]",
                                     metrics_.total_frames,
                                     obs_ptr[0], obs_ptr[1], obs_ptr[2],
                                     obs_ptr[3], obs_ptr[4], obs_ptr[5],
                                     obs_ptr[6], obs_ptr[7], obs_ptr[8],
                                     obs_ptr[9], obs_ptr[10], obs_ptr[11],
                                     obs_ptr[21], obs_ptr[22], obs_ptr[23],
                                     obs_ptr[33], obs_ptr[34], obs_ptr[35],
                                     actions[0], actions[1], actions[2]);
                    }

                    // joint_targets = default + action * action_scale * profile.action_scale
                    std::vector<float> joint_targets(num_joints_);
                    for (int i = 0; i < num_joints_; ++i) {
                        float scaled = actions[i] * action_scale_ * profile.action_scale;
                        if (std::isnan(scaled) || std::isinf(scaled)) scaled = 0.0f;
                        joint_targets[i] = default_joint_positions_[i] + scaled;
                    }

                    // 关节位置限位 (相对 default 的硬限位)
                    for (int i = 0; i < num_joints_; ++i) {
                        float lo = default_joint_positions_[i] - joint_position_limit_;
                        float hi = default_joint_positions_[i] + joint_position_limit_;
                        joint_targets[i] = std::clamp(joint_targets[i], lo, hi);
                    }

                    if (my_generation == task_generation_.load()) {
                        ApplyRateLimit(joint_targets);
                        ros_body_->PublishJointCommand(joint_targets);
                        FeedDog();
                        for (int i = 0; i < num_joints_; ++i) {
                            prev_actions_[i] = actions[i];
                        }
                        last_joint_targets_ = joint_targets;
                        metrics_.total_frames++;
                    } else {
                        spdlog::warn("[StaleResult] generation {} superseded by {}, discarding",
                                     my_generation, task_generation_.load());
                        metrics_.stale_frames++;
                    }

                } catch (const std::exception& e) {
                    spdlog::error("Inference pipeline error: {}", e.what());
                }

                auto t1 = std::chrono::steady_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                metrics_.last_inference_ms.store(elapsed_ms);

                double prev_avg = metrics_.avg_inference_ms.load();
                double new_avg = prev_avg * 0.9 + elapsed_ms * 0.1;
                metrics_.avg_inference_ms.store(new_avg);

                is_inferring_.store(false);
            });

        if (!future_opt.has_value()) {
            spdlog::warn("[Backpressure] inference pool full, dropping frame");
            is_inferring_.store(false);
            metrics_.dropped_frames++;
        }

    } else {
        // IDLE / ERROR:回 default 站立姿态
        ros_body_->PublishJointCommand(default_joint_positions_);
    }
}

void AIBrain::RampVelocityCommand(float dt) {
    float target[3];
    {
        std::lock_guard<std::mutex> lock(cmd_mtx_);
        target[0] = velocity_cmd_[0];
        target[1] = velocity_cmd_[1];
        target[2] = velocity_cmd_[2];
    }
    float max_delta = max_vel_ramp_ * dt;
    for (int i = 0; i < 3; ++i) {
        float delta = target[i] - current_vel_cmd_[i];
        if (std::abs(delta) > max_delta) {
            current_vel_cmd_[i] += std::copysign(max_delta, delta);
        } else {
            current_vel_cmd_[i] = target[i];
        }
    }
}

bool AIBrain::DetectFall(const core::Proprioception& prop) {
    // projected_gravity[2] 站立时 ≈ -1 (重力向下),倒地时 ≈ 0
    // 防抖:连续 N 帧满足条件才确认摔倒,避免行走摇摆误触发
    if (prop.projected_gravity[2] > fall_gravity_threshold_) {
        if (++fall_debounce_count_ >= fall_debounce_frames_) {
            fall_debounce_count_ = 0;
            return true;
        }
    } else {
        fall_debounce_count_ = 0;
    }
    return false;
}

void AIBrain::ApplyRateLimit(std::vector<float>& joint_targets) {
    // 使用控制周期(50Hz=0.02s)而非推理线程与控制线程的时间差,
    // 后者仅几毫秒,会导致限速比预期严数倍,policy 来不及修正关节。
    constexpr float control_dt = 0.02f;
    float max_delta = max_joint_velocity_ * control_dt;

    for (int i = 0; i < num_joints_; ++i) {
        float delta = joint_targets[i] - last_joint_targets_[i];
        if (std::abs(delta) > max_delta) {
            joint_targets[i] = last_joint_targets_[i] + std::copysign(max_delta, delta);
        }
    }
}

} // namespace business
} // namespace engine
