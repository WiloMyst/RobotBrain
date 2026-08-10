#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>
#include <thread>
#include "engine/infra/buffer_pool.hpp"
#include "robot_brain.pb.h"

namespace engine {
    namespace core { class RosNode; struct Proprioception; }
    namespace infra { class ThreadPool; }
    namespace business { namespace models { class PolicyModel; } }
}

namespace engine {
namespace business {

enum class RobotState {
    IDLE,
    EXECUTING,
    ERROR,
    STANDING
};

struct RuntimeMetrics {
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint64_t> dropped_frames{0};
    std::atomic<uint64_t> stale_frames{0};
    std::atomic<double>   last_inference_ms{0.0};
    std::atomic<double>   avg_inference_ms{0.0};
    std::atomic<uint64_t> current_generation{0};
    std::atomic<RobotState> current_state{RobotState::IDLE};
    std::atomic<int> current_task_type{0};
};

/// 任务画像:不同任务类型对应不同的速度指令缩放与动作幅度缩放
struct TaskProfile {
    float velocity_scale;  // 用户速度指令缩放 (NAVIGATE=1.0, FOLLOW=0.6, PICK=0.3, STOP=0)
    float action_scale;    // policy 输出缩放 (进一步限制动作幅度)
};

struct BrainConfig {
    std::string model_path;
    int intra_op_threads = 2;
    bool use_gpu = false;
    int num_joints = 12;
    int obs_dim = 47;
    float action_scale = 0.25f;
    std::vector<float> default_joint_positions;
    int control_rate_ms = 20;
    int64_t watchdog_timeout_ms = 500;
    float max_joint_velocity = 1.0f;
    float joint_position_limit = 0.5f;
    float max_vel_ramp = 2.0f;          // 速度指令渐变率 (m/s per second)
    float fall_gravity_threshold = -0.5f; // 倒地检测:projected_gravity[2] < threshold
    int fall_debounce_frames = 25;        // 防抖:连续 N 帧满足条件才确认摔倒
    float ang_vel_scale = 1.0f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 1.0f;
    float cmd_scale[3] = {1.0f, 1.0f, 1.0f};
    float gait_phase_period = 0.8f;
    bool rt_enabled = true;
    int rt_priority = 80;
    bool rt_use_mlockall = true;
};

class AIBrain {
public:
    explicit AIBrain(const BrainConfig& config, core::RosNode* ros_body);
    ~AIBrain();

    void ProcessRobotTask(const std::string& task_id,
                          ::Robot::TaskType task_type,
                          const std::string& command,
                          float vel_cmd_x, float vel_cmd_y, float vel_cmd_z);

    RuntimeMetrics& GetMetrics() { return metrics_; }

    // 供状态回传:获取当前生效的速度指令(渐变后的实际值)
    void GetCurrentVelocityCommand(float out[3]) {
        out[0] = current_vel_cmd_[0];
        out[1] = current_vel_cmd_[1];
        out[2] = current_vel_cmd_[2];
    }

private:
    void ControlLoopStep();
    void ApplyRateLimit(std::vector<float>& joint_targets);
    // 速度指令渐变 (STAND<->EXECUTING 平滑过渡)
    void RampVelocityCommand(float dt);
    // 倒地检测:基于 projected_gravity 判断
    bool DetectFall(const core::Proprioception& prop);

    static TaskProfile GetTaskProfile(::Robot::TaskType type);

    int64_t GetCurrentTimeMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void FeedDog() {
        last_heartbeat_ms_.store(GetCurrentTimeMs());
    }

    std::unique_ptr<infra::ThreadPool> inference_pool_;
    core::RosNode* ros_body_;
    std::unique_ptr<models::PolicyModel> policy_model_;

    int num_joints_;
    int obs_dim_;
    float action_scale_;
    std::vector<float> default_joint_positions_;
    float max_joint_velocity_;
    float joint_position_limit_;
    float ang_vel_scale_;
    float dof_pos_scale_;
    float dof_vel_scale_;
    float cmd_scale_[3];
    float gait_phase_period_;
    float control_dt_;        // 控制周期 (秒),用于 phase 计数
    float max_vel_ramp_;
    float fall_gravity_threshold_;
    int fall_debounce_frames_;
    int fall_debounce_count_ = 0;
    float current_vel_cmd_[3] = {0.0f, 0.0f, 0.0f};  // 渐变后的实际速度指令

    std::unique_ptr<infra::BufferPool<float>> tensor_pool_;

    std::atomic<int64_t> last_heartbeat_ms_{0};
    int64_t watchdog_timeout_ms_;

    std::atomic<bool> is_inferring_{false};
    std::atomic<uint64_t> task_generation_{0};

    std::mutex cmd_mtx_;
    std::string current_task_id_;
    std::string current_command_;
    ::Robot::TaskType current_task_type_{::Robot::TASK_UNKNOWN};
    float velocity_cmd_[3] = {0.0f, 0.0f, 0.0f};

    std::vector<float> prev_actions_;        // 上一帧 policy 输出 (obs 的 prev_actions 分量)
    std::vector<float> last_joint_targets_;  // 上一帧下发的关节目标 (rate limit 基准)
    int64_t last_control_ms_ = 0;

    RuntimeMetrics metrics_;
};

} // namespace business
} // namespace engine
