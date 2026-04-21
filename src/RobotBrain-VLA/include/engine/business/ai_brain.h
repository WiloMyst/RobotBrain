#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "engine/infra/buffer_pool.hpp"

namespace engine {
    namespace core { class RosNode; } 
    namespace infra { class ThreadPool; }
}

namespace engine {
namespace business {

// 1. 定义工业级状态机枚举
enum class RobotState {
    IDLE,           // 待机中，安全锁死底盘
    EXECUTING,      // 正在执行云端任务
    ERROR           // 软硬件故障，紧急刹车
};

class AIBrain {
public:
    explicit AIBrain(const std::string& vla_model_path, core::RosNode* ros_body);
    ~AIBrain();

    // gRPC 调用的入口，现在只负责“修改状态”，极速返回
    void ProcessRobotTask(const std::string& task_id, const std::string& command);

private:
    void ControlLoopStep();
    
    // 获取当前时间戳的内联辅助函数
    int64_t GetCurrentTimeMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    
    // 喂狗操作
    void FeedDog() {
        last_heartbeat_ms_.store(GetCurrentTimeMs());
    }

    std::unique_ptr<infra::ThreadPool> inference_pool_;
    core::RosNode* ros_body_; 

    // 专门用于 VLA 模型视觉张量的内存池
    // 假设模型输入固定为 224x224，3 个通道
    static constexpr int VLA_INPUT_H = 224;
    static constexpr int VLA_INPUT_W = 224;
    static constexpr int VLA_CHANNELS = 3;
    static constexpr size_t TENSOR_SIZE = VLA_CHANNELS * VLA_INPUT_H * VLA_INPUT_W;
    // 预分配 3 块张量内存，供异步推理流水线轮转使用
    std::unique_ptr<infra::BufferPool<float>> tensor_pool_;

    // 软件看门狗 (Watchdog) 核心基建
    std::atomic<int64_t> last_heartbeat_ms_{0};
    // 致命阈值：500毫秒！只要超过半秒没有大脑指令，系统立刻熔断！
    static constexpr int64_t WATCHDOG_TIMEOUT_MS = 500;

    // 3. 状态机与并发控制基建 (极致的原子操作)
    std::atomic<RobotState> current_state_{RobotState::IDLE};
    std::atomic<bool> is_inferring_{false};    // 极其关键：防止 AI 队列积压的防洪闸门！

    // 【企业级基建】：任务世代号（Generation ID）
    // 任何改变系统状态的新指令到来，世代号直接 +1。
    // 这是一把极其轻量级的“逻辑锁”，开销几乎为 0。
    std::atomic<uint64_t> task_generation_{0};

    // 保护当前指令的锁
    std::mutex cmd_mtx_;
    std::string current_task_id_;
    std::string current_command_;

    std::unique_ptr<models::VLAModel> vla_model_;
};

} // namespace business
} // namespace engine