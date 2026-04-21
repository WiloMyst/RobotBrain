#include "engine/business/ai_brain.h"
#include "engine/infra/thread_pool.hpp"
#include "engine/core/ros_node.h"
#include "engine/business/models/vla_model.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <opencv2/dnn.hpp>

namespace engine {
namespace business {

AIBrain::AIBrain(const std::string& vla_model_path, core::RosNode* ros_body) 
    : ros_body_(ros_body) {
    spdlog::info("AI 大脑初始化: 准备挂载 VLA 模型...");

    vla_model_ = std::make_unique<models::VLAModel>(vla_model_path, 2, true); // 使用 2 线程, 开启 GPU
    
    // 初始化推理线程池与控制线程 (保持不变)
    inference_pool_ = std::make_unique<infra::ThreadPool>(1, 10); 
    
    // 极其关键：初始化张量内存池，彻底消灭运行时的 Heap Allocation
    tensor_pool_ = std::make_unique<infra::BufferPool<float>>(3, TENSOR_SIZE);
    
    FeedDog(); // 启动时先喂一次，防止一上来就误杀
    // 不再自己 new std::thread，而是向 ROS 注册生命周期！锁定 100ms (10Hz) 黄金控制频率
    ros_body_->RegisterBrainTick(std::bind(&AIBrain::ControlLoopStep, this), 100);
}

AIBrain::~AIBrain() {
    current_state_ = RobotState::IDLE;
}

// 1. gRPC 网络层喂狗
void AIBrain::ProcessRobotTask(const std::string& task_id, const std::string& command) {
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    current_task_id_ = task_id;
    current_command_ = command;
    current_state_ = RobotState::EXECUTING;

    // 新指令到来，世代号递增
    // 无论后台还在算什么，它的身份证瞬间变废纸。
    task_generation_++;
    
    FeedDog(); // 网络层来指令了，喂狗！
    spdlog::info("[指令抢占] 接收到云端新指令 [{}], 任务世代号跃升至: {}", task_id, task_generation_.load());
}

// =================================================================
// 控制面：被 gRPC 线程调用，O(1) 复杂度，绝不阻塞网络！
// =================================================================
void AIBrain::ProcessRobotTask(const std::string& task_id, const std::string& command) {
    std::lock_guard<std::mutex> lock(cmd_mtx_);
    current_task_id_ = task_id;
    current_command_ = command;
    
    // 切换状态机，激活控制循环
    current_state_ = RobotState::EXECUTING;
    spdlog::info("[状态机变更] 接收到云端强指令 [{}], 状态切换至 EXECUTING", task_id);
}

// =================================================================
// 数据面与执行面
// =================================================================
// 这个函数每次被调用，就代表绝对精准的 100ms 过去了
void AIBrain::ControlLoopStep() {
    // 终极防线：看门狗死亡拦截逻辑
    int64_t current_time = GetCurrentTimeMs();
    if (current_time - last_heartbeat_ms_.load() > WATCHDOG_TIMEOUT_MS) {
        if (current_state_.load() == RobotState::EXECUTING) {
            spdlog::critical(" [WATCHDOG FATAL] 大脑心跳丢失超过 {} ms！系统触发紧急物理熔断！", WATCHDOG_TIMEOUT_MS);
            current_state_.store(RobotState::ERROR);
        }
    }

    RobotState state = current_state_.load();

    if (state == RobotState::EXECUTING) {
        // 1. 检查防洪闸门：如果上一帧还没算完，直接丢弃本周期的图像（Drop Frame）
        // 硬件级原子操作 CAS (Compare-And-Swap)
        // 语义：如果 is_inferring_ 等于 expected (false)，则瞬间将其改为 true，并返回 true。
        // 这个动作在 CPU 寄存器层面是绝对锁死的，任何其他线程绝对插不进来。
        bool expected = false;
        if (is_inferring_.compare_exchange_strong(expected, true)) {
            
            auto img = ros_body_->GetLatestImage();
            if (img) {
                std::string cmd;
                uint64_t my_generation;
                {
                    std::lock_guard<std::mutex> lock(cmd_mtx_);
                    cmd = current_command_;
                    my_generation = task_generation_.load(); 
                }
                
                // 将极其耗时的推理任务丢进后台线程，主控线程立刻解脱
                auto future_opt = inference_pool_->enqueue([this, img, cmd, my_generation]() {
                    try {
                        // 1. 零拷贝挂载 ROS 图像到 OpenCV Mat (假设输入是 BGR 或 RGB)
                        cv::Mat raw_img(img->height, img->width, CV_8UC3, const_cast<uint8_t*>(img->data.data()));

                        // 2. 魔法时刻：向内存池借用一块固定大小的 float 内存
                        auto tensor_buffer = tensor_pool_->Acquire();
                        float* dst_ptr = tensor_buffer->data();

                        // 3. 【神级零拷贝包装】：用一个 4 维的 cv::Mat 壳子，套住你的物理内存池！
                        // 告诉 OpenCV：一会儿算完，直接把结果写回我的 dst_ptr 里，不要自己去 new 内存！
                        int sizes[] = {1, VLA_CHANNELS, VLA_INPUT_H, VLA_INPUT_W};
                        cv::Mat blob_wrapper(4, sizes, CV_32F, dst_ptr);

                        // 4. 【性能核武启动】：高级指令集一站式预处理
                        // 它会同时利用 AVX2/NEON 指令集完成：Resize + 像素提取 + HWC转CHW + 归一化
                        cv::dnn::blobFromImage(
                            raw_img,                     // 输入：原始高清图像
                            blob_wrapper,                // 输出：直接写入我们的内存池包装壳！
                            1.0 / 255.0,                 // scale：瞬间完成像素归一化
                            cv::Size(VLA_INPUT_W, VLA_INPUT_H), // size：瞬间完成双线性插值缩放
                            cv::Scalar(0, 0, 0),         // mean：这里不减均值，依你模型而定
                            false,                       // swapRB：是否需要 BGR 转 RGB
                            false,                       // crop：是否中心裁剪
                            CV_32F                       // ddepth：输出深度
                        );

                        // 5. 调用 VLA 模型进行端到端推理
                        std::vector<float> actions = vla_model_->Forward(dst_ptr, TENSOR_SIZE);

                        // 此时 actions[0] 是线速度，actions[1] 是角速度
                        float cmd_linear_x = actions[0];
                        float cmd_angular_z = actions[1];

                        if (my_generation == task_generation_.load()) {
                            // 6. 物理输出：通过 ROS2 DDS 网络广播底层硬件电流指令
                            ros_body_->PublishControl(cmd_linear_x, cmd_angular_z); 
                            FeedDog(); 
                        } else {
                            // 世代号变了！说明 gRPC 刚才收到了新指令，且触发了 task_generation_++
                            spdlog::warn("[拦截] 世代号 {} 的算力结果已过期 (当前世代 {})，安全丢弃！", 
                                         my_generation, task_generation_.load());
                            // 静默丢弃，什么也不发，ROS 底层会在下一个 100ms 立刻带着新指令重新开始计算！
                        }

                    } catch (const std::exception& e) {
                        spdlog::error("推理管道崩溃: {}", e.what());
                    }
                    
                    // 【开闸释放】：无论推理成功还是异常崩溃，离开线程前绝对要释放闸门！
                    is_inferring_.store(false);
                });

                // 【背压保护与锁归还】：如果线程池满了，任务没塞进去！
                if (!future_opt.has_value()) {
                    spdlog::warn("推理引擎负载达到极限，丢弃当前帧！");
                    is_inferring_.store(false); // 必须归还状态，否则永久死锁
                }

            } else {
                // IDLE 或 ERROR 状态：彻底锁死底盘！
                is_inferring_.store(false); // 必须归还状态
                ros_body_->PublishControl(0.0f, 0.0f); // 没图像，底盘必须刹车！
            }
        }
    } else {
        // IDLE 或 ERROR 状态：无论发生什么，每秒 10 次疯狂下发 0 速度，彻底锁死底盘！
        ros_body_->PublishControl(0.0f, 0.0f);
    }
}

} // namespace business
} // namespace engine