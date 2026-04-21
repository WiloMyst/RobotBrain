#include "engine/core/task_session.h"
#include "engine/infra/thread_pool.hpp"
#include "engine/business/ai_brain.h"
#include <spdlog/spdlog.h>

namespace engine {
namespace core {

void TaskSession::Create(Robot::BrainService::AsyncService* service, grpc::ServerCompletionQueue* cq,
                         infra::ThreadPool* pool, business::AIBrain* brain) {
    // 使用 shared_ptr 管理生命周期，确保异步回调完成前对象不被析构
    auto session = std::shared_ptr<TaskSession>(new TaskSession(service, cq, pool, brain));
    session->Start();
}

TaskSession::TaskSession(Robot::BrainService::AsyncService* service, grpc::ServerCompletionQueue* cq,
                         infra::ThreadPool* pool, business::AIBrain* brain)
    : service_(service), cq_(cq), pool_(pool), brain_(brain), responder_(&ctx_) {}

void TaskSession::Start() {
    auto* tag = new EventTag{shared_from_this(), EventType::PROCESS};
    // 监听云端下发的 AssignTask 指令
    service_->RequestAssignTask(&ctx_, &request_, &responder_, cq_, cq_, tag);
}

void TaskSession::HandleEvent(EventType type, bool ok) {
    if (type == EventType::FINISH || !ok) {
        // 彻底清理：gRPC 挥手结束或连接异常
        return; 
    }

    if (type == EventType::PROCESS) {
        // 1. 立即拉起下一个 Session 监听，保持“大门敞开”
        TaskSession::Create(service_, cq_, pool_, brain_);

        // 2. 将耗时的 AI 推理与 ROS 交互任务丢进线程池
        ProcessRequestAsync();
    }
}

void TaskSession::ProcessRequestAsync() {
    pool_->enqueue([this, self = shared_from_this()]() {
        try {
            // 极其干净！网络层只管传文本，它根本不知道 ROS 的存在
            self->brain_->ProcessRobotTask(self->request_.task_id(), self->request_.command());

            self->response_.set_accepted(true);
            self->response_.set_status_msg("指令已下发至大脑中枢");
        } catch (const std::exception& e) {
            self->response_.set_accepted(false);
            self->response_.set_status_msg(std::string("执行失败: ") + e.what());
        }
        auto* tag = new EventTag{self, EventType::FINISH};
        self->responder_.Finish(self->response_, grpc::Status::OK, tag);
    });
}

} // namespace core
} // namespace engine