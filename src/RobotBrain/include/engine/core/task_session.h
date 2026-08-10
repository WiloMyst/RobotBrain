#pragma once
#include <grpcpp/grpcpp.h>
#include <grpcpp/alarm.h>
#include <atomic>
#include <functional>
#include <memory>
#include "robot_brain.grpc.pb.h"

namespace engine {
    namespace infra { class ThreadPool; }
    namespace business { class AIBrain; }
    namespace core { class RosNode; }
}

namespace engine {
namespace core {

enum class EventType {
    CONNECT,
    READ,
    WRITE,
    FINISH,
    STATUS_READ,
    STATUS_WRITE,
    STATUS_ALARM,
    STATUS_FINISH
};

/// 事件标签:挂在 CompletionQueue 上,回调时用来找到归属的 Session
struct SessionEventTag {
    std::function<void(bool)> callback;
    EventType type;
};

/// 任务指令会话:处理 AssignTask RPC
class TaskSession : public std::enable_shared_from_this<TaskSession> {
public:
    using AsyncService = Robot::BrainService::AsyncService;

    TaskSession(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                infra::ThreadPool* pool, business::AIBrain* brain);

    /// 在 CQ 上注册新的 TaskSession,等待客户端连接
    static void Create(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                       infra::ThreadPool* pool, business::AIBrain* brain) {
        auto session = std::make_shared<TaskSession>(svc, cq, pool, brain);
        // 调用 RequestAssignTask,等待下一个客户端调用 AssignTask RPC
        svc->RequestAssignTask(&session->ctx_, &session->request_,
                               &session->responder_, session->cq_,
                               session->cq_,
                               session->MakeTag(EventType::CONNECT));
        // shared_ptr 托管到 responder_ 完成为止
        session->self_ = session;
    }

    void HandleEvent(EventType type, bool ok);

private:
    SessionEventTag* MakeTag(EventType type) {
        return new SessionEventTag{[this, type](bool ok){ this->HandleEvent(type, ok); }, type};
    }

    AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    infra::ThreadPool* pool_;
    business::AIBrain* brain_;

    grpc::ServerContext ctx_;
    Robot::TaskRequest request_;
    grpc::ServerAsyncResponseWriter<Robot::TaskResponse> responder_;

    std::shared_ptr<TaskSession> self_;  // 自引用,保持生命周期
};

/// 状态流式回传会话:处理 StreamStatus RPC
class StatusSession : public std::enable_shared_from_this<StatusSession> {
public:
    using AsyncService = Robot::BrainService::AsyncService;

    StatusSession(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                  business::AIBrain* brain, core::RosNode* ros_node);

    static void Create(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                       business::AIBrain* brain, core::RosNode* ros_node) {
        auto session = std::make_shared<StatusSession>(svc, cq, brain, ros_node);
        svc->RequestStreamStatus(&session->ctx_, &session->request_,
                                  &session->writer_, session->cq_,
                                  session->cq_,
                                  session->MakeTag(EventType::STATUS_READ));
        session->self_ = session;
    }

    void HandleEvent(EventType type, bool ok);

private:
    SessionEventTag* MakeTag(EventType type) {
        return new SessionEventTag{[this, type](bool ok){ this->HandleEvent(type, ok); }, type};
    }

    Robot::RobotStatus BuildStatusMessage();

    void ScheduleNextWrite();

    AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    business::AIBrain* brain_;
    core::RosNode* ros_node_;

    grpc::ServerContext ctx_;
    Robot::StatusRequest request_;
    grpc::ServerAsyncWriter<Robot::RobotStatus> writer_;

    grpc::Alarm alarm_;
    bool writing_ = false;
    bool finished_ = false;

    std::shared_ptr<StatusSession> self_;
};

} // namespace core
} // namespace engine
