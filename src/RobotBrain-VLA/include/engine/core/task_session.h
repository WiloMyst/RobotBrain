#pragma once
#include <memory>
#include <grpcpp/grpcpp.h>
#include "robot_brain.grpc.pb.h" // 引入新的协议头

namespace engine {
    namespace infra { class ThreadPool; }
    namespace business { class AIBrain; }
}

namespace engine {
namespace core {

// 处理云端下发任务的异步 Session
class TaskSession : public std::enable_shared_from_this<TaskSession> {
public:
    enum class EventType { PROCESS, FINISH };

    struct EventTag {
        std::shared_ptr<TaskSession> instance;
        EventType type;
    };

    static void Create(Robot::BrainService::AsyncService* service, grpc::ServerCompletionQueue* cq,
                       infra::ThreadPool* pool, business::AIBrain* brain);

    void HandleEvent(EventType type, bool ok);

private:
    TaskSession(Robot::BrainService::AsyncService* service, grpc::ServerCompletionQueue* cq,
                infra::ThreadPool* pool, business::AIBrain* brain);

    void Start();
    void ProcessRequestAsync();

    Robot::BrainService::AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    infra::ThreadPool* pool_;
    business::AIBrain* brain_;

    grpc::ServerContext ctx_;
    Robot::TaskRequest request_;
    Robot::TaskResponse response_;
    grpc::ServerAsyncResponseWriter<Robot::TaskResponse> responder_;
};

} // namespace core
} // namespace engine