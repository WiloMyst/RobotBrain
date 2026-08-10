#pragma once
#include <string>
#include <memory>
#include <grpcpp/grpcpp.h>

namespace engine {
    namespace infra { class ThreadPool; }
    namespace business { class AIBrain; struct BrainConfig; }
    namespace core { class RosNode; }
}

namespace engine {
namespace core {

class GrpcServer final {
public:
    GrpcServer();
    ~GrpcServer();

    void Run(const std::string& host, int port, int threads,
             int max_queue, const business::BrainConfig& brain_config,
             core::RosNode* ros_node);

private:
    void HandleRpcs();

    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<infra::ThreadPool> pool_;
    std::unique_ptr<business::AIBrain> brain_;
    core::RosNode* ros_node_ = nullptr;
};

} // namespace core
} // namespace engine
