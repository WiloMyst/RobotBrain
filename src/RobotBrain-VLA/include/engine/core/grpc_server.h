#pragma once
#include <string>
#include <memory>
#include <grpcpp/grpcpp.h>

// 前置声明
namespace engine {
    namespace infra { class ThreadPool; }
    namespace business { class AIBrain; }
    namespace core { class RosNode; }
}

namespace engine {
namespace core {

class GrpcServer final {
public:
    GrpcServer();
    ~GrpcServer();

    void Run(const std::string& host, int port, int threads, 
             int max_queue, const std::string& model_path, core::RosNode* ros_node);

private:
    void HandleRpcs();

    // gRPC 核心服务组件
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    struct Impl; 
    std::unique_ptr<Impl> pimpl_; // 使用 Pimpl 模式可以更彻底地隔离协议细节

    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<infra::ThreadPool> pool_;
    std::unique_ptr<business::AIBrain> brain_;
};

} // namespace core
} // namespace engine