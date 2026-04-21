#include "engine/core/grpc_server.h"
#include "engine/core/task_session.h"
#include "engine/infra/thread_pool.hpp"
#include "engine/business/ai_brain.h"
#include <spdlog/spdlog.h>

// 引入 gRPC 自动生成的服务接口
#include "robot_brain.grpc.pb.h" 

namespace engine {
namespace core {

// 真正的 Pimpl 实现体，包裹了 gRPC 的 Service
struct GrpcServer::Impl {
    Robot::BrainService::AsyncService service;
};

GrpcServer::GrpcServer() : pimpl_(std::make_unique<Impl>()) {}

GrpcServer::~GrpcServer() {
    if (server_) {
        server_->Shutdown();
    }
    if (cq_) {
        cq_->Shutdown();
        void* ignored_tag;
        bool ignored_ok;
        while (cq_->Next(&ignored_tag, &ignored_ok)) {}
    }
}

void GrpcServer::Run(const std::string& host, int port, int threads, 
                     int max_queue, const std::string& model_path, core::RosNode* ros_node) {
    std::string server_address = host + ":" + std::to_string(port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&(pimpl_->service));
    
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();

    if (!server_) {
        spdlog::critical("机器人大脑 gRPC 端口启动失败！地址: {}", server_address);
        return;
    }

    spdlog::info("云端控制接口已就绪: {}", server_address);

    // 初始化核心资源
    pool_ = std::make_unique<infra::ThreadPool>(threads, max_queue);
    brain_ = std::make_unique<business::AIBrain>(model_path, ros_node);

    // 进入主循环处理 RPC 事件
    HandleRpcs();
}

void GrpcServer::HandleRpcs() {
    // 启动第一个监听器
    TaskSession::Create(&(pimpl_->service), cq_.get(), pool_.get(), brain_.get());

    void* raw_tag;
    bool ok;
    while (cq_->Next(&raw_tag, &ok)) {
        TaskSession::EventTag* tag = static_cast<TaskSession::EventTag*>(raw_tag);
        tag->instance->HandleEvent(tag->type, ok);
        delete tag; 
    }
}

} // namespace core
} // namespace engine