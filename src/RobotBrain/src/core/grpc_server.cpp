#include "engine/core/grpc_server.h"
#include "engine/core/task_session.h"
#include "engine/infra/thread_pool.hpp"
#include "engine/business/ai_brain.h"
#include <spdlog/spdlog.h>
#include "robot_brain.grpc.pb.h"

namespace engine {
namespace core {

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
                     int max_queue, const business::BrainConfig& brain_config,
                     core::RosNode* ros_node) {
    ros_node_ = ros_node;
    std::string server_address = host + ":" + std::to_string(port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&(pimpl_->service));

    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();

    if (!server_) {
        spdlog::critical("gRPC server failed to start on {}", server_address);
        return;
    }

    spdlog::info("gRPC server listening on {}", server_address);

    pool_ = std::make_unique<infra::ThreadPool>(threads, max_queue);
    brain_ = std::make_unique<business::AIBrain>(brain_config, ros_node);

    HandleRpcs();
}

void GrpcServer::HandleRpcs() {
    // 启动指令监听器
    TaskSession::Create(&(pimpl_->service), cq_.get(), pool_.get(), brain_.get());
    // 启动状态流监听器
    StatusSession::Create(&(pimpl_->service), cq_.get(), brain_.get(), ros_node_);

    void* raw_tag;
    bool ok;
    while (cq_->Next(&raw_tag, &ok)) {
        auto* tag = static_cast<SessionEventTag*>(raw_tag);
        tag->callback(ok);
        delete tag;
    }
}

} // namespace core
} // namespace engine
