#include "engine/core/task_session.h"
#include "engine/business/ai_brain.h"
#include "engine/core/ros_node.h"
#include "engine/infra/thread_pool.hpp"
#include <spdlog/spdlog.h>
#include <chrono>

namespace engine {
namespace core {

TaskSession::TaskSession(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                         infra::ThreadPool* pool, business::AIBrain* brain)
    : service_(svc), cq_(cq), pool_(pool), brain_(brain),
      responder_(&ctx_) {}

void TaskSession::HandleEvent(EventType type, bool ok) {
    if (!ok) {
        self_.reset();
        return;
    }

    switch (type) {
        case EventType::CONNECT: {
            Create(service_, cq_, pool_, brain_);

            const auto& vc = request_.velocity_cmd();
            brain_->ProcessRobotTask(request_.task_id(), request_.task_type(),
                                     request_.command(),
                                     vc.linear_x(), vc.linear_y(), vc.angular_z());

            Robot::TaskResponse response;
            response.set_accepted(true);
            response.set_status_msg("Task accepted, generation updated");

            responder_.Finish(response, grpc::Status::OK, MakeTag(EventType::FINISH));
            break;
        }

        case EventType::FINISH: {
            self_.reset();
            break;
        }

        default:
            spdlog::warn("TaskSession: unexpected event type");
            self_.reset();
            break;
    }
}

StatusSession::StatusSession(AsyncService* svc, grpc::ServerCompletionQueue* cq,
                             business::AIBrain* brain, core::RosNode* ros_node)
    : service_(svc), cq_(cq), brain_(brain), ros_node_(ros_node),
      writer_(&ctx_) {}

void StatusSession::HandleEvent(EventType type, bool ok) {
    if (!ok) {
        self_.reset();
        return;
    }

    switch (type) {
        case EventType::STATUS_READ: {
            Create(service_, cq_, brain_, ros_node_);

            auto status_msg = BuildStatusMessage();
            writer_.Write(status_msg, MakeTag(EventType::STATUS_WRITE));
            break;
        }

        case EventType::STATUS_WRITE: {
            ScheduleNextWrite();
            break;
        }

        case EventType::STATUS_ALARM: {
            if (!finished_) {
                auto status_msg = BuildStatusMessage();
                writer_.Write(status_msg, MakeTag(EventType::STATUS_WRITE));
            }
            break;
        }

        case EventType::STATUS_FINISH: {
            self_.reset();
            break;
        }

        default:
            spdlog::warn("StatusSession: unexpected event type");
            self_.reset();
            break;
    }
}

void StatusSession::ScheduleNextWrite() {
    int interval_ms = request_.interval_ms() > 0 ? request_.interval_ms() : 200;
    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::milliseconds(interval_ms);
    alarm_.Set(cq_, deadline, MakeTag(EventType::STATUS_ALARM));
}

Robot::RobotStatus StatusSession::BuildStatusMessage() {
    auto& m = brain_->GetMetrics();

    Robot::RobotStatus status;

    switch (m.current_state.load()) {
        case business::RobotState::IDLE:
            status.set_state(Robot::RobotStatus::IDLE); break;
        case business::RobotState::EXECUTING:
            status.set_state(Robot::RobotStatus::EXECUTING); break;
        case business::RobotState::ERROR:
            status.set_state(Robot::RobotStatus::ERROR); break;
        case business::RobotState::STANDING:
            status.set_state(Robot::RobotStatus::STANDING); break;
    }

    status.set_task_type(static_cast<::Robot::TaskType>(m.current_task_type.load()));
    status.set_task_generation(m.current_generation.load());
    status.set_total_frames(m.total_frames.load());
    status.set_dropped_frames(m.dropped_frames.load());
    status.set_stale_frames(m.stale_frames.load());
    status.set_last_inference_ms(m.last_inference_ms.load());
    status.set_avg_inference_ms(m.avg_inference_ms.load());

    if (ros_node_) {
        // 当前下发的关节目标
        const auto& targets = ros_node_->GetLastJointTargets();
        for (float v : targets) status.add_joint_targets(v);

        // 最近观测到的关节位置 (反馈)
        auto prop = ros_node_->GetLatestProprioception();
        for (float v : prop.joint_positions) status.add_joint_positions(v);

        // 控制周期 jitter 统计 [p50, p95, p99, max]
        auto jit = ros_node_->GetJitterPercentiles();
        status.set_jitter_p50_ms(jit[0]);
        status.set_jitter_p95_ms(jit[1]);
        status.set_jitter_p99_ms(jit[2]);
        status.set_jitter_max_ms(jit[3]);
    }

    // 当前生效的速度指令
    float vc[3];
    brain_->GetCurrentVelocityCommand(vc);
    auto* vel_cmd = status.mutable_velocity_cmd();
    vel_cmd->set_linear_x(vc[0]);
    vel_cmd->set_linear_y(vc[1]);
    vel_cmd->set_angular_z(vc[2]);

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    status.set_timestamp_ms(now.count());

    return status;
}

} // namespace core
} // namespace engine
