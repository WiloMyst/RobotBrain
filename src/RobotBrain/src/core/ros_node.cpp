#include "engine/core/ros_node.h"
#include <spdlog/spdlog.h>
#include <sched.h>
#include <pthread.h>
#include <ctime>
#include <sys/mman.h>
#include <algorithm>
#include <cerrno>
#include <cstring>

namespace engine {
namespace core {

RosNode::RosNode(const std::string& joint_state_topic,
                 const std::string& imu_topic,
                 const std::string& joint_cmd_topic,
                 int num_joints)
    : Node("edge_brain_node"), num_joints_(num_joints) {

    latest_joint_positions_.assign(num_joints, 0.0f);
    latest_joint_velocities_.assign(num_joints, 0.0f);
    last_joint_targets_.assign(num_joints, 0.0f);

    // 传感器 QoS: Best Effort + 队列深度 1,数据面追求低延迟,允许丢帧但不积压
    auto sensor_qos = rclcpp::SensorDataQoS();
    sensor_qos.keep_last(1);

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        joint_state_topic, sensor_qos,
        std::bind(&RosNode::JointStateCallback, this, std::placeholders::_1)
    );

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, sensor_qos,
        std::bind(&RosNode::ImuCallback, this, std::placeholders::_1)
    );

    // 关节目标指令用 Reliable 传输,控制指令不能丢
    cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(joint_cmd_topic, 10);

    RCLCPP_INFO(this->get_logger(),
                "RosNode ready [joint_state: %s, imu: %s, cmd: %s, joints: %d, QoS: BestEffort/depth=1]",
                joint_state_topic.c_str(), imu_topic.c_str(),
                joint_cmd_topic.c_str(), num_joints);
}

RosNode::~RosNode() {
    if (rt_running_.load()) {
        rt_running_.store(false);
        if (rt_thread_.joinable()) {
            rt_thread_.join();
        }
    }
}

void RosNode::RegisterRealtimeTick(std::function<void()> tick_cb, int rate_ms,
                                   int priority, bool use_mlockall) {
    brain_tick_cb_ = std::move(tick_cb);
    rt_running_.store(true);

    rt_thread_ = std::thread([this, rate_ms, priority, use_mlockall]() {
        // mlockall: 锁页内存,防止 swap 导致缺页中断
        if (use_mlockall) {
            if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
                RCLCPP_INFO(this->get_logger(), "RT: mlockall ok (pages locked)");
            } else {
                RCLCPP_WARN(this->get_logger(),
                    "RT: mlockall failed: %s (needs root/CAP_IPC_LOCK, continuing without)",
                    std::strerror(errno));
            }
        }

        // SCHED_FIFO: 实时调度策略,最高优先级
        struct sched_param sp;
        sp.sched_priority = priority;
        bool fifo_ok = false;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0) {
            fifo_ok = true;
            RCLCPP_INFO(this->get_logger(), "RT: SCHED_FIFO priority=%d", priority);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "RT: SCHED_FIFO failed: %s (needs root/CAP_SYS_NICE, fallback SCHED_OTHER)",
                std::strerror(errno));
        }
        {
            std::lock_guard<std::mutex> lock(jitter_stats_.mtx);
            jitter_stats_.sched_fifo_ok = fifo_ok;
        }

        // clock_nanosleep 精确定时:用绝对时间,消除累积误差
        struct timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);
        int64_t period_ns = static_cast<int64_t>(rate_ms) * 1000000L;

        while (rt_running_.load()) {
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);

            // jitter = 实际唤醒时间 - 预期唤醒时间
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t actual_ns = static_cast<int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec;
            int64_t expected_ns = static_cast<int64_t>(next.tv_sec) * 1000000000LL + next.tv_nsec;
            double jitter_ms = static_cast<double>(actual_ns - expected_ns) / 1000000.0;
            if (jitter_ms < 0.0) jitter_ms = 0.0;
            UpdateJitter(jitter_ms);

            if (brain_tick_cb_) brain_tick_cb_();

            next.tv_nsec += period_ns;
            while (next.tv_nsec >= 1000000000L) {
                next.tv_nsec -= 1000000000L;
                next.tv_sec++;
            }
        }
    });

    RCLCPP_INFO(this->get_logger(),
        "RT thread started [rate=%dms, prio=%d, mlockall=%d]",
        rate_ms, priority, use_mlockall ? 1 : 0);
}

void RosNode::UpdateJitter(double jitter_ms) {
    std::lock_guard<std::mutex> lock(jitter_stats_.mtx);
    jitter_stats_.ring_buffer[jitter_stats_.ring_idx] = jitter_ms;
    jitter_stats_.ring_idx = (jitter_stats_.ring_idx + 1) % JitterStats::WINDOW;
    if (jitter_stats_.ring_count < JitterStats::WINDOW) {
        jitter_stats_.ring_count++;
    }
    if (jitter_ms > jitter_stats_.max_ms) {
        jitter_stats_.max_ms = jitter_ms;
    }
}

std::array<double, 4> RosNode::GetJitterPercentiles() {
    std::lock_guard<std::mutex> lock(jitter_stats_.mtx);
    if (jitter_stats_.ring_count == 0) {
        return {0.0, 0.0, 0.0, 0.0};
    }

    std::vector<double> sorted(
        jitter_stats_.ring_buffer.begin(),
        jitter_stats_.ring_buffer.begin() + jitter_stats_.ring_count);
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    return {
        sorted[n * 50 / 100],
        sorted[n * 95 / 100],
        sorted[n * 99 / 100],
        jitter_stats_.max_ms
    };
}

Proprioception RosNode::GetLatestProprioception() {
    std::lock_guard<std::mutex> lock(proprio_mtx_);
    Proprioception p;
    p.joint_positions = latest_joint_positions_;
    p.joint_velocities = latest_joint_velocities_;
    p.angular_velocity[0] = latest_ang_vel_[0];
    p.angular_velocity[1] = latest_ang_vel_[1];
    p.angular_velocity[2] = latest_ang_vel_[2];
    p.projected_gravity[0] = latest_gravity_[0];
    p.projected_gravity[1] = latest_gravity_[1];
    p.projected_gravity[2] = latest_gravity_[2];
    p.valid = has_joint_state_ && has_imu_;
    return p;
}

void RosNode::PublishJointCommand(const std::vector<float>& joint_targets) {
    std_msgs::msg::Float64MultiArray msg;
    // Float64MultiArray::data 是 vector<double>,需显式转换 float->double
    msg.data.assign(joint_targets.begin(), joint_targets.end());
    cmd_pub_->publish(msg);
    last_joint_targets_ = joint_targets;
}

void RosNode::RegisterBrainTick(std::function<void()> tick_cb, int rate_ms) {
    brain_tick_cb_ = std::move(tick_cb);
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(rate_ms),
        [this]() {
            if (brain_tick_cb_) brain_tick_cb_();
        }
    );
    RCLCPP_INFO(this->get_logger(),
                "Brain tick registered at %d ms (%.1f Hz)", rate_ms, 1000.0 / rate_ms);
}

void RosNode::JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(proprio_mtx_);
    size_t n = msg->position.size();
    if (n > msg->velocity.size()) n = msg->velocity.size();
    if (n > static_cast<size_t>(num_joints_)) n = static_cast<size_t>(num_joints_);
    for (size_t i = 0; i < n; ++i) {
        latest_joint_positions_[i] = msg->position[i];
        latest_joint_velocities_[i] = msg->velocity[i];
    }
    has_joint_state_ = true;
}

void RosNode::ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(proprio_mtx_);
    latest_ang_vel_[0] = msg->angular_velocity.x;
    latest_ang_vel_[1] = msg->angular_velocity.y;
    latest_ang_vel_[2] = msg->angular_velocity.z;
    QuaternionToProjectedGravity(msg->orientation, latest_gravity_);
    has_imu_ = true;
    static int imu_log_count = 0;
    if (imu_log_count < 5) {
        imu_log_count++;
        RCLCPP_INFO(this->get_logger(),
            "[IMU_CB #%d] orient=[w=%.3f x=%.3f y=%.3f z=%.3f] ang_vel=[%.3f %.3f %.3f] grav=[%.3f %.3f %.3f]",
            imu_log_count, msg->orientation.w, msg->orientation.x,
            msg->orientation.y, msg->orientation.z,
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
            latest_gravity_[0], latest_gravity_[1], latest_gravity_[2]);
    }
}

void RosNode::QuaternionToProjectedGravity(const geometry_msgs::msg::Quaternion& q,
                                            float out[3]) {
    // projected_gravity = R_world_to_base * [0, 0, -1] = -R 第三行
    // R 为 base->world 旋转矩阵,由四元数 q=(w,x,y,z) 构造
    float qw = q.w, qx = q.x, qy = q.y, qz = q.z;
    out[0] =  2.0f * (qw * qy - qx * qz);
    out[1] = -2.0f * (qw * qx + qy * qz);
    out[2] =  2.0f * (qx * qx + qy * qy) - 1.0f;
}

} // namespace core
} // namespace engine
