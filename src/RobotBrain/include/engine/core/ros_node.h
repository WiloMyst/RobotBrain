#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <functional>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <array>

namespace engine {
namespace core {

/// 本体感知数据 (由 JointState + IMU 合成)
/// 对齐 unitree_rl_gym 的 G1 观测格式
struct Proprioception {
    std::vector<float> joint_positions;    // num_joints,绝对位置
    std::vector<float> joint_velocities;   // num_joints
    float angular_velocity[3] = {0, 0, 0}; // base 角速度 (IMU, body frame)
    float projected_gravity[3] = {0, 0, -1}; // 重力在 base frame 的投影
    bool valid = false;                    // joint_state 与 IMU 均到达后置 true
};

/// 控制周期 jitter 统计 (实时线程采集,状态流上报时读取)
struct JitterStats {
    static constexpr size_t WINDOW = 1000;
    std::mutex mtx;
    std::array<double, WINDOW> ring_buffer{};
    size_t ring_idx = 0;
    size_t ring_count = 0;
    double max_ms = 0.0;
    bool sched_fifo_ok = false;
};

class RosNode : public rclcpp::Node {
public:
    RosNode(const std::string& joint_state_topic = "/joint_states",
            const std::string& imu_topic = "/imu",
            const std::string& joint_cmd_topic = "/joint_commands",
            int num_joints = 12);
    ~RosNode() override;

    /// 获取最新本体感知 (加锁拷贝,50Hz 控制频率下 mutex 开销可忽略)
    Proprioception GetLatestProprioception();

    /// 发布关节目标位置 (12 维绝对目标)
    void PublishJointCommand(const std::vector<float>& joint_targets);

    /// 注册控制循环回调,由 ROS WallTimer 驱动 (软实时回退模式)
    void RegisterBrainTick(std::function<void()> tick_cb, int rate_ms);

    /// 注册实时控制线程:SCHED_FIFO + clock_nanosleep 精确定时 + jitter 测量
    void RegisterRealtimeTick(std::function<void()> tick_cb, int rate_ms,
                              int priority, bool use_mlockall);

    /// 获取 jitter 统计 [p50, p95, p99, max] (ms),供状态流上报
    std::array<double, 4> GetJitterPercentiles();

    /// 最近下发的关节目标 (供状态回传)
    const std::vector<float>& GetLastJointTargets() const { return last_joint_targets_; }

private:
    void JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // 从 IMU 四元数计算 projected gravity (g_world=[0,0,-1] 投影到 base frame)
    static void QuaternionToProjectedGravity(const geometry_msgs::msg::Quaternion& q,
                                              float out[3]);

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;

    rclcpp::TimerBase::SharedPtr control_timer_;
    std::function<void()> brain_tick_cb_;

    // 实时线程
    std::thread rt_thread_;
    std::atomic<bool> rt_running_{false};
    JitterStats jitter_stats_;
    void UpdateJitter(double jitter_ms);

    int num_joints_;

    // 本体感知缓存:JointState 与 IMU 两个话题独立到达,合并时加锁
    std::mutex proprio_mtx_;
    std::vector<float> latest_joint_positions_;
    std::vector<float> latest_joint_velocities_;
    float latest_ang_vel_[3] = {0, 0, 0};
    float latest_gravity_[3] = {0, 0, -1};
    bool has_joint_state_ = false;
    bool has_imu_ = false;

    std::vector<float> last_joint_targets_;
};

} // namespace core
} // namespace engine
