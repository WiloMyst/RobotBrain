#include "engine/core/ros_node.h"
#include <atomic>

namespace engine {
namespace core {

RosNode::RosNode() : Node("vla_edge_brain_node") {
    // =================================================================
    // 【工业级改造】：传感器专属 QoS (Quality of Service)
    // 默认的 QoS 是 Reliable (可靠，底层类似 TCP，会阻塞重传)。
    // 这里我们强制启用 SensorDataQoS (Best Effort，底层类似 UDP)。
    // =================================================================
    auto sensor_qos = rclcpp::SensorDataQoS();
    
    // 极其暴躁且专业的设置：队列深度设为 1！
    // 既然 AI 大脑算力有限（比如只能跑 10 FPS），而摄像头是 30 FPS。
    // 中间多出来的 2 帧，我们连缓都不缓存，直接在 DDS 底层网络层无情丢弃！
    // 保证 GetLatestImage() 永远只拿此时此刻绝对新鲜的第一手画面。
    sensor_qos.keep_last(1); 

    // 1. 订阅视觉流 (注入 QoS)
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", sensor_qos,
        std::bind(&RosNode::ImageCallback, this, std::placeholders::_1)
    );

    // 2. 发布控制流
    // 指令数据极小（几个 float），且不能随意丢失（比如刹车指令），
    // 所以控制流保留默认的 Reliable (可靠传输)，队列深度给 10 即可。
    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    
    RCLCPP_INFO(this->get_logger(), "ROS2 物理控制中枢就绪！ [视觉 QoS: BestEffort/极低延迟]");
}

sensor_msgs::msg::Image::SharedPtr RosNode::GetLatestImage() {
    // 绝对无锁！O(1) 极速读取
    return std::atomic_load(&latest_img_);
}

void RosNode::PublishControl(float linear_x, float angular_z) {
    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = linear_x;
    msg.angular.z = angular_z;
    cmd_pub_->publish(msg);
    
    RCLCPP_INFO(this->get_logger(), ">> 下发底盘扭矩 | v: %.2f m/s, w: %.2f rad/s", linear_x, angular_z);
}

// 新增时钟挂载接口，允许外部业务中枢将自己的 Tick 函数注册为 ROS 原生定时器
void RosNode::RegisterBrainTick(std::function<void()> tick_cb, int rate_ms) {
    brain_tick_cb_ = tick_cb;
    // 创建 ROS 原生 WallTimer
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(rate_ms),
        [this]() { 
            if (brain_tick_cb_) brain_tick_cb_(); 
        }
    );
    RCLCPP_INFO(this->get_logger(), "已挂载外部中枢心跳，频率: %d ms", rate_ms);
}

void RosNode::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    // 绝对无锁！O(1) 极速写入。旧的 shared_ptr 如果没人用，会自动在后台析构，绝不阻塞当前 I/O 线程！
    std::atomic_store(&latest_img_, msg);
}

} // namespace core
} // namespace engine