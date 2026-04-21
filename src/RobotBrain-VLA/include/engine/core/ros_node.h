#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <functional>

namespace engine {
namespace core {

class RosNode : public rclcpp::Node {
public:
    RosNode();
    ~RosNode() override = default;

    // 核心暴露接口：供 AI 大脑抓取“世界快照”
    sensor_msgs::msg::Image::SharedPtr GetLatestImage();

    // 核心暴露接口：供 AI 大脑下发“肌肉电流”
    void PublishControl(float linear_x, float angular_z);

    // 新增时钟挂载接口，允许外部业务中枢将自己的 Tick 函数注册为 ROS 原生定时器
    void RegisterBrainTick(std::function<void()> tick_cb, int rate_ms);

private:
    // ROS2 底层回调函数
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    
    sensor_msgs::msg::Image::SharedPtr latest_img_;

    // 定时器与回调函数
    rclcpp::TimerBase::SharedPtr control_timer_;
    std::function<void()> brain_tick_cb_;
};

} // namespace core
} // namespace engine