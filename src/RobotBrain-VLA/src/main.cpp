#include <rclcpp/rclcpp.hpp>
#include <spdlog/spdlog.h>
#include <thread>
#include "engine/core/ros_node.h"
#include "engine/core/grpc_server.h"
#include "engine/infra/logger_setup.hpp"

int main(int argc, char** argv) {
    // 1. 初始化日志与系统环境
    engine::infra::InitLogger();
    setenv("http_proxy", "", 1);
    setenv("https_proxy", "", 1);

    spdlog::info("====================================");
    spdlog::info(" VLA 具身智能边缘大模型节点 启动程序");
    spdlog::info("====================================");

    // 2. 初始化 ROS2 物理控制引擎
    rclcpp::init(argc, argv);
    auto ros_node = std::make_shared<engine::core::RosNode>();

    // 3. 极其关键：将 ROS2 的事件循环放在独立后台线程！
    // 这样它才能源源不断地以 30FPS 接收摄像头数据，且绝对不阻塞主线程。
    std::thread ros_spin_thread([&ros_node]() {
        // 使用多线程执行器，允许 ROS 底层使用多核处理传感器海量并发数据
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(ros_node);
        spdlog::info("ROS2 物理中枢已挂载 [MultiThreadedExecutor]");
        executor.spin();
    });

    try {
        // 真正加载企业级配置
        auto config = engine::infra::LoadConfig("../config.yaml");

        engine::core::GrpcServer server;
        int max_queue_size = 1000;

        // 将读取到的配置注入引擎！
        server.Run(config.host, config.port, config.worker_threads, 
                   max_queue_size, config.vla_model_path, ros_node.get());

    } catch (const std::exception& e) {
        spdlog::critical("核心引擎遭遇致命错误崩毁: {}", e.what());
    }

    // 5. 优雅关机清理
    rclcpp::shutdown();
    if (ros_spin_thread.joinable()) {
        ros_spin_thread.join();
    }

    return 0;
}