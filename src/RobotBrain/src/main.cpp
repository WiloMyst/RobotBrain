#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <spdlog/spdlog.h>
#include <csignal>
#include <thread>
#include <atomic>
#include "engine/core/ros_node.h"
#include "engine/core/grpc_server.h"
#include "engine/business/ai_brain.h"
#include "engine/infra/logger_setup.hpp"
#include "engine/infra/config_manager.hpp"

static std::atomic<bool> g_shutdown_requested{false};

void SignalHandler(int signal) {
    spdlog::info("Received signal {}, initiating graceful shutdown...", signal);
    g_shutdown_requested.store(true);
}

int main(int argc, char** argv) {
    engine::infra::InitLogger();
    setenv("http_proxy", "", 1);
    setenv("https_proxy", "", 1);

    spdlog::info("====================================");
    spdlog::info(" RobotBrain Edge Node v3.0");
    spdlog::info(" RL Locomotion Policy Deployment");
    spdlog::info("====================================");

    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    rclcpp::init(argc, argv);

    std::shared_ptr<engine::core::RosNode> ros_node;
    std::thread ros_spin_thread;

    try {
        std::string pkg_share = ament_index_cpp::get_package_share_directory("robot_brain");
        auto config = engine::infra::LoadConfig(pkg_share + "/config.yaml");

        if (!config.policy_model_path.empty() && config.policy_model_path[0] != '/') {
            config.policy_model_path = pkg_share + "/" + config.policy_model_path;
        }

        ros_node = std::make_shared<engine::core::RosNode>(
            config.joint_state_topic, config.imu_topic,
            config.joint_command_topic, config.num_joints);

        ros_spin_thread = std::thread([&ros_node]() {
            rclcpp::executors::MultiThreadedExecutor executor;
            executor.add_node(ros_node);
            spdlog::info("ROS2 executor started [MultiThreadedExecutor]");
            executor.spin();
        });

        engine::business::BrainConfig brain_config;
        brain_config.model_path = config.policy_model_path;
        brain_config.intra_op_threads = config.intra_op_threads;
        brain_config.use_gpu = config.use_gpu;
        brain_config.num_joints = config.num_joints;
        brain_config.obs_dim = config.obs_dim;
        brain_config.action_scale = config.action_scale;
        brain_config.default_joint_positions = config.default_joint_positions;
        brain_config.control_rate_ms = config.control_rate_ms;
        brain_config.watchdog_timeout_ms = config.watchdog_timeout_ms;
        brain_config.max_joint_velocity = config.max_joint_velocity;
        brain_config.joint_position_limit = config.joint_position_limit;
        brain_config.max_vel_ramp = config.max_vel_ramp;
        brain_config.fall_gravity_threshold = config.fall_gravity_threshold;
        brain_config.fall_debounce_frames = config.fall_debounce_frames;
        brain_config.ang_vel_scale = config.ang_vel_scale;
        brain_config.dof_pos_scale = config.dof_pos_scale;
        brain_config.dof_vel_scale = config.dof_vel_scale;
        brain_config.cmd_scale[0] = config.cmd_scale.size() > 0 ? config.cmd_scale[0] : 1.0f;
        brain_config.cmd_scale[1] = config.cmd_scale.size() > 1 ? config.cmd_scale[1] : 1.0f;
        brain_config.cmd_scale[2] = config.cmd_scale.size() > 2 ? config.cmd_scale[2] : 1.0f;
        brain_config.gait_phase_period = config.gait_phase_period;
        brain_config.rt_enabled = config.rt_enabled;
        brain_config.rt_priority = config.rt_priority;
        brain_config.rt_use_mlockall = config.rt_use_mlockall;

        engine::core::GrpcServer server;
        server.Run(config.host, config.port, config.worker_threads,
                   1000, brain_config, ros_node.get());

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
    }

    spdlog::info("Shutting down ROS2...");
    rclcpp::shutdown();
    if (ros_spin_thread.joinable()) {
        ros_spin_thread.join();
    }

    spdlog::info("RobotBrain stopped.");
    return 0;
}
