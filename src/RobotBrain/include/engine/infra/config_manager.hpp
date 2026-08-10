#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

namespace engine {
namespace infra {

struct AppConfig {
    // Server
    std::string host = "0.0.0.0";
    int port = 50051;
    int worker_threads = 4;

    // AI Brain
    std::string policy_model_path;
    int intra_op_threads = 2;
    bool use_gpu = false;
    int num_joints = 12;
    int obs_dim = 47;
    float action_scale = 0.25f;
    std::vector<float> default_joint_positions;
    float ang_vel_scale = 1.0f;
    float dof_pos_scale = 1.0f;
    float dof_vel_scale = 1.0f;
    std::vector<float> cmd_scale = {1.0f, 1.0f, 1.0f};
    float gait_phase_period = 0.8f;

    // Robot
    std::string joint_state_topic = "/joint_states";
    std::string imu_topic = "/imu";
    std::string joint_command_topic = "/joint_commands";
    int control_rate_ms = 20;
    int64_t watchdog_timeout_ms = 500;

    // Safety
    float max_joint_velocity = 1.0f;
    float max_joint_accel = 5.0f;
    float joint_position_limit = 0.5f;
    float max_vel_ramp = 2.0f;
    float fall_gravity_threshold = -0.5f;
    int fall_debounce_frames = 25;

    // Realtime
    bool rt_enabled = true;
    int rt_priority = 80;
    bool rt_use_mlockall = true;
};

inline AppConfig LoadConfig(const std::string& filepath) {
    AppConfig config;
    try {
        YAML::Node node = YAML::LoadFile(filepath);

        if (node["server"]) {
            config.host = node["server"]["host"].as<std::string>(config.host);
            config.port = node["server"]["port"].as<int>(config.port);
            config.worker_threads = node["server"]["worker_threads"].as<int>(config.worker_threads);
        }

        if (node["ai_brain"]) {
            const auto& ab = node["ai_brain"];
            config.policy_model_path = ab["policy_model_path"].as<std::string>(config.policy_model_path);
            config.intra_op_threads = ab["intra_op_threads"].as<int>(config.intra_op_threads);
            config.use_gpu = ab["use_gpu"].as<bool>(config.use_gpu);
            config.num_joints = ab["num_joints"].as<int>(config.num_joints);
            config.obs_dim = ab["obs_dim"].as<int>(config.obs_dim);
            config.action_scale = ab["action_scale"].as<float>(config.action_scale);
            if (ab["default_joint_positions"]) {
                config.default_joint_positions =
                    ab["default_joint_positions"].as<std::vector<float>>();
            }
            config.ang_vel_scale = ab["ang_vel_scale"].as<float>(config.ang_vel_scale);
            config.dof_pos_scale = ab["dof_pos_scale"].as<float>(config.dof_pos_scale);
            config.dof_vel_scale = ab["dof_vel_scale"].as<float>(config.dof_vel_scale);
            if (ab["cmd_scale"]) {
                config.cmd_scale = ab["cmd_scale"].as<std::vector<float>>();
            }
            config.gait_phase_period = ab["gait_phase_period"].as<float>(config.gait_phase_period);
        }

        if (node["robot"]) {
            const auto& r = node["robot"];
            config.joint_state_topic = r["joint_state_topic"].as<std::string>(config.joint_state_topic);
            config.imu_topic = r["imu_topic"].as<std::string>(config.imu_topic);
            config.joint_command_topic = r["joint_command_topic"].as<std::string>(config.joint_command_topic);
            config.control_rate_ms = r["control_rate_ms"].as<int>(config.control_rate_ms);
            config.watchdog_timeout_ms = r["watchdog_timeout_ms"].as<int64_t>(config.watchdog_timeout_ms);
        }

        if (node["safety"]) {
            const auto& s = node["safety"];
            config.max_joint_velocity = s["max_joint_velocity"].as<float>(config.max_joint_velocity);
            config.max_joint_accel = s["max_joint_accel"].as<float>(config.max_joint_accel);
            config.joint_position_limit = s["joint_position_limit"].as<float>(config.joint_position_limit);
            config.max_vel_ramp = s["max_vel_ramp"].as<float>(config.max_vel_ramp);
            config.fall_gravity_threshold = s["fall_gravity_threshold"].as<float>(config.fall_gravity_threshold);
            config.fall_debounce_frames = s["fall_debounce_frames"].as<int>(config.fall_debounce_frames);
        }

        if (node["realtime"]) {
            const auto& rt = node["realtime"];
            config.rt_enabled = rt["enabled"].as<bool>(config.rt_enabled);
            config.rt_priority = rt["priority"].as<int>(config.rt_priority);
            config.rt_use_mlockall = rt["use_mlockall"].as<bool>(config.rt_use_mlockall);
        }

        if (config.default_joint_positions.size() != static_cast<size_t>(config.num_joints)) {
            spdlog::warn("default_joint_positions size ({}) != num_joints ({}), using zeros",
                         config.default_joint_positions.size(), config.num_joints);
            config.default_joint_positions.assign(config.num_joints, 0.0f);
        }

        return config;
    } catch (const YAML::Exception& e) {
        spdlog::critical("Config load failed: {}", e.what());
        throw std::runtime_error("Config load failed");
    }
}

} // namespace infra
} // namespace engine
