#pragma once
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

namespace engine {
namespace infra {

struct AppConfig {
    std::string host;
    int port;
    int worker_threads;
    std::string vla_model_path;
    std::string camera_topic;
    std::string cmd_topic;
};

inline AppConfig LoadConfig(const std::string& filepath) {
    AppConfig config;
    try {
        YAML::Node node = YAML::LoadFile(filepath);
        config.host = node["server"]["host"].as<std::string>();
        config.port = node["server"]["port"].as<int>();
        config.worker_threads = node["server"]["worker_threads"].as<int>();
        
        config.vla_model_path = node["ai_brain"]["vla_model_path"].as<std::string>();
        config.camera_topic = node["robot"]["camera_topic"].as<std::string>();
        config.cmd_topic = node["robot"]["cmd_topic"].as<std::string>();
        return config;
    } catch (const YAML::Exception& e) {
        spdlog::critical("配置文件读取失败: {}", e.what());
        throw std::runtime_error("Config load failed");
    }
}

} // namespace infra
} // namespace engine