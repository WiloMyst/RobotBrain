#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace engine {
namespace infra {

/// 日志初始化:控制台 + 轮转文件双输出
inline void InitLogger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "logs/robot_brain.log", 1024 * 1024 * 10, 3);
    file_sink->set_level(spdlog::level::trace);

    spdlog::logger logger("RobotBrain", {console_sink, file_sink});
    logger.set_level(spdlog::level::debug);
    logger.set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

    spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
}

} // namespace infra
} // namespace engine
