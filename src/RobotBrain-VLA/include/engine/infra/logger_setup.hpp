#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace engine {
namespace infra {

inline void InitLogger() {
    spdlog::init_thread_pool(8192, 1);
    auto async_logger = spdlog::stdout_color_mt<spdlog::async_factory>("EngineLogger");
    async_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] %v");
    spdlog::set_default_logger(async_logger);
    spdlog::set_level(spdlog::level::info);
}

} // namespace infra
} // namespace engine