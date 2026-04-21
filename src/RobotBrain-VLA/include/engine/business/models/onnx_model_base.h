#pragma once
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include <filesystem> // 用于创建缓存目录

namespace engine {
namespace business {
namespace models {

class OnnxModelBase {
public:
    explicit OnnxModelBase(const std::string& model_path, int intra_op_threads = 1, bool use_gpu = false) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DualModeEngine");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(intra_op_threads);

        // ====================================================================
        // 异构调度：只有明确要求使用 GPU 的模型，才挂载 CUDA！
        // ====================================================================
        if (use_gpu) {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 1; // 克制显存分配，不与 UE5 抢
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            spdlog::info("模型挂载至 [GPU CUDA] 节点: {}", model_path);
        } else {
            spdlog::info("模型挂载至 [CPU] 节点: {}", model_path);
        }
        // ====================================================================

        //std::filesystem::create_directories("./trt_cache");
        // 异构调度：TensorRT
        // if (use_tensorrt) {
        //     // 尝试挂载 TensorRT
        //     OrtTensorRTProviderOptions trt_options{};
        //     trt_options.device_id = 0;
        //     trt_options.trt_fp16_enable = 1;
        //     trt_options.trt_engine_cache_enable = 1;
        //     trt_options.trt_engine_cache_path = "./trt_cache";
        //     session_options.AppendExecutionProvider_TensorRT(trt_options);
        //     spdlog::info("模型启用 [TensorRT + CUDA] 异构加速: {}", model_path);
        // } else {
        //     spdlog::info("模型启用 [纯 CUDA] 稳定加速: {}", model_path);
        // }
        // // 无论是 TRT 还是纯 CUDA，都必须挂载 CUDA EP 作为兜底/主力
        // OrtCUDAProviderOptions cuda_options{};
        // cuda_options.device_id = 0;
        // // 禁止 ORT 贪婪预分配显存
        // // 0 = kNextPowerOfTwo（默认，极度浪费显存）
        // // 1 = kSameAsRequested（极其克制，按需申请，不与 UE5 抢地盘）
        // cuda_options.arena_extend_strategy = 1;
        // session_options.AppendExecutionProvider_CUDA(cuda_options);

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
    } catch (const Ort::Exception& e) {
        spdlog::critical("模型加载失败: {}", model_path);
        spdlog::critical("底层报错: {}", e.what());
        throw; 
    }
}
    
    virtual ~OnnxModelBase() = default;

protected:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
};

} // namespace models
} // namespace business
} // namespace engine