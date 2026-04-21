#include "engine/business/models/vla_model.h"

namespace engine {
namespace business {
namespace models {

VLAModel::VLAModel(const std::string& model_path, int intra_threads, bool use_gpu)
    : OnnxModelBase(model_path, intra_threads, use_gpu) {
    spdlog::info(" [VLA Model] 视觉-动作大模型引擎已挂载.");
}

std::vector<float> VLAModel::Forward(float* nchw_tensor_data, size_t tensor_element_count) {
    // 1. 直接复用池化内存创建 Tensor (Zero-Copy)
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, 
        nchw_tensor_data, 
        tensor_element_count, 
        input_shape.data(), 
        input_shape.size()
    );

    const char* input_names[] = {"image_input"};
    const char* output_names[] = {"action_output"};

    // 2. 执行推理
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // 3. 提取结果
    float* out_arr = output_tensors.front().GetTensorMutableData<float>();
    
    // 返回 [前进速度, 转向角速度]
    return {out_arr[0], out_arr[1]};
}

}
}
}