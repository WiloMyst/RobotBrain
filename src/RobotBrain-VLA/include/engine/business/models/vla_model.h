#pragma once
#include "onnx_model_base.h"
#include <vector>

namespace engine {
namespace business {
namespace models {

class VLAModel : public OnnxModelBase {
public:
    explicit VLAModel(const std::string& model_path, int intra_threads, bool use_gpu);

    // 核心推理接口：传入已经通过 OpenCV 归一化好的 NCHW 裸指针，返回两个 float [linear_x, angular_z]
    std::vector<float> Forward(float* nchw_tensor_data, size_t tensor_element_count);
};

}
}
}