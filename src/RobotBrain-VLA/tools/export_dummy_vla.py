import torch
import torch.nn as nn
import os
import onnx

class DummyVLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 极简的特征提取层 (模拟视觉主干网络 ResNet/MobileNet)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)) # 直接压缩成 1x1
        )
        # 决策层 (拟合出 2 个动作控制量：线速度 linear_x, 角速度 angular_z)
        self.action_head = nn.Linear(16, 2)

    def forward(self, img_tensor):
        # img_tensor shape: [batch, 3, 224, 224]
        x = self.features(img_tensor)
        x = torch.flatten(x, 1)
        # 强行截断输出范围，保证物理底盘安全 (例如: 速度在 -1.0 到 1.0 之间)
        actions = torch.tanh(self.action_head(x))
        return actions

if __name__ == '__main__':
    model = DummyVLAModel()
    model.eval()

    # 构造假输入: 1张图片, 3通道(RGB), 宽224, 高224
    dummy_input = torch.randn(1, 3, 224, 224)
    os.makedirs("../models", exist_ok=True)
    onnx_path = "../models/dummy_vla.onnx"

    print("正在导出端到端视觉控制 VLA 模型...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=['image_input'],
        output_names=['action_output'],
        dynamic_axes={'image_input': {0: 'batch'}, 'action_output': {0: 'batch'}}
    )
    print(f"成功生成: {onnx_path}")