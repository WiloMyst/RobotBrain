#!/usr/bin/env python3
"""检查 ONNX 模型的输入输出结构"""
import onnxruntime as ort
s = ort.InferenceSession('/home/wzh/robotbrain_ws/src/RobotBrain/models/policy_g1.onnx')
print("Inputs:")
for i in s.get_inputs():
    print(f"  {i.name}: {i.shape}")
print("Outputs:")
for o in s.get_outputs():
    print(f"  {o.name}: {o.shape}")
