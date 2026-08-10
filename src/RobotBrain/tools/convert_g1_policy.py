#!/usr/bin/env python3
"""把 unitree_rl_gym 的 G1 motion.pt (JIT) 转成 ONNX。

用法:
  python3 tools/convert_g1_policy.py [path/to/motion.pt]

默认从 models/pretrained/motion_g1.pt 读取(已备份到本地)。

输出:
  models/policy_g1.onnx  (47 -> 12)
"""
import sys
import os
import torch


def main():
    if len(sys.argv) < 2:
        pt_path = os.path.join(os.path.dirname(__file__), "..", "models", "pretrained", "motion_g1.pt")
    else:
        pt_path = sys.argv[1]

    print(f"Loading: {pt_path}")
    model = torch.jit.load(pt_path)
    model.eval()

    dummy = torch.zeros(1, 47)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Sample: {out.squeeze().tolist()[:6]}...")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "policy_g1.onnx")

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["obs"], output_names=["actions"],
        opset_version=14,
        dynamo=False,
    )
    print(f"\nExported: {onnx_path}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        result = sess.run(None, {"obs": dummy.numpy()})[0]
        match = abs(result - out.numpy()).max() < 1e-5
        print(f"ONNX verify: {result.shape}, match: {match}")
    except ImportError:
        print("onnxruntime not installed, skip verify")


if __name__ == "__main__":
    main()
