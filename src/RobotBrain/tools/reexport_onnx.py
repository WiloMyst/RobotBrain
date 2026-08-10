#!/usr/bin/env python3
"""
重新导出 ONNX:提取 JIT 模型权重,重建无状态 LSTM policy。
motion.pt 结构:
  memory: LSTM(47, 64)
  actor: Sequential(Linear(64,32), ELU, Linear(32,12))
  hidden_state/cell_state: [1, 1, 64] (内部状态)
"""
import os, torch, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PT = os.path.join(SCRIPT_DIR, '..', 'models', 'pretrained', 'motion_g1.pt')
ONNX_OUT = os.path.join(SCRIPT_DIR, '..', 'models', 'policy_g1.onnx')

policy = torch.jit.load(PT)
policy.eval()

HIDDEN_DIM = 64
OBS_DIM = 47
ACTION_DIM = 12

# 重建无状态模型
class StatelessLSTMPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = torch.nn.LSTM(OBS_DIM, HIDDEN_DIM, batch_first=False)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_DIM, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, ACTION_DIM),
        )

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # obs: [1, 47], hidden/cell: [1, 1, 64]
        out, (h, c) = self.memory(obs.unsqueeze(0), (hidden, cell))
        action = self.actor(out.squeeze(0))
        return action, h, c

new_model = StatelessLSTMPolicy()
new_model.eval()

# 提取 JIT 模型的权重
print("=== Extracting weights from JIT model ===")
# LSTM 权重
lstm_w_ih = policy.memory.weight_ih_l0
lstm_w_hh = policy.memory.weight_hh_l0
lstm_b_ih = policy.memory.bias_ih_l0
lstm_b_hh = policy.memory.bias_hh_l0
print(f"LSTM weight_ih: {lstm_w_ih.shape}, weight_hh: {lstm_w_hh.shape}")

# Actor 权重 (JIT Sequential 不支持索引,用 children)
actor_children = list(policy.actor.children())
actor_w0 = actor_children[0].weight
actor_b0 = actor_children[0].bias
actor_w2 = actor_children[2].weight
actor_b2 = actor_children[2].bias
print(f"Actor[0]: {actor_w0.shape}, Actor[2]: {actor_w2.shape}")

# 复制到新模型
with torch.no_grad():
    new_model.memory.weight_ih_l0.copy_(lstm_w_ih)
    new_model.memory.weight_hh_l0.copy_(lstm_w_hh)
    new_model.memory.bias_ih_l0.copy_(lstm_b_ih)
    new_model.memory.bias_hh_l0.copy_(lstm_b_hh)
    new_model.actor[0].weight.copy_(actor_w0)
    new_model.actor[0].bias.copy_(actor_b0)
    new_model.actor[2].weight.copy_(actor_w2)
    new_model.actor[2].bias.copy_(actor_b2)

# 验证:对比新模型和 JIT 模型的输出
print("\n=== Verify: new model vs JIT ===")
obs = torch.zeros(1, 47); obs[0, 5] = -1.0; obs[0, 46] = 1.0
h0 = torch.zeros(1, 1, HIDDEN_DIM)
c0 = torch.zeros(1, 1, HIDDEN_DIM)

# 重置 JIT 模型状态
with torch.no_grad():
    policy.hidden_state[:] = 0
    policy.cell_state[:] = 0
    jit_out = policy(obs)
    new_out, new_h, new_c = new_model(obs, h0, c0)
    print(f"JIT output:    {jit_out.numpy()[0][:3]}")
    print(f"New model:     {new_out.numpy()[0][:3]}")
    diff = (jit_out - new_out).abs().max().item()
    print(f"Max diff: {diff:.8f}")
    assert diff < 1e-5, f"Mismatch! diff={diff}"

# 连续 5 步验证
print("\n=== Verify: 5 steps continuous ===")
with torch.no_grad():
    h, c = h0, c0
    policy.hidden_state[:] = 0
    policy.cell_state[:] = 0
    for i in range(5):
        obs_i = torch.randn(1, 47) * 0.1
        jit_a = policy(obs_i)
        new_a, h, c = new_model(obs_i, h, c)
        d = (jit_a - new_a).abs().max().item()
        print(f"  Step {i}: diff={d:.8f} jit={jit_a.numpy()[0][:2]} new={new_a.numpy()[0][:2]}")
        assert d < 1e-4, f"Step {i} mismatch!"

# 导出 ONNX (dynamo=False: 旧导出器,权重内嵌,IR 版本低,兼容性好)
print(f"\n=== Exporting ONNX: {ONNX_OUT} ===")
torch.onnx.export(
    new_model,
    (torch.zeros(1, 47), torch.zeros(1, 1, HIDDEN_DIM), torch.zeros(1, 1, HIDDEN_DIM)),
    ONNX_OUT,
    input_names=['obs', 'hidden', 'cell'],
    output_names=['action', 'new_hidden', 'new_cell'],
    opset_version=17,
    dynamo=False,
)

# 验证 ONNX
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_OUT)
print(f"Inputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

# 5 步 ONNX 验证
print("\n=== Verify: ONNX 5 steps ===")
h_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
c_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
for i in range(5):
    obs_np = np.random.randn(1, 47).astype(np.float32) * 0.1
    outputs = sess.run(None, {'obs': obs_np, 'hidden': h_np, 'cell': c_np})
    action_np, h_np, c_np = outputs
    print(f"  Step {i}: action={action_np[0][:2]}")

print("\nSUCCESS: ONNX exported and verified.")
print(f"Model: {ONNX_OUT}")
print(f"Inputs: obs[1,47] + hidden[1,1,64] + cell[1,1,64]")
print(f"Outputs: action[1,12] + new_hidden[1,1,64] + new_cell[1,1,64]")
