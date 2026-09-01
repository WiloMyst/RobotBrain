#!/usr/bin/env python3
"""诊断 v4:测试 ONNX 在 cmd=0.5 下能否站住(对比 JIT)"""
import os, math, numpy as np, mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(SCRIPT_DIR, '..', 'assets', 'mujoco', 'unitree_g1', 'scene.xml')
PT = os.path.join(SCRIPT_DIR, '..', 'models', 'pretrained', 'motion_g1.pt')
ONNX = os.path.join(SCRIPT_DIR, '..', 'models', 'policy_g1.onnx')

DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0])
KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
ANG_VEL_SCALE = 0.25; DOF_POS_SCALE = 1.0; DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32); ACTION_SCALE = 0.25
STAND_H = 0.793; DT = 0.002; CTRL_DEC = 10; PERIOD = 0.8
NUM_ACTIONS = 12; NUM_OBS = 47

def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2*(-qz*qx+qw*qy), -2*(qz*qy+qw*qx), 1-2*(qw*qw+qz*qz)])

m = mujoco.MjModel.from_xml_path(XML); m.opt.timestep = DT
d = mujoco.MjData(m)

import torch
jit = torch.jit.load(PT); jit.eval()
import onnxruntime as ort
onnx_sess = ort.InferenceSession(ONNX)

renderer = mujoco.Renderer(m, 480, 480)

def run_policy(name, infer_fn, cmd_vel, steps=5000):
    mujoco.mj_resetData(m, d); d.qpos[2] = STAND_H
    d.qpos[7:7+NUM_ACTIONS] = DEFAULT_ANGLES
    mujoco.mj_forward(m, d)
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    cmd = np.array(cmd_vel, dtype=np.float32)
    counter = 0; falls = 0
    state = None  # LSTM 状态(ONNX 用)

    for step in range(steps):
        tau = (target - d.qpos[7:7+NUM_ACTIONS]) * KP + (np.zeros_like(KD) - d.qvel[6:6+NUM_ACTIONS]) * KD
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if counter % CTRL_DEC == 0:
            qj = d.qpos[7:7+NUM_ACTIONS]; dqj = d.qvel[6:6+NUM_ACTIONS]
            obs[:3] = d.qvel[3:6] * ANG_VEL_SCALE
            obs[3:6] = get_gravity_orientation(d.qpos[3:7])
            obs[6:9] = cmd * CMD_SCALE
            obs[9:21] = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
            obs[21:33] = dqj * DOF_VEL_SCALE
            obs[33:45] = action
            phase = (counter * DT) % PERIOD / PERIOD
            obs[45:47] = [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)]
            action, state = infer_fn(obs, state)
            target = action * ACTION_SCALE + DEFAULT_ANGLES

        h = d.qpos[2]; qw,qx,qy,qz = d.qpos[3:7]
        pitch = math.asin(max(-1,min(1,2*(qw*qy-qz*qx))))
        if h < 0.4 or abs(pitch) > 0.785:
            falls += 1
            print(f"  [{name}] Step {step}: FALL h={h:.3f} pitch={pitch:.2f}")
            mujoco.mj_resetData(m, d); d.qpos[2] = STAND_H
            d.qpos[7:7+NUM_ACTIONS] = DEFAULT_ANGLES
            mujoco.mj_forward(m, d)
            action = np.zeros(NUM_ACTIONS, dtype=np.float32); target = DEFAULT_ANGLES.copy()
            state = None  # 重置 LSTM 状态
            if falls >= 2: break

        ctrl_step = counter // CTRL_DEC
        if ctrl_step % 100 == 0 and counter % CTRL_DEC == 0:
            print(f"  [{name}] Ctrl {ctrl_step:3d}: h={h:.3f} pitch={pitch:.3f} act[:3]={action[:3]}")

    renderer.update_scene(d, camera=-1)
    from PIL import Image; Image.fromarray(renderer.render()).save(
        os.path.join(SCRIPT_DIR, '..', f'diagnose_{name}.png'))
    return falls

def jit_infer(obs, state):
    with torch.no_grad():
        a = jit(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
    return a, None

def onnx_infer(obs, state):
    if state is None:
        state = (np.zeros((1,1,64), dtype=np.float32), np.zeros((1,1,64), dtype=np.float32))
    hidden, cell = state
    out = onnx_sess.run(None, {'obs': obs.reshape(1,-1), 'hidden': hidden, 'cell': cell})
    action = out[0][0]
    new_state = (out[1], out[2])
    return action, new_state

print("=== JIT + cmd=0.5 ===")
f1 = run_policy("jit_05", jit_infer, [0.5, 0, 0])
print(f"Falls: {f1}\n")

print("=== ONNX + cmd=0.5 ===")
f2 = run_policy("onnx_05", onnx_infer, [0.5, 0, 0])
print(f"Falls: {f2}\n")

print("=== ONNX + cmd=0.0 ===")
f3 = run_policy("onnx_00", onnx_infer, [0.0, 0, 0])
print(f"Falls: {f3}\n")

print(f"\nSummary: JIT/0.5={f1}, ONNX/0.5={f2}, ONNX/0.0={f3}")
