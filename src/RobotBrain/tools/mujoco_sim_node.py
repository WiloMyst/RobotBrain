"""
Unitree G1 人形机器人 MuJoCo 仿真器节点

作为 ROS2 节点提供真实物理仿真本体感知:
  - 加载 unitree_rl_gym 的 G1 MJCF (g1_12dof.xml, 12-DoF 双腿)
  - 100Hz 步进 MuJoCo 物理引擎 (500Hz 子步),发布 /joint_states + /imu
  - 订阅 /joint_commands (12 维绝对关节目标),PD 控制器转关节力矩

闭环链路:
  MuJoCo(物理) -> /joint_states + /imu -> RobotBrain(ONNX 推理)
  RobotBrain -> /joint_commands -> MuJoCo(PD 力矩)

关节顺序 (G1 标准,与 g1_12dof.xml actuator 一致):
  left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll

PD 参数 (对齐 unitree_rl_gym deploy_mujoco/configs/g1.yaml):
  kp = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
  kd = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]

运行:
  ros2 run robot_brain mujoco_sim_node
  或
  python mujoco_sim_node.py
"""

import os
import sys
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

import mujoco

try:
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


STANDARD_JOINTS = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
    'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
    'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
]

DEFAULT_JOINT_POSITIONS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
]

DEFAULT_KP = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
DEFAULT_KD = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
DEFAULT_TORQUE_LIMITS = [60, 60, 60, 90, 40, 40, 60, 60, 60, 90, 40, 40]

STAND_BASE_HEIGHT = 0.793
FALL_HEIGHT_THRESHOLD = 0.4
FALL_PITCH_THRESHOLD = 0.785


def _find_resource(rel_path):
    """按优先级查找 G1 MJCF: 命令行参数 > 相对源码目录 > 安装 share 目录"""
    candidates = []
    if rel_path and os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(src_root, "assets", "mujoco", "unitree_g1", "scene.xml"))
    candidates.append(os.path.join(src_root, rel_path or "_"))
    if 'AMENT_PREFIX_PATH' in os.environ:
        for prefix in os.environ['AMENT_PREFIX_PATH'].split(os.pathsep):
            candidates.append(os.path.join(prefix, "share", "robot_brain", "assets",
                                           "mujoco", "unitree_g1", "scene.xml"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class MuJoCoSimNode(Node):
    """MuJoCo Unitree G1 仿真器,提供真实物理本体感知"""

    def __init__(self):
        super().__init__('mujoco_sim_node')

        self.declare_parameter('model_xml', '')
        self.declare_parameter('sim_dt', 0.002)
        self.declare_parameter('control_rate_ms', 10)
        self.declare_parameter('viewer', False)
        self.declare_parameter('render_output', '')
        self.declare_parameter('render_interval_ms', 500)
        self.declare_parameter('direct_policy', False)

        self.use_viewer = self.get_parameter('viewer').value and HAS_VIEWER
        self.render_output = self.get_parameter('render_output').value
        self.render_interval_ms = int(self.get_parameter('render_interval_ms').value)
        self.direct_policy = self.get_parameter('direct_policy').value

        # direct_policy 模式:在 MuJoCo 进程内直接用 ONNX 推理,跳过 ROS2
        self.policy_sess = None
        self.lstm_hidden = None
        self.lstm_cell = None
        self.prev_actions = np.zeros(12, dtype=np.float32)
        self.policy_counter = 0
        if self.direct_policy:
            import onnxruntime as ort
            onnx_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'models', 'policy_g1.onnx')
            self.policy_sess = ort.InferenceSession(onnx_path)
            self.in_names = [i.name for i in self.policy_sess.get_inputs()]
            self.out_names = [o.name for o in self.policy_sess.get_outputs()]
            self.lstm_hidden = np.zeros((1, 1, 64), dtype=np.float32)
            self.lstm_cell = np.zeros((1, 1, 64), dtype=np.float32)
            self.get_logger().info(f"Direct policy mode: ONNX loaded from {onnx_path}")
            self.physics_enabled = True  # direct 模式直接启用 physics

        xml_param = self.get_parameter('model_xml').value
        xml_path = _find_resource(xml_param if xml_param else None)
        if not xml_path:
            self.get_logger().error(
                "G1 MJCF not found. Expected assets/mujoco/unitree_g1/scene.xml")
            raise FileNotFoundError("scene.xml not found")

        sim_dt = float(self.get_parameter('sim_dt').value)
        self.control_rate_ms = int(self.get_parameter('control_rate_ms').value)
        self.steps_per_control = max(1, int((self.control_rate_ms / 1000.0) / sim_dt))

        self.kp = list(DEFAULT_KP)
        self.kd = list(DEFAULT_KD)
        self.torque_limits = list(DEFAULT_TORQUE_LIMITS)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = []
        self.dof_adr = []
        self.ctrl_adr = []
        self._build_joint_mapping()

        self.target_positions = list(DEFAULT_JOINT_POSITIONS)
        self._reset_to_standing()

        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info("MuJoCo viewer launched (close window or Ctrl+C to exit)")

        self.renderer = None
        self.render_counter = 0
        if self.render_output and HAS_PIL:
            try:
                self.renderer = mujoco.Renderer(self.model, 480, 480)
                self.get_logger().info(f"Offscreen render -> {self.render_output} (every {self.render_interval_ms}ms)")
            except Exception as e:
                self.get_logger().warn(f"Renderer init failed: {e}")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.joint_pub = self.create_publisher(JointState, '/joint_states', sensor_qos)
        self.imu_pub = self.create_publisher(Imu, '/imu', sensor_qos)

        self.cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_commands',
            self._cmd_callback, 10)

        self.physics_enabled = False
        self._diag_counter = 0

        self.timer = self.create_timer(self.control_rate_ms / 1000.0, self._step)

        self.get_logger().info(
            f"MuJoCoSimNode (G1) ready [xml={os.path.basename(xml_path)}] "
            f"dt={sim_dt}s steps/ctrl={self.steps_per_control} "
            f"pub={self.control_rate_ms}ms"
        )

    def _build_joint_mapping(self):
        """建立 G1 标准关节顺序 -> MuJoCo qpos/qvel/ctrl 地址映射"""
        name_to_jnt = {}
        for i in range(self.model.njnt):
            raw = self.model.joint(i).name
            name_to_jnt[raw] = i

        name_to_act = {}
        for i in range(self.model.nu):
            raw = self.model.actuator(i).name
            name_to_act[raw] = i

        for std in STANDARD_JOINTS:
            key = std + '_joint'
            if key not in name_to_jnt:
                self.get_logger().error(f"Joint '{key}' not found in MuJoCo model")
                raise RuntimeError(f"missing joint {key}")
            jnt_idx = name_to_jnt[key]
            self.qpos_adr.append(int(self.model.jnt_qposadr[jnt_idx]))
            self.dof_adr.append(int(self.model.jnt_dofadr[jnt_idx]))

            if key not in name_to_act:
                self.get_logger().error(f"Actuator for '{key}' not found in MuJoCo model")
                raise RuntimeError(f"missing actuator {key}")
            self.ctrl_adr.append(name_to_act[key])

        self.get_logger().info(
            f"Joint mapping built ({len(STANDARD_JOINTS)} joints): "
            + ", ".join(STANDARD_JOINTS))

    def _reset_to_standing(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = STAND_BASE_HEIGHT
        for i, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = DEFAULT_JOINT_POSITIONS[i]
        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(f"Reset to standing pose (base_h={STAND_BASE_HEIGHT}m)")

    def _run_direct_policy(self):
        """direct_policy 模式:在 MuJoCo 进程内直接用 ONNX 推理,跳过 ROS2"""
        # obs 组装 (47 维,和 diagnose_v4.py 完全一致)
        obs = np.zeros(47, dtype=np.float32)
        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

        # omega (world frame,不转 body frame)
        obs[0:3] = np.array(self.data.qvel[3:6]) * 0.25
        # projected_gravity (官方公式)
        obs[3] = 2 * (-qz * qx + qw * qy)
        obs[4] = -2 * (qz * qy + qw * qx)
        obs[5] = 1 - 2 * (qw * qw + qz * qz)
        # cmd (G1 policy cmd_init=[0.5, 0, 0])
        obs[6:9] = np.array([0.5, 0.0, 0.0]) * np.array([2.0, 2.0, 0.25])
        # dof_pos (相对 default)
        qj = np.array([self.data.qpos[a] for a in self.qpos_adr])
        obs[9:21] = (qj - np.array(DEFAULT_JOINT_POSITIONS)) * 1.0
        # dof_vel
        dqj = np.array([self.data.qvel[a] for a in self.dof_adr])
        obs[21:33] = dqj * 0.05
        # prev_actions
        obs[33:45] = self.prev_actions
        # phase
        sim_time = self.policy_counter * (self.control_rate_ms / 1000.0)
        phase = (sim_time % 0.8) / 0.8
        obs[45] = math.sin(2 * math.pi * phase)
        obs[46] = math.cos(2 * math.pi * phase)

        # ONNX 推理 (3 输入:obs + hidden + cell)
        outputs = self.policy_sess.run(self.out_names, {
            self.in_names[0]: obs.reshape(1, -1),
            self.in_names[1]: self.lstm_hidden,
            self.in_names[2]: self.lstm_cell,
        })
        action = outputs[0][0]
        self.lstm_hidden = outputs[1]
        self.lstm_cell = outputs[2]
        self.prev_actions = action.copy()

        # 更新 target_positions
        self.target_positions = (action * 0.25 + np.array(DEFAULT_JOINT_POSITIONS)).tolist()

    def _detect_fall(self):
        """倒地检测:base height < 0.4m 或 |pitch| > 45°"""
        base_height = float(self.data.qpos[2])
        quat = self.data.qpos[3:7]
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return base_height < FALL_HEIGHT_THRESHOLD or abs(pitch) > FALL_PITCH_THRESHOLD

    def _cmd_callback(self, msg):
        if len(msg.data) == len(STANDARD_JOINTS):
            self.target_positions = [float(v) for v in msg.data]
            if not self.physics_enabled:
                self.physics_enabled = True
                self.get_logger().info("First joint_commands received, physics enabled")

    def _step(self):
        if self.physics_enabled:
            # direct_policy 模式:每 control_rate_ms 推理一次 (50Hz)
            if self.direct_policy and self.policy_sess is not None:
                self.policy_counter += 1
                if self.policy_counter % 2 == 0:  # 每 2 个 _step = 20ms = 50Hz
                    self._run_direct_policy()

            # PD 控制:每个物理步都算 (500Hz)
            for sub in range(self.steps_per_control):
                for i in range(len(STANDARD_JOINTS)):
                    q = float(self.data.qpos[self.qpos_adr[i]])
                    dq = float(self.data.qvel[self.dof_adr[i]])
                    tau = self.kp[i] * (self.target_positions[i] - q) - self.kd[i] * dq
                    tau = max(-self.torque_limits[i], min(self.torque_limits[i], tau))
                    self.data.ctrl[self.ctrl_adr[i]] = tau
                mujoco.mj_step(self.model, self.data)

            # 倒地检测
            if self._detect_fall():
                self.get_logger().warn(
                    f"Fall detected (h={self.data.qpos[2]:.3f}m), resetting to standing")
                self._reset_to_standing()
                self.target_positions = list(DEFAULT_JOINT_POSITIONS)
                if self.direct_policy:
                    self.lstm_hidden = np.zeros((1, 1, 64), dtype=np.float32)
                    self.lstm_cell = np.zeros((1, 1, 64), dtype=np.float32)
                    self.prev_actions = np.zeros(12, dtype=np.float32)
                else:
                    self.physics_enabled = False

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        elif self.viewer is not None and not self.viewer.is_running():
            self.get_logger().info("Viewer closed by user, shutting down")
            raise SystemExit

        if self.renderer is not None:
            self.render_counter += self.control_rate_ms
            if self.render_counter >= self.render_interval_ms:
                self.render_counter = 0
                self._render_frame()

        self._publish_joint_state()
        self._publish_imu()

        # 诊断日志
        self._diag_counter += 1
        if self._diag_counter % 100 == 0:
            q = self.data.qpos[3:7]
            self.get_logger().info(
                f"[DIAG] physics={self.physics_enabled} h={self.data.qpos[2]:.3f} "
                f"quat=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}] "
                f"qj[:3]=[{self.data.qpos[self.qpos_adr[0]]:.3f},{self.data.qpos[self.qpos_adr[1]]:.3f},{self.data.qpos[self.qpos_adr[2]]:.3f}]")

    def _publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(STANDARD_JOINTS)
        msg.position = [float(self.data.qpos[a]) for a in self.qpos_adr]
        msg.velocity = [float(self.data.qvel[a]) for a in self.dof_adr]
        self.joint_pub.publish(msg)

    def _publish_imu(self):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        msg.orientation.w = float(self.data.qpos[3])
        msg.orientation.x = float(self.data.qpos[4])
        msg.orientation.y = float(self.data.qpos[5])
        msg.orientation.z = float(self.data.qpos[6])

        # unitree_rl_gym 约定:omega 直接用 world frame (d.qvel[3:6])
        msg.angular_velocity.x = float(self.data.qvel[3])
        msg.angular_velocity.y = float(self.data.qvel[4])
        msg.angular_velocity.z = float(self.data.qvel[5])

        msg.orientation_covariance = [0.0] * 9
        msg.angular_velocity_covariance = [0.0] * 9
        self.imu_pub.publish(msg)

    def _render_frame(self):
        """离线渲染当前仿真状态到 PNG 文件"""
        try:
            self.renderer.update_scene(self.data, camera=-1)
            pixels = self.renderer.render()
            img = Image.fromarray(pixels)
            img.save(self.render_output)
        except Exception as e:
            self.get_logger().warn(f"Render failed: {e}")
            self.renderer = None


def main():
    rclpy.init()
    node = MuJoCoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[mujoco_sim_node] spin interrupted: {e}", file=sys.stderr)
    finally:
        try:
            if hasattr(node, 'viewer') and node.viewer is not None:
                node.viewer.close()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
