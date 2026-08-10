"""
云端客户端测试脚本

测试两个 RPC:
1. AssignTask: 向边缘节点下发语义指令 + 速度指令
2. StreamStatus: 接收实时运行状态流 (含关节目标反馈)

运行方式:
  python cloud_client.py
"""

import grpc
import time
import threading
import sys
import os

# 将 proto 生成目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'protos'))

try:
    import robot_brain_pb2 as rb_pb
    import robot_brain_pb2_grpc as rb_grpc
except ImportError:
    print("Error: proto files not generated. Run: cd tools && bash generate_proto.sh")
    sys.exit(1)


def stream_status(stub):
    try:
        request = rb_pb.StatusRequest(interval_ms=500)
        stream = stub.StreamStatus(request)

        state_names = {0: "IDLE", 1: "EXECUTING", 2: "ERROR", 3: "STANDING"}
        task_type_names = {0: "UNKNOWN", 1: "NAVIGATE", 2: "FOLLOW", 3: "PICK", 4: "ESTOP"}

        for status in stream:
            vc = status.velocity_cmd
            joints = status.joint_targets
            joints_str = ",".join(f"{j:+.2f}" for j in joints[:4])
            print(f"\r[{state_names.get(status.state, '?'):9s}] "
                  f"type={task_type_names.get(status.task_type, '?'):8s} "
                  f"gen={status.task_generation:3d} "
                  f"frames={status.total_frames:5d} "
                  f"dropped={status.dropped_frames:3d} "
                  f"stale={status.stale_frames:3d} "
                  f"inf={status.avg_inference_ms:6.1f}ms "
                  f"jit(P50/P99)={status.jitter_p50_ms:.1f}/{status.jitter_p99_ms:.1f}ms "
                  f"vel=({vc.linear_x:+.2f},{vc.linear_y:+.2f},{vc.angular_z:+.2f}) "
                  f"j=[{joints_str}...]",
                  end='', flush=True)

    except grpc.RpcError as e:
        print(f"\nStatus stream closed: {e.code()}")


def main():
    server_addr = "localhost:50051"
    print(f"Connecting to RobotBrain at {server_addr}...")

    channel = grpc.insecure_channel(server_addr)
    stub = rb_grpc.BrainServiceStub(channel)

    status_thread = threading.Thread(target=stream_status, args=(stub,), daemon=True)
    status_thread.start()

    time.sleep(2)

    # (task_id, task_type, command, velocity_cmd=(vx, vy, wz))
    tasks = [
        ("task-001", rb_pb.NAVIGATE, "navigate to kitchen", (0.5, 0.0, 0.0)),
        ("task-002", rb_pb.PICK, "pick up the red cup", (0.0, 0.0, 0.0)),
        ("task-003", rb_pb.FOLLOW, "follow the person in blue", (0.3, 0.0, 0.1)),
        ("task-004", rb_pb.EMERGENCY_STOP, "emergency stop", (0.0, 0.0, 0.0)),
    ]

    for task_id, task_type, command, vel in tasks:
        print(f"\n>> Sending task: [{task_id}] type={task_type} {command} vel={vel}")
        vc = rb_pb.VelocityCommand(linear_x=vel[0], linear_y=vel[1], angular_z=vel[2])
        request = rb_pb.TaskRequest(task_id=task_id, task_type=task_type,
                                    command=command, velocity_cmd=vc)

        try:
            response = stub.AssignTask(request)
            print(f"<< Response: accepted={response.accepted}, msg='{response.status_msg}'")
        except grpc.RpcError as e:
            print(f"<< Error: {e.code()} - {e.details()}")

        time.sleep(5)

    print("\n>> Stopping task submission, waiting for watchdog timeout...")
    time.sleep(3)

    print("\nDone.")
    channel.close()


if __name__ == '__main__':
    main()
