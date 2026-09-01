"""
键盘交互式云端控制客户端

W/S: 前进/减速  A/D: 左转/右转  Space: 急停  Q: 退出
实时显示 G1 状态 + 当前下发的速度指令
"""

import grpc
import time
import threading
import sys
import os
import termios
import tty
import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'protos'))

try:
    import robot_brain_pb2 as rb_pb
    import robot_brain_pb2_grpc as rb_grpc
except ImportError:
    print("Error: proto files not generated. Run: cd tools && bash generate_proto.sh")
    sys.exit(1)

# 共享状态
current_vx = 0.0
current_wz = 0.0
latest_status = None
status_lock = threading.Lock()
running = True


def stream_status(stub):
    global latest_status
    try:
        stream = stub.StreamStatus(rb_pb.StatusRequest(interval_ms=200))
        for status in stream:
            with status_lock:
                latest_status = status
    except grpc.RpcError:
        pass


def send_task(stub, vx, wz, task_type=rb_pb.NAVIGATE):
    vc = rb_pb.VelocityCommand(linear_x=vx, linear_y=0.0, angular_z=wz)
    req = rb_pb.TaskRequest(
        task_id=f"key-{int(time.time()*1000)}",
        task_type=task_type,
        command=f"vx={vx:.1f} wz={wz:.1f}",
        velocity_cmd=vc)
    try:
        resp = stub.AssignTask(req)
        return resp.accepted
    except grpc.RpcError:
        return False


def draw_screen():
    global current_vx, current_wz
    state_names = {0: "IDLE", 1: "EXECUTING", 2: "ERROR", 3: "STANDING"}

    with status_lock:
        s = latest_status

    if s:
        vc = s.velocity_cmd
        joints = s.joint_targets[:4]
        joints_str = ",".join(f"{j:+.2f}" for j in joints)
        line1 = (f"\r[{state_names.get(s.state, '?'):9s}] "
                 f"gen={s.task_generation:3d} "
                 f"frames={s.total_frames:5d} "
                 f"drop={s.dropped_frames:2d} "
                 f"stale={s.stale_frames:2d} "
                 f"inf={s.avg_inference_ms:4.1f}ms "
                 f"jit={s.jitter_p50_ms:.1f}/{s.jitter_p99_ms:.1f}ms")
    else:
        line1 = "\r[connecting...]"

    bar_vx = int(current_vx * 20) * '#'
    bar_wz_l = int(max(0, -current_wz) * 20) * '<'
    bar_wz_r = int(max(0, current_wz) * 20) * '>'
    line2 = (f"  vx=[{bar_vx:20s}] {current_vx:+.2f}  "
             f"wz=[{bar_wz_l:>10s}{bar_wz_r:10s}] {current_wz:+.2f}")

    line3 = "  W/S:前/退  A/D:左/右  Space:急停  Q:退出"

    sys.stdout.write(f"\r\033[K{line1}\n\033[K{line2}\n\033[K{line3}\033[1A\033[1A")
    sys.stdout.flush()


def main():
    global current_vx, current_wz, running

    server_addr = "localhost:50051"
    print(f"Connecting to RobotBrain at {server_addr}...")

    channel = grpc.insecure_channel(server_addr)
    stub = rb_grpc.BrainServiceStub(channel)

    # 等待连接
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except grpc.FutureTimeoutError:
        print("Error: cannot connect to RobotBrain")
        return

    # 状态接收线程
    threading.Thread(target=stream_status, args=(stub,), daemon=True).start()
    time.sleep(0.5)

    # 初始下发 STAND(vx=0)
    send_task(stub, 0.0, 0.0)

    # 键盘 raw 模式
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    print("\n=== G1 Keyboard Control ===")
    print("W: 前进  S: 减速  A: 左转  D: 右转  Space: 急停  Q: 退出\n")

    try:
        while running:
            # 非阻塞读键盘
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                ch_lower = ch.lower()

                if ch_lower == 'q':
                    running = False
                    break
                elif ch_lower == 'w':
                    current_vx = min(1.0, current_vx + 0.1)
                    send_task(stub, current_vx, current_wz)
                elif ch_lower == 's':
                    current_vx = max(0.0, current_vx - 0.1)
                    send_task(stub, current_vx, current_wz)
                elif ch_lower == 'a':
                    current_wz = min(0.5, current_wz + 0.1)
                    send_task(stub, current_vx, current_wz)
                elif ch_lower == 'd':
                    current_wz = max(-0.5, current_wz - 0.1)
                    send_task(stub, current_vx, current_wz)
                elif ch == ' ':
                    current_vx = 0.0
                    current_wz = 0.0
                    send_task(stub, 0.0, 0.0, task_type=rb_pb.EMERGENCY_STOP)

            draw_screen()

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # 急停
        try:
            send_task(stub, 0.0, 0.0, task_type=rb_pb.EMERGENCY_STOP)
        except:
            pass
        channel.close()
        print("\n\nDisconnected.")


if __name__ == '__main__':
    main()
