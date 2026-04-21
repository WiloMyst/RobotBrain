import grpc
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../protos'))
import robot_brain_pb2, robot_brain_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = robot_brain_pb2_grpc.BrainServiceStub(channel)

    print("====== 云端车队/机器人管理系统 (FMS) ======")
    while True:
        cmd = input("请输入高级语义指令 (输入 'q' 退出) ❯ ")
        if cmd == 'q': break

        req = robot_brain_pb2.TaskRequest(
            task_id="TASK_" + os.urandom(4).hex(),
            command=cmd,
            priority=1
        )
        try:
            response = stub.AssignTask(req)
            print(f"[+] 边缘节点回执: 接受={response.accepted}, 状态={response.status_msg}")
        except grpc.RpcError as e:
            print(f"[!] 边缘节点离线: {e.code()}")

if __name__ == '__main__':
    run()