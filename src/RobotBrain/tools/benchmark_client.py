import grpc, time, sys, os, statistics, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'protos'))
import robot_brain_pb2 as pb
import robot_brain_pb2_grpc as pb_grpc

DURATION = 30
channel = grpc.insecure_channel('localhost:50051')
stub = pb_grpc.BrainServiceStub(channel)

samples = []
final = [None]
def read_stream():
    try:
        for s in stub.StreamStatus(pb.StatusRequest()):
            samples.append(s)
            final[0] = s
    except Exception as e:
        print('stream err:', e)

t = threading.Thread(target=read_stream, daemon=True)
t.start()
time.sleep(1.0)

vc = pb.VelocityCommand(linear_x=0.5, linear_y=0.0, angular_z=0.0)
stub.AssignTask(pb.TaskRequest(task_id='bench-nav', task_type=pb.NAVIGATE,
                               command='benchmark', velocity_cmd=vc))
print('NAVIGATE sent (vel=0.5,0,0), running %ds ...' % DURATION)
time.sleep(DURATION)
stub.AssignTask(pb.TaskRequest(task_id='bench-stop', task_type=pb.EMERGENCY_STOP,
                               command='stop'))
time.sleep(1.5)

if not samples:
    print('no samples'); sys.exit(1)

latencies = [s.last_inference_ms for s in samples if s.last_inference_ms > 0]
f = final[0]
total = f.total_frames
dropped = f.dropped_frames
stale = f.stale_frames
attempts = total + dropped + stale
print('=== benchmark (%ds, G1 RL policy, 47-dim obs -> 12 joint targets) ===' % DURATION)
print('frames: total=%d dropped=%d stale=%d' % (total, dropped, stale))
if attempts > 0:
    print('rate: drop=%.2f%% stale=%.2f%%' % (dropped/attempts*100, stale/attempts*100))
if latencies:
    ls = sorted(latencies); n = len(ls)
    print('inference_ms: avg=%.2f P50=%.2f P95=%.2f P99=%.2f max=%.2f (n=%d)' % (
        statistics.mean(ls), ls[n//2], ls[int(n*0.95)], ls[int(n*0.99)], ls[-1], n))
print('node_avg_inference_ms=%.2f' % f.avg_inference_ms)
