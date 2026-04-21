import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time

class MockCamera(Node):
    def __init__(self):
        super().__init__('mock_camera_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.033, self.timer_callback) # 30 FPS
        print("摄像头节点已启动，正在以 30FPS 狂暴发送空图像...")

    def timer_callback(self):
        msg = Image()
        msg.height = 224
        msg.width = 224
        msg.encoding = 'rgb8'
        # 伪造一帧 224x224x3 的空白图像字节流
        msg.data = b'\x00' * (224 * 224 * 3) 
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(MockCamera())
    rclpy.shutdown()

if __name__ == '__main__':
    main()