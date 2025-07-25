import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_parameter('video_device', '/dev/video0')  # Default camera device
        self.declare_parameter('image_width', 640)  # Default width
        self.declare_parameter('image_height', 480)  # Default height
        self.declare_parameter('frame_rate', 30.0)  # Default frame rate
        
        # Get parameters
        self.video_device = self.get_parameter('video_device').get_parameter_value().string_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.video_device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera device: {self.video_device}")
            raise RuntimeError(f"Cannot open camera device: {self.video_device}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        # Verify camera settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"Camera initialized: {self.video_device}, "
            f"resolution={actual_width}x{actual_height}, fps={actual_fps}"
        )
        
        # Publisher
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Timer for capturing and publishing images
        timer_period = 1.0 / self.frame_rate  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # For debugging: display captured images
        self.debug_window = True  # Set to False to disable debug window
        self.last_frame = None  # Store current frame for color click detection

        if self.debug_window:
            cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Camera Feed", self.on_mouse_click)

    def timer_callback(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to capture image from camera")
                return
            
            # Ensure frame is in BGR format
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Convert to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Publish image
            self.image_pub.publish(image_msg)
            
            # Debug: Show image
            if self.debug_window:
                self.last_frame = frame.copy()
                cv2.imshow("Camera Feed", frame)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.last_frame is not None:
            bgr = self.last_frame[y, x]
            hsv_image = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2HSV)
            hsv = hsv_image[y, x]
            self.get_logger().info(f"Clicked pixel at ({x}, {y}) - BGR: {bgr.tolist()}, HSV: {hsv.tolist()}")

    def destroy_node(self):
        self.get_logger().info("Shutting down camera publisher node")
        self.cap.release()
        if self.debug_window:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    try:
        rclpy.init(args=args)
        node = CameraPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main execution failed: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()