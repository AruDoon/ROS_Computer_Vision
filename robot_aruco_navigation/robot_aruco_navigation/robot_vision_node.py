#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point, Twist
import cv2
import numpy as np
import math
import json

class RobotVisionNode(Node):
    def __init__(self):
        super().__init__('robot_vision_node')
        
        # Publishers
        self.command_publisher = self.create_publisher(String, 'robot_commands', 10)
        self.robot_pose_publisher = self.create_publisher(Point, 'robot_pose', 10)
        
        # Parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('robot_marker_id', 0)
        self.declare_parameter('arrival_threshold', 30)
        self.declare_parameter('angle_threshold', 15)
        
        # Zoom parameters
        self.zoom_factor = 1.0          # Current zoom level (1.0 = no zoom)
        self.zoom_center = None         # Center point for zooming (x, y)
        self.min_zoom = 0.5            # Minimum zoom level
        self.max_zoom = 3.0            # Maximum zoom level
        self.zoom_step = 0.1           # Zoom increment/decrement step
        
        # Get parameters
        camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        self.robot_marker_id = self.get_parameter('robot_marker_id').get_parameter_value().integer_value
        self.arrival_threshold = self.get_parameter('arrival_threshold').get_parameter_value().integer_value
        self.angle_threshold = self.get_parameter('angle_threshold').get_parameter_value().integer_value
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Robot tracking
        self.robot_pos = None
        self.robot_angle = None
        
        # Navigation target (set by clicking or color detection)
        self.target_pos = None
        
        # Mode selection
        self.auto_mode = False  # False = Manual, True = Auto
        
        # HSV color detection for black objects
        self.hsv_lower = np.array([0, 0, 0])        # Lower HSV bound for black
        self.hsv_upper = np.array([180, 255, 55])    # Upper HSV bound - adjusted for solid black objects
        
        # Distance circles
        self.min_detection_radius = 80   # Grey circle - ignore objects inside
        self.max_detection_radius = 300  # Blue circle - ignore objects outside
        
        # Color detection parameters
        self.min_contour_area = 600     # Minimum area for valid black objects (adjustable)
        self.min_object_width = 10       # Minimum width in pixels
        self.min_object_height = 10      # Minimum height in pixels
        self.show_color_mask = False
        
        # Timer for main loop
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 FPS
        
        # Initialize OpenCV window
        cv2.namedWindow('Robot Navigation')
        cv2.setMouseCallback('Robot Navigation', self.mouse_callback)
        
        self.get_logger().info('Robot Vision Node initialized')
        self.get_logger().info('Instructions:')
        self.get_logger().info('- Attach ArUco marker (ID 0) to your robot')
        self.get_logger().info('- Click on the video feed to set navigation targets (Manual mode)')
        self.get_logger().info('- z/x : Zoom in/out | c : Reset zoom | Right-click : Set zoom center')
        self.get_logger().info('- Press "a" to toggle AUTO/MANUAL mode')
        self.get_logger().info('- Press "+/-" to adjust max detection radius')
        self.get_logger().info('- Press "e/d" to adjust min detection radius')
        self.get_logger().info('- Press "u/j" to adjust min object area')
        self.get_logger().info('- Press "i/k" to adjust HSV sensitivity')
        self.get_logger().info('- Press "m" to show/hide color mask')
        self.get_logger().info('- Press "r" to reset target, "q" to quit')

        
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to set navigation target (Manual mode only) and zoom center"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.auto_mode:
                self.target_pos = (x, y)
                self.get_logger().info(f'Manual target set at: ({x}, {y})')
                # Send stop command when new target is set
                self.send_command("STOP")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to set zoom center
            self.zoom_center = (x, y)
            self.get_logger().info(f'Zoom center set at: ({x}, {y})')
    
    def detect_robot(self, frame):
        """Detect the robot marker"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        self.robot_pos = None
        self.robot_angle = None
        
        if ids is not None and self.robot_marker_id in ids.flatten():
            # Find our robot marker
            robot_index = np.where(ids.flatten() == self.robot_marker_id)[0][0]
            robot_corners = corners[robot_index]
            
            # Calculate robot position (center of marker)
            center = np.mean(robot_corners[0], axis=0)
            self.robot_pos = (int(center[0]), int(center[1]))
            
            # Calculate robot orientation
            # Vector from bottom-left to bottom-right corner
            p1 = robot_corners[0][0]  # bottom-left
            p2 = robot_corners[0][1]  # bottom-right
            
            self.robot_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            
            # Publish robot pose
            pose_msg = Point()
            pose_msg.x = float(self.robot_pos[0])
            pose_msg.y = float(self.robot_pos[1])
            pose_msg.z = self.robot_angle
            self.robot_pose_publisher.publish(pose_msg)
            
            return True
        return False
    
    def detect_black_objects(self, frame):
        """Detect black objects using HSV color space with improved filtering"""
        if not self.robot_pos:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for black color
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Apply morphological operations to clean up the mask
        # Use larger kernel to remove noise and shadows
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        # Additional filtering with erosion and dilation to remove thin shadows
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_objects = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle for size filtering
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by minimum width and height
            if w < self.min_object_width or h < self.min_object_height:
                continue
            
            # Filter by aspect ratio to avoid very thin objects (like shadows)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 5.0:  # Skip very elongated objects
                continue
            
            # Calculate solidity (area / convex hull area) to filter irregular shapes
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.3:  # Skip very irregular shapes (shadows tend to be irregular)
                    continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance from robot
            distance = math.sqrt((cx - self.robot_pos[0])**2 + (cy - self.robot_pos[1])**2)
            
            # Check if object is within detection range
            if distance < self.min_detection_radius or distance > self.max_detection_radius:
                continue
            
            valid_objects.append({
                'position': (cx, cy),
                'distance': distance,
                'area': area,
                'width': w,
                'height': h,
                'solidity': solidity if hull_area > 0 else 0,
                'contour': contour
            })
        
        # Return the closest valid object
        if valid_objects:
            closest_object = min(valid_objects, key=lambda x: x['distance'])
            return closest_object, mask
        
        return None, mask
    
    def calculate_navigation(self):
        """Calculate navigation commands"""
        if not self.robot_pos or not self.target_pos:
            return "NO_TARGET", {}
        
        # Calculate distance to target
        dx = self.target_pos[0] - self.robot_pos[0]
        dy = self.target_pos[1] - self.robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if arrived
        if distance < self.arrival_threshold:
            return "ARRIVED", {"distance": distance}
        
        # Calculate target angle
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate angle difference
        if self.robot_angle is not None:
            angle_diff = target_angle - self.robot_angle
            # Normalize to [-180, 180]
            angle_diff = ((angle_diff + 180) % 360) - 180
            
            # Decide on action
            if abs(angle_diff) > self.angle_threshold:
                direction = "RIGHT" if angle_diff > 0 else "LEFT"
                return "TURN", {
                    "direction": direction, 
                    "angle": abs(angle_diff),
                    "distance": distance
                }
            else:
                return "FORWARD", {"distance": distance, "angle_diff": angle_diff}
        else:
            return "NO_ORIENTATION", {"distance": distance}
    
    def send_command(self, command, params=None):
        """Send command via ROS topic"""
        message = {
            "command": command,
            "params": params if params else {}
        }
        
        msg = String()
        msg.data = json.dumps(message)
        self.command_publisher.publish(msg)
        
        self.get_logger().info(f'Sent command: {command}')
    
    def execute_navigation_command(self, command, params):
        """Execute navigation command by publishing to ROS topic"""
        if command == "FORWARD":
            self.send_command("FORWARD")
                
        elif command == "TURN":
            direction = params.get("direction", "LEFT")
            self.send_command("TURN", {"direction": direction})
                
        elif command == "ARRIVED":
            self.send_command("STOP")
            self.get_logger().info('Robot ARRIVED - sent STOP command')
                
        elif command in ["NO_TARGET", "NO_ORIENTATION"]:
            self.send_command("STOP")
    
    def draw_detection_circles(self, frame):
        """Draw detection radius circles"""
        if not self.robot_pos:
            return frame
        
        # Draw minimum detection radius (grey circle)
        cv2.circle(frame, self.robot_pos, self.min_detection_radius, (128, 128, 128), 2)
        
        # Draw maximum detection radius (blue circle)
        cv2.circle(frame, self.robot_pos, self.max_detection_radius, (255, 0, 0), 2)
        
        return frame
    
    def draw_visualization(self, frame):
        """Draw robot, target, and navigation info"""
        # Draw detection circles first
        frame = self.draw_detection_circles(frame)
        
        # Draw robot
        if self.robot_pos:
            cv2.circle(frame, self.robot_pos, 15, (0, 255, 0), -1)
            cv2.putText(frame, "ROBOT", 
                       (self.robot_pos[0] - 30, self.robot_pos[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw robot orientation arrow
            if self.robot_angle is not None:
                arrow_length = 40
                end_x = int(self.robot_pos[0] + arrow_length * math.cos(math.radians(self.robot_angle)))
                end_y = int(self.robot_pos[1] + arrow_length * math.sin(math.radians(self.robot_angle)))
                cv2.arrowedLine(frame, self.robot_pos, (end_x, end_y), (0, 255, 0), 3)
        
        # Display zoom info
        zoom_info = f"Zoom: {self.zoom_factor:.1f}x"
        if self.zoom_center:
            zoom_info += f" | Center: ({self.zoom_center[0]}, {self.zoom_center[1]})"
        cv2.putText(frame, zoom_info, 
                   (frame.shape[1] - 300, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Draw target
        if self.target_pos:
            color = (0, 255, 255) if self.auto_mode else (0, 0, 255)  # Yellow for auto, red for manual
            cv2.circle(frame, self.target_pos, 10, color, -1)
            label = "AUTO TARGET" if self.auto_mode else "TARGET"
            cv2.putText(frame, label, 
                       (self.target_pos[0] - 40, self.target_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw navigation line
        if self.robot_pos and self.target_pos:
            color = (255, 255, 0) if self.auto_mode else (255, 0, 255)
            cv2.line(frame, self.robot_pos, self.target_pos, color, 2)
        
        # Draw mode indicator
        mode_text = "AUTO" if self.auto_mode else "MANUAL"
        mode_color = (0, 255, 255) if self.auto_mode else (255, 255, 255)
        cv2.putText(frame, f"Mode: {mode_text}", 
                   (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Draw instructions
        instructions = [
            "Press 'a' to toggle AUTO/MANUAL mode",
            f"Robot ID: {self.robot_marker_id} (attach marker to robot)",
            "Manual: Click to set target | Auto: Follows black objects",
            "+/- : Max radius | e/d : Min radius | u/j : Min area",
            "i/k : HSV sensitivity | m : Show mask | r : Reset | q : Quit",
            "z/x : Zoom in/out | c : Reset zoom | Right-click : Set zoom center"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_black_objects(self, frame, detection_result):
        """Draw detected black objects"""
        if detection_result is None:
            return frame
        
        black_object, mask = detection_result
        
        if black_object:
            # Draw the detected object
            cv2.drawContours(frame, [black_object['contour']], -1, (0, 255, 255), 2)
            
            # Draw center point
            pos = black_object['position']
            cv2.circle(frame, pos, 8, (0, 255, 255), -1)
            
            # Draw info text with more details
            info_text = f"Area: {black_object['area']:.0f} | Size: {black_object['width']}x{black_object['height']}"
            cv2.putText(frame, info_text, 
                       (pos[0] - 80, pos[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            dist_text = f"Dist: {black_object['distance']:.0f}px | Sol: {black_object['solidity']:.2f}"
            cv2.putText(frame, dist_text, 
                       (pos[0] - 80, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def draw_navigation_info(self, frame, command, params):
        """Draw navigation command info"""
        # Display current command
        command_color = {
            "FORWARD": (0, 255, 0),
            "TURN": (0, 255, 255),
            "ARRIVED": (255, 0, 255),
            "NO_TARGET": (128, 128, 128),
            "NO_ORIENTATION": (0, 0, 255)
        }
        
        color = command_color.get(command, (255, 255, 255))
        
        cv2.putText(frame, f"Command: {command}", 
                   (10, frame.shape[0] - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display parameters
        if params:
            param_text = []
            for key, value in params.items():
                if isinstance(value, float):
                    param_text.append(f"{key}: {value:.1f}")
                else:
                    param_text.append(f"{key}: {value}")
            
            param_str = ", ".join(param_text)
            cv2.putText(frame, param_str, 
                       (10, frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display detection parameters
        param_info = f"Min Area: {self.min_contour_area} | HSV Upper: {self.hsv_upper[2]} | Min Size: {self.min_object_width}x{self.min_object_height}"
        cv2.putText(frame, param_info, 
                   (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Display robot info
        if self.robot_pos and self.robot_angle is not None:
            robot_info = f"Robot: ({self.robot_pos[0]}, {self.robot_pos[1]}) @ {self.robot_angle:.1f}°"
            cv2.putText(frame, robot_info, 
                       (10, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    def apply_zoom(self, frame):
        """Apply zoom to the frame"""
        if self.zoom_factor == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        
        # Set zoom center to frame center if not set
        if self.zoom_center is None:
            self.zoom_center = (width // 2, height // 2)
        
        # Calculate the size of the region to extract
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)
        
        # Calculate the top-left corner of the region to extract
        x1 = max(0, self.zoom_center[0] - new_width // 2)
        y1 = max(0, self.zoom_center[1] - new_height // 2)
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)
        
        # Adjust if we're at the edges
        if x2 - x1 < new_width:
            x1 = max(0, x2 - new_width)
        if y2 - y1 < new_height:
            y1 = max(0, y2 - new_height)
        
        # Extract the region of interest
        roi = frame[y1:y2, x1:x2]
        
        # Resize back to original dimensions
        zoomed_frame = cv2.resize(roi, (width, height))
        
        return zoomed_frame
    
    def timer_callback(self):
        """Main vision processing loop"""
        ret, frame = self.cap.read()
        # Apply zoom to the frame
        frame = self.apply_zoom(frame)
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return
        
        # Detect robot
        robot_detected = self.detect_robot(frame)
        
        if robot_detected:
            # Draw detected ArUco markers
            corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Auto mode: detect black objects
            if self.auto_mode:
                detection_result = self.detect_black_objects(frame)
                
                frame = self.draw_black_objects(frame, detection_result)
                
                if detection_result[0] is not None:  
                    black_object = detection_result[0]
                    self.target_pos = black_object['position']
                elif not hasattr(self, '_last_detection_time') or \
                     (hasattr(self, '_last_detection_time') and cv2.getTickCount() - self._last_detection_time > cv2.getTickFrequency() * 2):
                    self.target_pos = None
                
                if detection_result[0] is not None:
                    self._last_detection_time = cv2.getTickCount()
                
                if self.show_color_mask and detection_result[1] is not None:
                    mask_display = cv2.applyColorMap(detection_result[1], cv2.COLORMAP_JET)
                    cv2.imshow('Color Mask', mask_display)
        
        # Calculate navigation
        command, params = self.calculate_navigation()
        
        # Execute navigation command
        if robot_detected:
            self.execute_navigation_command(command, params)
        
        # Draw visualization
        frame = self.draw_visualization(frame)
        self.draw_navigation_info(frame, command, params)
        
        # Display frame
        cv2.imshow('Robot Navigation', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Shutting down vision node...')
            rclpy.shutdown()
        elif key == ord('r'):
            self.target_pos = None
            self.send_command("STOP")
            self.get_logger().info('Target reset')
        elif key == ord('a'):
            self.auto_mode = not self.auto_mode
            mode = "AUTO" if self.auto_mode else "MANUAL"
            self.get_logger().info(f'Switched to {mode} mode')
            if self.auto_mode:
                self.target_pos = None  # Clear manual target when switching to auto
        elif key == ord('+') or key == ord('='):
            self.max_detection_radius = min(500, self.max_detection_radius + 20)
            self.get_logger().info(f'Max detection radius: {self.max_detection_radius}px')
        elif key == ord('-'):
            self.max_detection_radius = max(self.min_detection_radius + 50, self.max_detection_radius - 20)
            self.get_logger().info(f'Max detection radius: {self.max_detection_radius}px')
        elif key == ord('e'):
            self.min_detection_radius = min(self.max_detection_radius - 50, self.min_detection_radius + 10)
            self.get_logger().info(f'Min detection radius: {self.min_detection_radius}px')
        elif key == ord('d'):
            self.min_detection_radius = max(20, self.min_detection_radius - 10)
            self.get_logger().info(f'Min detection radius: {self.min_detection_radius}px')
        elif key == ord('u'):
            self.min_contour_area = min(5000, self.min_contour_area + 200)
            self.get_logger().info(f'Min object area: {self.min_contour_area}px²')
        elif key == ord('j'):
            self.min_contour_area = max(500, self.min_contour_area - 200)
            self.get_logger().info(f'Min object area: {self.min_contour_area}px²')
        elif key == ord('i'):
            # Decrease HSV sensitivity (higher V value = less sensitive)
            self.hsv_upper[2] = min(120, self.hsv_upper[2] + 10)
            self.get_logger().info(f'HSV sensitivity decreased (V upper: {self.hsv_upper[2]})')
        elif key == ord('k'):
            # Increase HSV sensitivity (lower V value = more sensitive)
            self.hsv_upper[2] = max(30, self.hsv_upper[2] - 10)
            self.get_logger().info(f'HSV sensitivity increased (V upper: {self.hsv_upper[2]})')
        elif key == ord('m'):
            self.show_color_mask = not self.show_color_mask
            if not self.show_color_mask:
                cv2.destroyWindow('Color Mask')
            self.get_logger().info(f'Color mask display: {"ON" if self.show_color_mask else "OFF"}')
        #zoom
        elif key == ord('z'):
            # Zoom in
            self.zoom_factor = min(self.max_zoom, self.zoom_factor + self.zoom_step)
            self.get_logger().info(f'Zoom: {self.zoom_factor:.1f}x')
        elif key == ord('x'):
            # Zoom out
            self.zoom_factor = max(self.min_zoom, self.zoom_factor - self.zoom_step)
            self.get_logger().info(f'Zoom: {self.zoom_factor:.1f}x')
        elif key == ord('c'):
            # Reset zoom
            self.zoom_factor = 1.0
            self.zoom_center = None
            self.get_logger().info('Zoom reset to 1.0x')
    
    def destroy_node(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = RobotVisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()