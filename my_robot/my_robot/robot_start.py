import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from cv_bridge import CvBridge
import cv2
import numpy as np
from heapq import heappush, heappop
import math
#from icecream import ic
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class ChaseTargetNode(Node):
    def __init__(self):
        super().__init__('chase_target_node')
        self.bridge = CvBridge()
        
        best_effort_qos = QoSProfile(
            depth=10,  # Queue size
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.orientation_sub = self.create_subscription(
            Vector3, '/imu/orientation', self.orientation_callback, qos_profile=best_effort_qos)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Robot state
        self.robot_pos = None  # (x, y) in world coordinates
        self.target_pos = None  # (x, y) in world coordinates
        self.yaw_degrees = 0.0  # Current orientation from IMU in degrees
        self.grid_map = None  # 2D grid for path planning
        self.dilated_grid_map = None  # Dilated grid map with obstacle clearance
        self.path = []  # List of waypoints
        self.current_waypoint_index = 0  # Track which waypoint we're heading to
        
        # Camera parameters
        self.image_width = 640
        self.image_height = 480
        self.world_width = 10.0   # x: -5 to 5 meters
        self.world_height = 10.0  # y: -5 to 5 meters
        
        # Grid parameters
        self.cell_size = 0.1  # 0.1m per cell
        self.grid_cols = int(self.world_width / self.cell_size)  # 100
        self.grid_rows = int(self.world_height / self.cell_size)  # 100
        
        # FIXED: Reasonable obstacle clearance parameters
        self.obstacle_clearance = 0.8 # meters - minimum distance from obstacles (was 15.0!)
        self.clearance_cells = int(self.obstacle_clearance / self.cell_size)  # cells (now ~3 cells)
        
        # Control parameters - TUNED FOR BETTER FOLLOWING (now in degrees)
        self.linear_speed = 0.5  # Reduced for better control
        self.angular_gain = 1.0  # Increased for more responsive turning
        self.waypoint_threshold = 0.2  # Distance to consider waypoint reached
        self.heading_tolerance_degrees = 35.0  # Degrees (was 0.6 radians ~34 degrees)
        
        # Safety flags
        self.last_robot_pos = None
        self.last_target_pos = None
        self.detection_confidence = 0
        
        # Timers
        self.create_timer(2.0, self.planning_callback)  # 0.5 Hz for planning (slower)
        self.create_timer(0.05, self.control_callback)  # 20 Hz for control (faster)

    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        px = ((5.0 - y) / self.world_height) * self.image_width
        py = self.image_height - ((x + 5.0) / self.world_width) * self.image_height
        return int(px), int(py)

    def pixel_to_world(self, px, py):
        """Convert pixel coordinates to world coordinates."""
        x_world = -5.0 + ((self.image_height - py) / self.image_height) * self.world_width
        y_world = 5.0 - (px / self.image_width) * self.world_height
        return x_world, y_world

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int((x + 5.0) / self.cell_size)
        grid_y = int((y + 5.0) / self.cell_size)
        return max(0, min(grid_x, self.grid_cols - 1)), max(0, min(grid_y, self.grid_rows - 1))

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates (center of cell)."""
        x = -5.0 + (grid_x + 0.5) * self.cell_size
        y = -5.0 + (grid_y + 0.5) * self.cell_size
        return x, y

    def is_valid_position(self, pos):
        """Check if position is within world bounds."""
        x, y = pos
        return (-5.0 <= x <= 5.0 and -5.0 <= y <= 5.0)

    def normalize_angle_degrees(self, angle_deg):
        """Normalize angle to [-180, 180] degrees."""
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    def dilate_obstacles(self, grid_map):
        """IMPROVED: Dilate obstacles to create clearance buffer with better error handling."""
        try:
            if self.clearance_cells <= 0:
                return grid_map.copy()
            
            # Create a structuring element (circular kernel) for dilation
            kernel_size = 2 * self.clearance_cells + 1
            # Ensure kernel size is reasonable
            kernel_size = min(kernel_size, 21)  # Cap at 21x21 kernel
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Convert grid_map to uint8 format (0 or 255)
            binary_map = (grid_map * 255).astype(np.uint8)
            
            # Dilate the obstacles
            dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
            
            # Convert back to binary (0 or 1)
            dilated_grid = (dilated_map > 0).astype(np.uint8)
            
            # Debug: Log dilation info
            obstacle_count_original = np.sum(grid_map)
            obstacle_count_dilated = np.sum(dilated_grid)
            self.get_logger().info(f"Obstacle dilation: {obstacle_count_original} -> {obstacle_count_dilated} cells "
                                 f"(clearance: {self.obstacle_clearance}m, kernel: {kernel_size}x{kernel_size}) Yaw {self.yaw_degrees}")
            
            return dilated_grid
            
        except Exception as e:
            self.get_logger().error(f"Obstacle dilation failed: {e}")
            return grid_map.copy()  # Return copy of original if dilation fails

    def image_callback(self, msg):
        """Process camera image to detect robot, target, and obstacles."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detect black (target)
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 110])
            black_mask = cv2.inRange(hsv_image, black_lower, black_upper)
            
            # Detect blue (robot)
            blue_lower = np.array([100, 150, 0])
            blue_upper = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
            
            # Detect red (obstacles)
            red_lower1 = np.array([0, 120, 70])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 120, 70])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
            red_mask = red_mask1 + red_mask2
            
            # Find robot (largest blue contour) 
            robot_detected = False
            robot_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if robot_contours:
                robot_contour = max(robot_contours, key=cv2.contourArea)
                if cv2.contourArea(robot_contour) > 50:
                    M = cv2.moments(robot_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        new_robot_pos = self.pixel_to_world(cx, cy)
                        
                        if self.is_valid_position(new_robot_pos):
                            self.robot_pos = new_robot_pos
                            self.last_robot_pos = new_robot_pos
                            robot_detected = True
                            
                            # Draw robot
                            cv2.drawContours(cv_image, [robot_contour], -1, (0, 255, 0), 2)
                            cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(cv_image, f"Robot ({new_robot_pos[0]:.1f},{new_robot_pos[1]:.1f})", 
                                      (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if not robot_detected and self.last_robot_pos is not None:
                self.robot_pos = self.last_robot_pos
            
            # Find target (black circle)
            target_detected = False
            target_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if target_contours:
                for contour in target_contours:
                    # Skip small contours
                    if cv2.contourArea(contour) < 50:
                        continue
                        
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    if perimeter == 0:
                        continue
                        
                    circularity = 4 * math.pi * (area / (perimeter * perimeter))
                    
                    # Only accept contours with high circularity (circle-like shapes)
                    if circularity > 0.7:  # Threshold for circle detection (1.0 is perfect circle)
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            new_target_pos = self.pixel_to_world(cx, cy)
                            
                            if self.is_valid_position(new_target_pos):
                                self.target_pos = new_target_pos
                                self.last_target_pos = new_target_pos
                                target_detected = True
                                
                                # Draw target with circle verification
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                center = (int(x), int(y))
                                radius = int(radius)
                                cv2.circle(cv_image, center, radius, (255, 0, 0), 2)
                                cv2.circle(cv_image, center, 5, (255, 0, 0), -1)
                                cv2.putText(cv_image, f"Circle Target ({new_target_pos[0]:.1f},{new_target_pos[1]:.1f})", 
                                        (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                break  # Stop after finding first valid circle
            
            if not target_detected and self.last_target_pos is not None:
                self.target_pos = self.last_target_pos
            
            # Update detection confidence
            if robot_detected and target_detected:
                self.detection_confidence = min(self.detection_confidence + 1, 10)
            else:
                self.detection_confidence = max(self.detection_confidence - 1, 0)
            
            # Draw obstacles
            obstacle_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in obstacle_contours:
                if cv2.contourArea(contour) > 50:
                    cv2.drawContours(cv_image, [contour], -1, (0, 0, 255), 2)
            
            # IMPROVED: Draw both original and dilated obstacle boundaries for visualization
            if hasattr(self, 'grid_map') and self.grid_map is not None:
                # Draw original obstacles in red
                original_obstacles = np.where(self.grid_map == 1)
                for gy, gx in zip(original_obstacles[0], original_obstacles[1]):
                    world_x, world_y = self.grid_to_world(gx, gy)
                    px, py = self.world_to_pixel(world_x, world_y)
                    if 0 <= px < self.image_width and 0 <= py < self.image_height:
                        cv2.circle(cv_image, (px, py), 2, (0, 0, 255), -1)
                        
                # Draw dilated obstacles in orange for clearance visualization
                if hasattr(self, 'dilated_grid_map') and self.dilated_grid_map is not None:
                    dilated_obstacles = np.where(self.dilated_grid_map == 1)
                    for gy, gx in zip(dilated_obstacles[0], dilated_obstacles[1]):
                        world_x, world_y = self.grid_to_world(gx, gy)
                        px, py = self.world_to_pixel(world_x, world_y)
                        if 0 <= px < self.image_width and 0 <= py < self.image_height:
                            cv2.circle(cv_image, (px, py), 1, (0, 165, 255), -1)  # Orange
            
            # Draw planned path with waypoint numbers
            if self.path and len(self.path) > 1:
                for i in range(len(self.path) - 1):
                    p1 = self.world_to_pixel(*self.path[i])
                    p2 = self.world_to_pixel(*self.path[i + 1])
                    if (0 <= p1[0] < self.image_width and 0 <= p1[1] < self.image_height and
                        0 <= p2[0] < self.image_width and 0 <= p2[1] < self.image_height):
                        cv2.line(cv_image, p1, p2, (0, 255, 255), 2)  # Thicker line for visibility
                
                # Highlight current target waypoint
                if self.current_waypoint_index < len(self.path):
                    wp_pixel = self.world_to_pixel(*self.path[self.current_waypoint_index])
                    if (0 <= wp_pixel[0] < self.image_width and 0 <= wp_pixel[1] < self.image_height):
                        cv2.circle(cv_image, wp_pixel, 8, (255, 255, 0), -1)
                        cv2.putText(cv_image, f"WP{self.current_waypoint_index}", 
                                  (wp_pixel[0] + 10, wp_pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display status info (updated to show degrees)
            cv2.putText(cv_image, f"Confidence: {self.detection_confidence}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Path waypoints: {len(self.path)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Current waypoint: {self.current_waypoint_index}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Yaw: {self.yaw_degrees:.1f}°", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Clearance: {self.obstacle_clearance:.1f}m", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cv_image, f"Heading tolerance: {self.heading_tolerance_degrees:.1f}°", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Chase Target", cv_image)
            cv2.waitKey(1)
            
            # Create grid map with obstacles
            try:
                self.grid_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
                obstacle_pixels = np.where(red_mask > 0)
                for py, px in zip(obstacle_pixels[0], obstacle_pixels[1]):
                    world_x, world_y = self.pixel_to_world(px, py)
                    grid_x, grid_y = self.world_to_grid(world_x, world_y)
                    if 0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows:
                        self.grid_map[grid_y, grid_x] = 1
                
                # Create dilated grid map for path planning
                self.dilated_grid_map = self.dilate_obstacles(self.grid_map)
                
            except Exception as e:
                self.get_logger().error(f"Grid map creation failed: {e}")
                
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

    def orientation_callback(self, msg):
        """Update robot orientation from /imu/orientation topic (Vector3 with yaw in degrees)."""
        try:
            self.yaw_degrees = msg.x
         #   ic(self.yaw_degrees)
            self.get_logger().debug(f"Received yaw: {self.yaw_degrees:.1f}°")
        except Exception as e:
            self.get_logger().error(f"Orientation processing failed: {e}")

    def a_star(self, start, goal):
        """FIXED: A* path planning that properly handles robot inside dilated areas."""
        try:
            if self.dilated_grid_map is None:
                self.get_logger().warn("No dilated grid map available")
                return []
                
            if not (0 <= start[0] < self.grid_cols and 0 <= start[1] < self.grid_rows and
                    0 <= goal[0] < self.grid_cols and 0 <= goal[1] < self.grid_rows):
                self.get_logger().warn(f"Start {start} or goal {goal} out of bounds")
                return []
            
            # Always use dilated grid for planning, but handle special cases
            planning_grid = self.dilated_grid_map.copy()
            
            # SOLUTION 1: Allow movement FROM current position even if in dilated area
            # This lets the robot escape from dilated zones but still maintains clearance for the rest of the path
            original_start_blocked = planning_grid[start[1], start[0]] == 1
            if original_start_blocked:
                # Temporarily allow the start position for planning
                planning_grid[start[1], start[0]] = 0
                self.get_logger().warn("Robot is in dilated obstacle area - allowing escape from current position")
            
            # Check if goal is in actual obstacle (not just dilated area)
            if self.grid_map is not None and self.grid_map[goal[1], goal[0]] == 1:
                self.get_logger().error("Goal is inside an actual obstacle!")
                return []
            
            # If goal is in dilated area but not actual obstacle, allow it temporarily for planning
            if planning_grid[goal[1], goal[0]] == 1:
                if self.grid_map is not None and self.grid_map[goal[1], goal[0]] == 0:
                    planning_grid[goal[1], goal[0]] = 0
                    self.get_logger().warn("Goal is in dilated area but not actual obstacle - allowing path to goal")
            
            open_set = []
            heappush(open_set, (0, start))
            came_from = {}
            g_score = {start: 0}
            f_score = {start: self.heuristic(start, goal)}
            
            max_iterations = 10000
            iterations = 0
            
            while open_set and iterations < max_iterations:
                iterations += 1
                current = heappop(open_set)[1]
                
                if current == goal:
                    path = self.reconstruct_path(came_from, current)
                    self.get_logger().info(f"Path found with dilated grid: {len(path)} waypoints, {iterations} iterations")
                    return path
                
                # 8-connectivity for smoother paths
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if (0 <= neighbor[0] < self.grid_cols and                   
                        0 <= neighbor[1] < self.grid_rows and 
                        planning_grid[neighbor[1], neighbor[0]] == 0):
                        
                        # Cost is higher for diagonal moves
                        move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                        tentative_g = g_score[current] + move_cost
                        
                        if neighbor not in g_score or tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                            heappush(open_set, (f_score[neighbor], neighbor))
            
            self.get_logger().warn(f"No path found with dilated grid after {iterations} iterations")
            return []
            
        except Exception as e:
            self.get_logger().error(f"A* planning failed: {e}")
            return []

    def find_nearest_free_space(self, position, max_search_radius=20):
        """Find the nearest free space in the dilated grid from a given position."""
        if self.dilated_grid_map is None:
            return position
        
        start_grid = self.world_to_grid(*position)
        
        # If already free, return as-is
        if (0 <= start_grid[0] < self.grid_cols and 0 <= start_grid[1] < self.grid_rows and
            self.dilated_grid_map[start_grid[1], start_grid[0]] == 0):
            return position
        
        # Search in expanding circles
        for radius in range(1, max_search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        check_x = start_grid[0] + dx
                        check_y = start_grid[1] + dy
                        
                        if (0 <= check_x < self.grid_cols and 0 <= check_y < self.grid_rows and
                            self.dilated_grid_map[check_y, check_x] == 0):
                            return self.grid_to_world(check_x, check_y)
        
        # If no free space found, return original position
        return position

    def planning_callback(self):
        """IMPROVED: Plan the path with better handling of dilated obstacles."""
        try:
            if (self.robot_pos is None or self.target_pos is None or 
                self.dilated_grid_map is None or self.detection_confidence < 3):
                return
            
            # Check if robot is in dilated obstacle area
            robot_grid = self.world_to_grid(*self.robot_pos)
            robot_in_dilated = (self.dilated_grid_map[robot_grid[1], robot_grid[0]] == 1)
            
            start_grid = robot_grid
            goal_grid = self.world_to_grid(*self.target_pos)
            
            # ALTERNATIVE SOLUTION 2: Move to nearest free space if robot is stuck
            if robot_in_dilated:
                nearest_free = self.find_nearest_free_space(self.robot_pos)
                if nearest_free != self.robot_pos:
                    start_grid = self.world_to_grid(*nearest_free)
                    self.get_logger().info(f"Robot in dilated area, planning from nearest free space: {nearest_free}")
            
            # Check if already at target
            if self.heuristic(start_grid, goal_grid) < 2:
                self.path = []
                self.current_waypoint_index = 0
                return
                
            new_path = self.a_star(start_grid, goal_grid)
            if new_path and len(new_path) > 1:
                # If we planned from a different start point, add current robot position as first waypoint
                if robot_in_dilated and start_grid != robot_grid:
                    new_path.insert(0, self.robot_pos)
                
                self.path = new_path
                self.current_waypoint_index = 0
                self.get_logger().info(f"New path planned with {len(self.path)} waypoints (clearance: {self.obstacle_clearance}m)")
            else:
                self.get_logger().warn("No path found to target with current clearance")
                
        except Exception as e:
            self.get_logger().error(f"Planning callback failed: {e}")

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from A*."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        world_path = [(self.grid_to_world(px, py)) for px, py in path]
        return world_path

    def control_callback(self):
        """IMPROVED: Control the robot to follow the path step by step (now using degrees)."""
        try:
            twist = Twist()  # Default: stop
            
            # Only control if we have confident detections and a path
            if (not self.path or self.robot_pos is None or 
                self.detection_confidence < 2 or 
                self.current_waypoint_index >= len(self.path)):
                self.cmd_vel_pub.publish(twist)
                return
            
            # Get current target waypoint
            target_waypoint = self.path[self.current_waypoint_index]
            
            # Calculate distance and angle to current waypoint
            dx = target_waypoint[0] - self.robot_pos[0]
            dy = target_waypoint[1] - self.robot_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Check if we've reached the current waypoint
            if distance < self.waypoint_threshold:
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}")
                self.current_waypoint_index += 1
                
                # Check if we've reached the end of the path
                if self.current_waypoint_index >= len(self.path):
                    self.get_logger().info("Reached end of path!")
                    self.cmd_vel_pub.publish(twist)  # Stop
                    return
                
                # Move to next waypoint
                target_waypoint = self.path[self.current_waypoint_index]
                dx = target_waypoint[0] - self.robot_pos[0]
                dy = target_waypoint[1] - self.robot_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
            
            # Calculate desired heading to current waypoint (in degrees)
            desired_yaw_degrees = math.degrees(math.atan2(dy, dx))
            
            # Calculate heading error (in degrees)
            heading_error_degrees = desired_yaw_degrees - self.yaw_degrees
            
            # Normalize heading error to [-180, 180] degrees
            heading_error_degrees = self.normalize_angle_degrees(heading_error_degrees)
            
            # Set angular velocity to correct heading (convert to rad/s for Twist message)
            max_angular_vel = 1.0  # rad/s
            angular_velocity_radians = math.radians(self.angular_gain * heading_error_degrees)
            twist.angular.z = max(-max_angular_vel, 
                                min(max_angular_vel, angular_velocity_radians))
            
            # Only move forward if we're facing roughly the right direction
            if abs(heading_error_degrees) < self.heading_tolerance_degrees:
                # Move forward with speed proportional to distance (but capped)
                forward_speed = min(self.linear_speed, distance * 2.0)
                twist.linear.x = forward_speed
            else:
                # Don't move forward while turning
                twist.linear.x = 0.0
            
            # Publish the command
            self.cmd_vel_pub.publish(twist)
            
            # Debug output (every 20 calls = ~1 second at 20Hz)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 20 == 0:
                self.get_logger().info(
                    f"WP{self.current_waypoint_index}/{len(self.path)-1}: "
                    f"dist={distance:.2f}m, heading_err={heading_error_degrees:.1f}°, "
                    f"linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
                                      
        except Exception as e:
            self.get_logger().error(f"Control callback failed: {e}")
            twist = Twist()
            self.cmd_vel_pub.publish(twist)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = ChaseTargetNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Main execution failed: {e}")
    finally:
        try:
            if 'node' in locals():
                node.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == '__main__':
    main()