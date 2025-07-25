#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import threading
import time
import json

class RobotBridgeNode(Node):
    def __init__(self):
        super().__init__('robot_bridge_node')
        
        # Parameters
        self.declare_parameter('esp_ip', '192.168.43.222')
        # self.declare_parameter('esp_ip', '192.168.179.88')
        self.declare_parameter('esp_port', 8080)
        self.declare_parameter('command_cooldown', 0.1)
        
        # Get parameters
        self.esp_ip = self.get_parameter('esp_ip').get_parameter_value().string_value
        self.esp_port = self.get_parameter('esp_port').get_parameter_value().integer_value
        self.command_cooldown = self.get_parameter('command_cooldown').get_parameter_value().double_value
        
        # TCP Communication
        self.tcp_socket = None
        self.connection_status = "DISCONNECTED"
        self.last_command_time = 0
        
        # Subscriber to robot commands
        self.command_subscriber = self.create_subscription(
            String,
            'robot_commands',
            self.command_callback,
            10
        )
        
        # Publisher for connection status
        self.status_publisher = self.create_publisher(String, 'esp_connection_status', 10)
        
        # Timer for status publishing and connection monitoring
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.reconnect_timer = self.create_timer(5.0, self.check_connection)
        
        # Initialize TCP connection
        self.connect_to_esp()
        
        self.get_logger().info('Robot Bridge Node initialized')
        self.get_logger().info(f'Connecting to ESP32 at {self.esp_ip}:{self.esp_port}')
    
    def connect_to_esp(self):
        """Connect to ESP32 via TCP"""
        try:
            if self.tcp_socket:
                self.tcp_socket.close()
                
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(5)  # 5 second timeout
            self.tcp_socket.connect((self.esp_ip, self.esp_port))
            self.connection_status = "CONNECTED"
            self.get_logger().info(f'✅ Connected to ESP32 at {self.esp_ip}:{self.esp_port}')
        except Exception as e:
            self.connection_status = "FAILED"
            self.get_logger().error(f'❌ Failed to connect to ESP32: {e}')
            self.tcp_socket = None
    
    def send_command_to_esp(self, command, params=None):
        """Send command to ESP32"""
        if not self.tcp_socket or self.connection_status != "CONNECTED":
            self.get_logger().warn(f'Cannot send command {command}: ESP32 not connected')
            return False
            
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False
            
        try:
            # Create command message
            message = {
                "command": command,
                "params": params if params else {}
            }
            
            message_str = json.dumps(message) + "\n"
            self.tcp_socket.send(message_str.encode())
            self.last_command_time = current_time
            self.get_logger().debug(f'Sent to ESP32: {command}')
            return True
            
        except Exception as e:
            self.get_logger().error(f'❌ Failed to send command: {e}')
            self.connection_status = "DISCONNECTED"
            return False
    
    def command_callback(self, msg):
        """Handle incoming robot commands from vision node"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get("command", "")
            params = command_data.get("params", {})
            
            self.get_logger().info(f'Received command: {command}')
            
            # Send command to ESP32
            success = self.send_command_to_esp(command, params)
            
            if success:
                if command == "FORWARD":
                    self.get_logger().info('🤖 Sent FORWARD command to ESP32')
                elif command == "TURN":
                    direction = params.get("direction", "LEFT")
                    self.get_logger().info(f'🤖 Sent TURN {direction} command to ESP32')
                elif command == "STOP":
                    self.get_logger().info('🛑 Sent STOP command to ESP32')
            else:
                self.get_logger().warn(f'Failed to send {command} command to ESP32')
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in command message: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
    
    def publish_status(self):
        """Publish connection status"""
        status_msg = String()
        status_msg.data = self.connection_status
        self.status_publisher.publish(status_msg)
    
    def check_connection(self):
        """Check and maintain connection to ESP32"""
        if self.connection_status != "CONNECTED":
            self.get_logger().info('Attempting to reconnect to ESP32...')
            self.connect_to_esp()
    
    def test_connection(self):
        """Test ESP32 connection by sending a ping"""
        if self.connection_status == "CONNECTED":
            try:
                # Send a simple test message
                test_message = {"command": "PING", "params": {}}
                message_str = json.dumps(test_message) + "\n"
                self.tcp_socket.send(message_str.encode())
                return True
            except Exception as e:
                self.get_logger().warn(f'Connection test failed: {e}')
                self.connection_status = "DISCONNECTED"
                return False
        return False
    
    def destroy_node(self):
        """Clean up resources"""
        if self.tcp_socket:
            try:
                # Send final STOP command
                self.send_command_to_esp("STOP")
                time.sleep(0.1)  # Give time for command to be sent
                self.tcp_socket.close()
                self.get_logger().info('TCP connection closed')
            except Exception as e:
                self.get_logger().error(f'Error closing TCP connection: {e}')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = RobotBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()