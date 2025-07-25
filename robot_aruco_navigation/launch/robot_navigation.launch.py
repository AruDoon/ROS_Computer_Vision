#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID'
    )
    
    esp_ip_arg = DeclareLaunchArgument(
        'esp_ip',
        default_value='192.168.43.222',
        # default_value='192.168.179.88',192.168.43.222
        description='ESP32 IP address'
    )
    
    esp_port_arg = DeclareLaunchArgument(
        'esp_port',
        default_value='8080',
        description='ESP32 TCP port'
    )
    
    robot_marker_id_arg = DeclareLaunchArgument(
        'robot_marker_id',
        default_value='0',
        description='ArUco marker ID for the robot'
    )
    
    arrival_threshold_arg = DeclareLaunchArgument(
        'arrival_threshold',
        default_value='30',
        description='Distance threshold for arrival detection (pixels)'
    )
    
    angle_threshold_arg = DeclareLaunchArgument(
        'angle_threshold',
        default_value='15',
        description='Angle threshold for turning decisions (degrees)'
    )
    
    # Vision node
    vision_node = Node(
        package='robot_aruco_navigation',
        executable='vision_node',
        name='robot_vision_node',
        output='screen',
        parameters=[{
            'camera_id': LaunchConfiguration('camera_id'),
            'robot_marker_id': LaunchConfiguration('robot_marker_id'),
            'arrival_threshold': LaunchConfiguration('arrival_threshold'),
            'angle_threshold': LaunchConfiguration('angle_threshold'),
        }]
    )
    
    # Bridge node
    bridge_node = Node(
        package='robot_aruco_navigation',
        executable='bridge_node',
        name='robot_bridge_node',
        output='screen',
        parameters=[{
            'esp_ip': LaunchConfiguration('esp_ip'),
            'esp_port': LaunchConfiguration('esp_port'),
            'command_cooldown': 0.1,
        }]
    )
    
    return LaunchDescription([
        camera_id_arg,
        esp_ip_arg,
        esp_port_arg,
        robot_marker_id_arg,
        arrival_threshold_arg,
        angle_threshold_arg,
        vision_node,
        bridge_node,
    ])