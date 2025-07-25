#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Path to your SDF file
    world_file_path = '/home/aldon/ros2_ws/src/my_robot/MySDF/MyWorld.sdf'  # Update this path
    
    return LaunchDescription([
        
        # Launch Ignition Gazebo with your world
        ExecuteProcess(
            cmd=['gz', 'sim', world_file_path, '-v', '4'],
            output='screen'
        ),
        
        # Bridge for vehicle control (cmd_vel)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
            output='screen'
        ),
        
        # Bridge for camera data
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/camera/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image'],
            output='screen'
        ),
        
        # Bridge for IMU data
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU'],
            output='screen'
        ),
        
        # Optional: Bridge for odometry if needed
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
            output='screen'
        ),
        
        # Optional: Bridge for clock synchronization
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock'],
            output='screen'
        ),
    ])