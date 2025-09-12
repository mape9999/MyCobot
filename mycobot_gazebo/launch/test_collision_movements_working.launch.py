#!/usr/bin/env python3
"""
Launch file for testing collision movements with WORKING collision detection.

This launch file uses a simple geometric detector that actually works
to detect self-collisions for PPO training integration.

:author: AI Assistant
:date: December 2024
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate launch description for collision movement testing with WORKING detection.
    
    Returns:
        LaunchDescription: Complete launch description for movement testing
    """
    # Package paths
    pkg_share_gazebo = FindPackageShare('mycobot_gazebo').find('mycobot_gazebo')
    
    # Launch configuration variables
    robot_name = LaunchConfiguration('robot_name')
    world_file = LaunchConfiguration('world_file')
    use_rviz = LaunchConfiguration('use_rviz')
    show_topics = LaunchConfiguration('show_topics')
    
    # Declare launch arguments
    declare_robot_name_cmd = DeclareLaunchArgument(
        name='robot_name',
        default_value='mycobot_280',
        description='Name of the robot')
    
    declare_world_file_cmd = DeclareLaunchArgument(
        name='world_file',
        default_value='empty.world',
        description='World file for Gazebo simulation')
    
    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='false',  # Disable RViz by default for faster startup
        description='Whether to launch RViz')
    
    declare_show_topics_cmd = DeclareLaunchArgument(
        name='show_topics',
        default_value='true',
        description='Whether to show topic monitoring in separate terminals')
    
    # Include main Gazebo launch WITHOUT contact sensors (they don't work properly)
    gazebo_launch_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share_gazebo, 'launch', 'mycobot.gazebo.launch.py')
        ]),
        launch_arguments={
            'robot_name': robot_name,
            'world_file': world_file,
            'use_rviz': use_rviz,
            'use_contact_sensors': 'false',  # Disable broken Gazebo contact sensors
            'use_sim_time': 'true',
            'load_controllers': 'true'
        }.items()
    )
    
    # WORKING Simple collision detector
    simple_detector_cmd = TimerAction(
        period=5.0,  # Wait 5 seconds for Gazebo to initialize
        actions=[
            Node(
                package='mycobot_description',
                executable='simple_collision_detector.py',
                name='simple_collision_detector',
                output='screen',
                parameters=[{
                    'robot_name': robot_name,
                    'use_sim_time': True
                }]
            )
        ]
    )
    
    # Movement test node (delayed start to allow everything to initialize)
    movement_test_cmd = TimerAction(
        period=8.0,  # Wait 8 seconds for Gazebo and detector to fully initialize
        actions=[
            Node(
                package='mycobot_description',
                executable='test_collision_movements.py',
                name='collision_movement_tester',
                output='screen',
                parameters=[{
                    'robot_name': robot_name,
                    'use_sim_time': True
                }]
            )
        ]
    )
    
    # Topic echo for collision alerts (in separate terminal if requested)
    topic_echo_alerts_cmd = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=['gnome-terminal', '--', 'bash', '-c', 
                     'source /home/migue/mycobot_ws/install/setup.bash && '
                     'echo "🚨 Monitoring Collision Alerts (WORKING Simple Detection)..." && '
                     'echo "This detector actually works and detects dangerous joint positions!" && '
                     'echo "=========================================================" && '
                     'ros2 topic echo /mycobot_280/self_collision_alert; '
                     'read -p "Press Enter to close..."'],
                output='screen',
                condition=IfCondition(show_topics)
            )
        ]
    )
    
    # Topic echo for collision details (in separate terminal if requested)
    topic_echo_details_cmd = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=['gnome-terminal', '--', 'bash', '-c', 
                     'source /home/migue/mycobot_ws/install/setup.bash && '
                     'echo "📋 Monitoring Collision Details (JSON Format for PPO)..." && '
                     'echo "Use this data for negative rewards in PPO training!" && '
                     'echo "=========================================================" && '
                     'ros2 topic echo /mycobot_280/collision_details; '
                     'read -p "Press Enter to close..."'],
                output='screen',
                condition=IfCondition(show_topics)
            )
        ]
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_robot_name_cmd)
    ld.add_action(declare_world_file_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_show_topics_cmd)
    
    # Add main actions
    ld.add_action(gazebo_launch_cmd)
    ld.add_action(simple_detector_cmd)
    ld.add_action(movement_test_cmd)
    ld.add_action(topic_echo_alerts_cmd)
    ld.add_action(topic_echo_details_cmd)
    
    return ld