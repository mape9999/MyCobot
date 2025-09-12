#!/usr/bin/env python3
"""
Control robot arm to move to a specific cartesian position.

This script creates a ROS 2 node that moves a robot arm to a specified
cartesian position (x,y,z) using MoveIt's motion planning capabilities.
After reaching the target position, the robot will wait for a specified time
and then return to the home position.

Usage:
    python3 cartesian_move.py --position 0.3 0.2 0.5 --orientation 0.0 0.0 0.0 --close-gripper

:author: Your Name
:date: Current Date
"""

import time
import argparse
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import tf_transformations
import numpy as np


class CartesianMoveController(Node):
    """
    A ROS 2 node for controlling robot arm movements to cartesian positions.

    This class moves the arm to a specified cartesian position using MoveIt's
    inverse kinematics capabilities, waits for a specified time, and then
    returns to the home position.
    """

    def __init__(self, position, orientation, close_gripper, open_gripper, 
                 time_for_movement, monitor_joints, wait_time):
        """
        Initialize the node and set up clients for arm control.

        Args:
            position (list): Target position [x, y, z] in meters
            orientation (list): Target orientation as [roll, pitch, yaw] in radians
            close_gripper (bool): Whether to close the gripper at the target position
            open_gripper (bool): Whether to open the gripper at the target position
            time_for_movement (float): Time allowed for arm movement in seconds
            monitor_joints (bool): Whether to monitor joint positions
            wait_time (float): Time to wait at target position before returning home
        """
        super().__init__('cartesian_move_controller')

        # Store parameters
        self.position = position
        self.orientation = orientation
        self.home_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Home position is always all zeros
        self.close_gripper = close_gripper
        self.open_gripper = open_gripper
        self.time_for_movement = time_for_movement
        self.monitor_joints = monitor_joints
        self.wait_time = wait_time
        self.current_joint_positions = None
        self.trajectory_completed = False
        self.shutdown_requested = False
        self.execution_phase = "start"  # Tracks the current phase of execution

        # Set up arm trajectory action client
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        # Set up gripper action client
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_action_controller/gripper_cmd'
        )

        # Set up IK service client
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        # Set up TF listener for frame transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Wait for services and action servers to be available
        self.get_logger().info('Waiting for action servers and services...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')
        self.get_logger().info('All services and action servers connected!')

        # List of joint names for the robot arm
        self.joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'
        ]

        # Subscribe to joint states if monitoring is enabled
        if self.monitor_joints:
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_state_callback,
                10)
            self.get_logger().info('Subscribed to joint states')
            # Create a timer to periodically check and print joint positions
            self.monitor_timer = self.create_timer(1.0, self.monitor_joint_positions)

        # Create timer to start the trajectory execution
        self.create_timer(0.1, self.execute_trajectory)

    def joint_state_callback(self, msg):
        """
        Callback for joint state messages.

        Args:
            msg (JointState): The joint state message
        """
        # Extract the arm joint positions (filtering out gripper joints)
        arm_positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                arm_positions[name] = msg.position[i]
        
        # Store the positions in order
        if len(arm_positions) == len(self.joint_names):
            self.current_joint_positions = [arm_positions[name] for name in self.joint_names]

    def monitor_joint_positions(self):
        """
        Periodically print the current joint positions.
        """
        if self.current_joint_positions:
            self.get_logger().info(f'Current joint positions ({self.execution_phase}):')
            for i, name in enumerate(self.joint_names):
                self.get_logger().info(f'  {name}: {self.current_joint_positions[i]:.4f}')

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion.

        Args:
            roll (float): Roll angle in radians
            pitch (float): Pitch angle in radians
            yaw (float): Yaw angle in radians

        Returns:
            list: Quaternion as [x, y, z, w]
        """
        return tf_transformations.quaternion_from_euler(roll, pitch, yaw)

    def compute_inverse_kinematics(self, position, orientation):
        """
        Compute inverse kinematics for a given cartesian position and orientation.

        Args:
            position (list): Target position [x, y, z] in meters
            orientation (list): Target orientation as [roll, pitch, yaw] in radians

        Returns:
            list: Joint angles for the target position, or None if IK fails
        """
        # Create the IK request
        request = GetPositionIK.Request()
        request.ik_request = PositionIKRequest()
        request.ik_request.group_name = "arm_group"  # The planning group name from MoveIt config
        request.ik_request.robot_state = RobotState()
        request.ik_request.avoid_collisions = True
        
        # Set the target pose
        request.ik_request.pose_stamped.header.frame_id = "base_link"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position = Point(x=position[0], y=position[1], z=position[2])
        
        # Convert Euler angles to quaternion
        quat = self.euler_to_quaternion(orientation[0], orientation[1], orientation[2])
        request.ik_request.pose_stamped.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        
        # Call the IK service
        self.get_logger().info(f'Computing IK for position {position} and orientation {orientation}')
        future = self.ik_client.call_async(request)
        
        # Wait for the result
        rclpy.spin_until_future_complete(self, future)
        
        # Check if IK succeeded
        response = future.result()
        if response.error_code.val != 1:  # 1 means SUCCESS
            self.get_logger().error(f'IK failed with error code {response.error_code.val}')
            return None
        
        # Extract joint positions from the solution
        joint_positions = []
        for joint_name in self.joint_names:
            idx = response.solution.joint_state.name.index(joint_name)
            joint_positions.append(response.solution.joint_state.position[idx])
        
        self.get_logger().info(f'IK solution found: {joint_positions}')
        return joint_positions

    def send_arm_command(self, positions, movement_time=2.0):
        """
        Send a command to move the robot arm to specified joint positions.

        Args:
            positions (list): List of 6 joint angles in radians
            movement_time (float): Time allowed for the movement in seconds
        """
        # Create a trajectory point with the target positions
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(movement_time), 
                                         nanosec=int((movement_time % 1) * 1e9))

        # Create and send the goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = [point]

        self.arm_client.send_goal_async(goal_msg)
        self.get_logger().info(f'Sending arm command: {positions}')

    def send_gripper_command(self, position):
        """
        Send a command to the gripper to open or close.

        Args:
            position (float): Position value for gripper (0.0 for open, -0.7 for closed)
        """
        # Create and send the gripper command
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 5.0

        self.gripper_client.send_goal_async(goal_msg)
        
        if position == 0.0:
            self.get_logger().info('Opening gripper')
        else:
            self.get_logger().info('Closing gripper')

    def execute_trajectory(self):
        """
        Execute the trajectory sequence:
        1. Move to home position (all joints at 0.0)
        2. Compute IK for target cartesian position
        3. Move to target position
        4. Perform gripper actions as requested
        5. Wait for specified time
        6. Return to home position
        """
        # Cancel the timer to ensure this runs only once
        if hasattr(self, 'timer'):
            self.timer.cancel()
        
        # First, ensure we're at the home position
        self.execution_phase = "moving_to_home"
        self.get_logger().info('Moving to home position')
        self.send_arm_command(self.home_pos, self.time_for_movement)
        time.sleep(self.time_for_movement + 0.5)  # Wait for movement + a little extra
        
        # Compute inverse kinematics for the target position
        self.execution_phase = "computing_ik"
        target_joints = self.compute_inverse_kinematics(self.position, self.orientation)
        
        if target_joints is None:
            self.get_logger().error('Failed to compute inverse kinematics. Cannot reach target position.')
            self.shutdown_requested = True
            return
        
        # Move to target position
        self.execution_phase = "moving_to_target"
        self.get_logger().info(f'Moving to target cartesian position {self.position}')
        self.send_arm_command(target_joints, self.time_for_movement)
        time.sleep(self.time_for_movement + 0.5)  # Wait for movement + a little extra

        # Perform gripper actions
        if self.close_gripper:
            self.send_gripper_command(-0.7)  # Close gripper
            time.sleep(0.5)  # Wait for gripper to close
        
        if self.open_gripper:
            self.send_gripper_command(0.0)  # Open gripper
            time.sleep(0.5)  # Wait for gripper to open

        # Wait at the target position
        self.execution_phase = "at_target"
        self.get_logger().info(f'Waiting at target position for {self.wait_time} seconds')
        time.sleep(self.wait_time)
        
        # Return to home position
        self.execution_phase = "returning_home"
        self.get_logger().info('Returning to home position')
        self.send_arm_command(self.home_pos, self.time_for_movement)
        time.sleep(self.time_for_movement + 0.5)  # Wait for movement + a little extra

        # Open gripper at home position if it was closed
        if self.close_gripper and not self.open_gripper:
            self.send_gripper_command(0.0)  # Open gripper
            time.sleep(0.5)  # Wait for gripper to open
            
        self.get_logger().info('Trajectory sequence complete')
        self.execution_phase = "complete"
        self.trajectory_completed = True
        
        # If monitoring joints, wait a bit longer to see final positions
        if self.monitor_joints:
            self.get_logger().info('Monitoring final joint positions...')
            time.sleep(2.0)  # Wait to see final positions
        
        # Signal to main that we're ready to shutdown
        self.shutdown_requested = True


def main(args=None):
    """
    Initialize and run the cartesian move control node.

    Args:
        args: Command-line arguments (default: None)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Control robot arm to move to cartesian position')
    
    parser.add_argument('--position', nargs=3, type=float, required=True,
                        help='Target position [x, y, z] in meters')
    
    parser.add_argument('--orientation', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                        help='Target orientation [roll, pitch, yaw] in radians (default: [0.0, 0.0, 0.0])')
    
    parser.add_argument('--close-gripper', action='store_true',
                        help='Close the gripper at target position')
    
    parser.add_argument('--open-gripper', action='store_true',
                        help='Open the gripper at target position')
    
    parser.add_argument('--time', type=float, default=2.0,
                        help='Time allowed for movement in seconds (default: 2.0)')
    
    parser.add_argument('--monitor', action='store_true',
                        help='Monitor joint positions during execution')
    
    parser.add_argument('--wait', type=float, default=5.0,
                        help='Time to wait at target position in seconds (default: 5.0)')
    
    # Parse command line arguments
    parsed_args, remaining_args = parser.parse_known_args(args=args)
    
    # Initialize ROS
    rclpy.init(args=remaining_args)
    
    # Create and run the controller
    controller = CartesianMoveController(
        position=parsed_args.position,
        orientation=parsed_args.orientation,
        close_gripper=parsed_args.close_gripper,
        open_gripper=parsed_args.open_gripper,
        time_for_movement=parsed_args.time,
        monitor_joints=parsed_args.monitor,
        wait_time=parsed_args.wait
    )

    try:
        # Use a custom spin that checks for shutdown requests
        max_duration = 60.0  # Maximum duration in seconds to run the node
        start_time = time.time()
        
        while time.time() - start_time < max_duration:
            rclpy.spin_once(controller, timeout_sec=0.1)
            
            if controller.shutdown_requested:
                break
            
            time.sleep(0.1)  # Small sleep to reduce CPU usage
            
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down cartesian move controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 