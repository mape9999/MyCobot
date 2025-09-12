#!/usr/bin/env python3
"""
Control robot arm to move to a specific cartesian position using direct calculation.

This script creates a ROS 2 node that moves a robot arm to a specified
cartesian position (x,y,z) using a simple approximation of inverse kinematics.
After reaching the target position, the robot will wait for a specified time
and then return to the home position.

Usage:
    python3 simple_cartesian_move.py --position 0.3 0.2 0.5 --close-gripper

:author: Your Name
:date: Current Date
"""

import time
import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState


class SimpleCartesianMoveController(Node):
    """
    A ROS 2 node for controlling robot arm movements to cartesian positions.

    This class moves the arm to a specified cartesian position using a simple
    approximation of inverse kinematics, waits for a specified time, and then
    returns to the home position.
    """

    def __init__(self, position, close_gripper, open_gripper, 
                 time_for_movement, monitor_joints, wait_time):
        """
        Initialize the node and set up clients for arm control.

        Args:
            position (list): Target position [x, y, z] in meters
            close_gripper (bool): Whether to close the gripper at the target position
            open_gripper (bool): Whether to open the gripper at the target position
            time_for_movement (float): Time allowed for arm movement in seconds
            monitor_joints (bool): Whether to monitor joint positions
            wait_time (float): Time to wait at target position before returning home
        """
        super().__init__('simple_cartesian_move_controller')

        # Store parameters
        self.position = position
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
        self.target_reached = False
        self.current_gripper_position = None

        # Robot parameters for myCobot 280 (approximate values)
        # These are approximate link lengths in meters
        self.link_lengths = [
            0.132,  # Base to joint 1
            0.110,  # Joint 1 to joint 2
            0.096,  # Joint 2 to joint 3
            0.060,  # Joint 3 to joint 4
            0.065   # Joint 4 to end effector
        ]

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

        # Wait for action servers to be available
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Action servers connected!')

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
            # Also track gripper position if available
            elif 'gripper' in name.lower():
                self.current_gripper_position = msg.position[i]
        
        # Store the positions in order
        if len(arm_positions) == len(self.joint_names):
            self.current_joint_positions = [arm_positions[name] for name in self.joint_names]
            
            # Calculate current end effector position
            if self.current_joint_positions:
                self.update_end_effector_position()

    def update_end_effector_position(self):
        """
        Calculate the current end effector position using forward kinematics.
        """
        if not self.current_joint_positions:
            return
            
        try:
            # Get current joint angles
            joint1, joint2, joint3, joint4, joint5, joint6 = self.current_joint_positions
            
            # Calculate forward kinematics (simplified model)
            # This is a simplified calculation and may not be 100% accurate
            
            # Calculate the position after joint 1 (base rotation)
            # At this point, we're still at the base height but rotated
            x0 = 0
            y0 = 0
            z0 = self.link_lengths[0]  # Base height
            
            # Apply joint 1 rotation (around Z axis)
            c1 = math.cos(joint1)
            s1 = math.sin(joint1)
            
            # Calculate position after joint 2
            # Joint 2 rotates around Y axis in the rotated frame
            c2 = math.cos(joint2)
            s2 = math.sin(joint2)
            
            # Length of upper arm
            l1 = self.link_lengths[1]
            
            # Position after joint 2
            x1 = l1 * s2 * c1
            y1 = l1 * s2 * s1
            z1 = z0 + l1 * c2
            
            # Calculate position after joint 3
            # Joint 3 also rotates around Y axis in the rotated frame
            c3 = math.cos(joint3)
            s3 = math.sin(joint3)
            
            # Length of forearm
            l2 = self.link_lengths[2]
            
            # Position after joint 3
            x2 = x1 + l2 * s3 * c1
            y2 = y1 + l2 * s3 * s1
            z2 = z1 + l2 * c3
            
            # Calculate position after joint 4
            # Joint 4 rotates around Y axis in the rotated frame
            c4 = math.cos(joint4)
            s4 = math.sin(joint4)
            
            # Length to wrist
            l3 = self.link_lengths[3]
            
            # Position after joint 4
            x3 = x2 + l3 * s4 * c1
            y3 = y2 + l3 * s4 * s1
            z3 = z2 + l3 * c4
            
            # Calculate final end effector position
            # Joints 5 and 6 mainly affect orientation, not position
            # So we'll just add the final link length in the direction of joint 4
            l4 = self.link_lengths[4]
            
            # Final position
            x_final = x3 + l4 * s4 * c1
            y_final = y3 + l4 * s4 * s1
            z_final = z3 + l4 * c4
            
            # Store the current end effector position
            self.current_end_effector_position = [x_final, y_final, z_final]
            
            # Check if we've reached the target position (within a tolerance)
            if self.execution_phase == "at_target" and not self.target_reached:
                distance = math.sqrt(
                    (x_final - self.position[0])**2 + 
                    (y_final - self.position[1])**2 + 
                    (z_final - self.position[2])**2
                )
                
                if distance < 0.05:  # 5cm tolerance
                    self.target_reached = True
                    self.get_logger().info('Target position reached within tolerance!')
                    
        except Exception as e:
            self.get_logger().error(f'Error calculating forward kinematics: {str(e)}')

    def monitor_joint_positions(self):
        """
        Periodically print the current joint positions and end effector position.
        """
        if self.current_joint_positions:
            self.get_logger().info(f'Current joint positions ({self.execution_phase}):')
            for i, name in enumerate(self.joint_names):
                self.get_logger().info(f'  {name}: {self.current_joint_positions[i]:.4f}')
                
            # Print current end effector position if available
            if hasattr(self, 'current_end_effector_position'):
                x, y, z = self.current_end_effector_position
                self.get_logger().info(f'Current end effector position: [{x:.4f}, {y:.4f}, {z:.4f}]')
                
                # If we have a target position, show the distance to it
                if self.position:
                    distance = math.sqrt(
                        (x - self.position[0])**2 + 
                        (y - self.position[1])**2 + 
                        (z - self.position[2])**2
                    )
                    self.get_logger().info(f'Distance to target: {distance:.4f} meters')
                    
                    # Show error in each axis
                    x_error = self.position[0] - x
                    y_error = self.position[1] - y
                    z_error = self.position[2] - z
                    self.get_logger().info(f'Position error: X={x_error:.4f}, Y={y_error:.4f}, Z={z_error:.4f}')
            
            # Print gripper position if available
            if self.current_gripper_position is not None:
                self.get_logger().info(f'Gripper position: {self.current_gripper_position:.4f}')

    def compute_simple_inverse_kinematics(self, position):
        """
        Compute a simple approximation of inverse kinematics for a given cartesian position.
        
        This is a very simplified approach and may not work for all positions.
        It's designed to give a reasonable approximation for demonstration purposes.

        Args:
            position (list): Target position [x, y, z] in meters

        Returns:
            list: Joint angles for the target position, or None if calculation fails
        """
        try:
            x, y, z = position
            
            # Calculate distance from base to target
            distance = math.sqrt(x**2 + y**2)
            
            # Calculate joint angles
            # Joint 1: Base rotation (around z-axis)
            joint1 = math.atan2(y, x)
            
            # For the remaining joints, we'll use a simplified approach
            # Adjust the height to account for the base height
            adjusted_z = z - self.link_lengths[0]
            
            # Calculate the distance from joint 1 to the target in the x-y plane
            planar_distance = distance
            
            # Calculate the straight-line distance from joint 1 to the target
            total_distance = math.sqrt(planar_distance**2 + adjusted_z**2)
            
            # Calculate the angle from the horizontal to the target
            elevation_angle = math.atan2(adjusted_z, planar_distance)
            
            # Use a simplified 2-joint solution for joints 2 and 3
            # These are the lengths of the two main arm segments
            l1 = self.link_lengths[1] + self.link_lengths[2]  # Upper arm
            l2 = self.link_lengths[3] + self.link_lengths[4]  # Forearm
            
            # Use the law of cosines to find the angle between the two segments
            cos_joint3 = (total_distance**2 - l1**2 - l2**2) / (2 * l1 * l2)
            
            # Ensure the value is within the valid range for arccos
            cos_joint3 = max(min(cos_joint3, 1.0), -1.0)
            
            # Joint 3: Elbow angle
            joint3 = -math.acos(cos_joint3)  # Negative for elbow-down configuration
            
            # Calculate joint 2 using the law of sines
            beta = math.atan2(l2 * math.sin(-joint3), l1 + l2 * math.cos(-joint3))
            joint2 = elevation_angle - beta
            
            # For joints 4, 5, and 6, we'll use simplified values to keep the end effector level
            joint4 = -(joint2 + joint3)  # This keeps the wrist level
            joint5 = math.pi/2  # Keep the end effector pointing downward
            joint6 = 0.0  # No rotation around the end effector axis
            
            # Adjust the joint angles to match the robot's conventions and limits
            joint_angles = [
                joint1,
                joint2,
                joint3,
                joint4,
                0.0,  # Simplified value for joint 5
                0.0   # Simplified value for joint 6
            ]
            
            # Check if any joint is NaN (calculation failed)
            if any(math.isnan(angle) for angle in joint_angles):
                self.get_logger().error('IK calculation produced NaN values')
                return None
            
            # Clip joint angles to reasonable limits
            joint_limits = [
                (-math.pi, math.pi),      # Joint 1 limits
                (-math.pi/2, math.pi/2),  # Joint 2 limits
                (-math.pi, math.pi),      # Joint 3 limits
                (-math.pi, math.pi),      # Joint 4 limits
                (-math.pi/2, math.pi/2),  # Joint 5 limits
                (-math.pi, math.pi)       # Joint 6 limits
            ]
            
            for i in range(len(joint_angles)):
                joint_angles[i] = max(min(joint_angles[i], joint_limits[i][1]), joint_limits[i][0])
            
            self.get_logger().info(f'Calculated joint angles: {[round(angle, 4) for angle in joint_angles]}')
            return joint_angles
            
        except Exception as e:
            self.get_logger().error(f'Error in IK calculation: {str(e)}')
            return None

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
        2. Compute simple IK for target cartesian position
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
        
        # Compute simple inverse kinematics for the target position
        self.execution_phase = "computing_ik"
        target_joints = self.compute_simple_inverse_kinematics(self.position)
        
        if target_joints is None:
            self.get_logger().error('Failed to compute inverse kinematics. Cannot reach target position.')
            self.shutdown_requested = True
            return
        
        # Move to target position
        self.execution_phase = "moving_to_target"
        self.get_logger().info(f'Moving to target cartesian position {self.position}')
        self.send_arm_command(target_joints, self.time_for_movement)
        time.sleep(self.time_for_movement + 0.5)  # Wait for movement + a little extra

        # Reset target reached flag for this new position
        self.target_reached = False
        
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
        
        # Monitor position more frequently while at target
        if self.monitor_joints:
            start_wait = time.time()
            while time.time() - start_wait < self.wait_time:
                # Update end effector position
                if self.current_joint_positions:
                    self.update_end_effector_position()
                    
                    # Print current position every second
                    if hasattr(self, 'current_end_effector_position'):
                        x, y, z = self.current_end_effector_position
                        self.get_logger().info(f'Current end effector position: [{x:.4f}, {y:.4f}, {z:.4f}]')
                        
                        # Calculate distance to target
                        distance = math.sqrt(
                            (x - self.position[0])**2 + 
                            (y - self.position[1])**2 + 
                            (z - self.position[2])**2
                        )
                        self.get_logger().info(f'Distance to target: {distance:.4f} meters')
                
                time.sleep(1.0)  # Check position every second
        else:
            # Just wait the specified time if not monitoring
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
    Initialize and run the simple cartesian move control node.

    Args:
        args: Command-line arguments (default: None)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Control robot arm to move to cartesian position')
    
    parser.add_argument('--position', nargs=3, type=float, required=True,
                        help='Target position [x, y, z] in meters')
    
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
    controller = SimpleCartesianMoveController(
        position=parsed_args.position,
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
        controller.get_logger().info('Shutting down simple cartesian move controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 