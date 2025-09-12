#!/usr/bin/env python3
"""
Control robot arm and gripper to move between user-specified positions.

This script creates a ROS 2 node that moves a robot arm from the home position (all joints at 0.0)
to a target position, waits for 5 seconds, and then returns to the home position.

Action Clients:
    /arm_controller/follow_joint_trajectory (control_msgs/FollowJointTrajectory):
        Commands for controlling arm joint positions
    /gripper_action_controller/gripper_cmd (control_msgs/GripperCommand):
        Commands for opening and closing the gripper

Usage:
    python3 trajectory.py --target 1.0 0.5 0.2 -0.3 0.4 -0.1 --close-gripper

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


class ArmTrajectoryController(Node):
    """
    A ROS 2 node for controlling robot arm movements between specified positions.

    This class moves the arm from the home position to a target position,
    waits for 5 seconds, and then returns to the home position.
    """

    def __init__(self, target_pos, close_gripper, open_gripper, time_for_movement, monitor_joints, wait_time):
        """
        Initialize the node and set up action clients for arm and gripper control.

        Args:
            target_pos (list): Target joint positions for the arm
            close_gripper (bool): Whether to close the gripper at the target position
            open_gripper (bool): Whether to open the gripper at the target position
            time_for_movement (float): Time allowed for arm movement in seconds
            monitor_joints (bool): Whether to monitor joint positions
            wait_time (float): Time to wait at target position before returning home
        """
        super().__init__('arm_trajectory_controller')

        # Store parameters
        self.target_pos = target_pos
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

        # Wait for both action servers to be available
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
            
            # If we're at the target position and in target phase, print a comparison
            if self.execution_phase == "at_target" and self.target_pos and all(abs(c - t) < 0.05 for c, t in zip(self.current_joint_positions, self.target_pos)):
                self.get_logger().info('Robot has reached target position!')
                self.get_logger().info('Comparison of target vs. actual positions:')
                for i, name in enumerate(self.joint_names):
                    self.get_logger().info(f'  {name}: Target={self.target_pos[i]:.4f}, Actual={self.current_joint_positions[i]:.4f}, Diff={abs(self.target_pos[i] - self.current_joint_positions[i]):.4f}')
            
            # If we're at the home position and in home phase, confirm it
            elif self.execution_phase == "returning_home" and all(abs(c) < 0.05 for c in self.current_joint_positions):
                self.get_logger().info('Robot has returned to home position!')

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
        2. Move to target position
        3. Perform gripper actions as requested
        4. Wait for specified time
        5. Return to home position
        """
        # Cancel the timer to ensure this runs only once
        if hasattr(self, 'timer'):
            self.timer.cancel()
        
        # First, ensure we're at the home position
        self.execution_phase = "moving_to_home"
        self.get_logger().info('Moving to home position')
        self.send_arm_command(self.home_pos, self.time_for_movement)
        time.sleep(self.time_for_movement + 0.5)  # Wait for movement + a little extra
        
        # Move to target position
        self.execution_phase = "moving_to_target"
        self.get_logger().info('Moving to target position')
        self.send_arm_command(self.target_pos, self.time_for_movement)
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
    Initialize and run the arm trajectory control node.

    Args:
        args: Command-line arguments (default: None)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Control robot arm trajectory')
    
    parser.add_argument('--target', nargs=6, type=float, required=True,
                        help='Six joint angles for target position')
    
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
    controller = ArmTrajectoryController(
        target_pos=parsed_args.target,
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
        controller.get_logger().info('Shutting down arm trajectory controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 