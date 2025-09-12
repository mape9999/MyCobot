#!/usr/bin/env python3
"""
Train a myCobot robot to reach specific cartesian coordinates using Reinforcement Learning.

This script implements a PPO (Proximal Policy Optimization) agent that learns to control
a myCobot robot to reach target positions in cartesian space. The agent learns through
trial and error, receiving rewards based on how close it gets to the target position.

Usage:
    python3 rl.py --target 0.3 0.2 0.5 --episodes 1000 --render

:author: Your Name
:date: Current Date
"""

import os
import time
import math
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState


# Define the Actor and Critic networks for PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = Normal(action_mean, torch.sqrt(self.action_var))
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        dist = Normal(action_mean, torch.sqrt(action_var))
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


# PPO Agent class
class PPO:
    def __init__(self, state_dim, action_dim, action_std_init=0.6, lr=0.0003, gamma=0.99, 
                 K_epochs=80, eps_clip=0.2, action_std_decay_rate=0.05, min_action_std=0.1):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.action_std = action_std_init
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def decay_action_std(self, decay_rate=None):
        if decay_rate is None:
            decay_rate = self.action_std_decay_rate
            
        self.action_std = max(self.min_action_std, self.action_std - decay_rate)
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)
        
        print(f"Action std decayed to {self.action_std}")
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        
        return action.numpy(), action_logprob
    
    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


# Memory class for storing trajectories
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


# Environment for the myCobot robot
class MyCobotEnv(gym.Env):
    def __init__(self, target_position, node, time_scale=0.5):
        super(MyCobotEnv, self).__init__()
        
        # ROS2 node
        self.node = node
        
        # Target position
        self.target_position = np.array(target_position, dtype=np.float32)
        
        # Robot parameters
        self.joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'
        ]
        
        self.num_joints = len(self.joint_names)
        self.joint_positions = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self.end_effector_position = np.zeros(3, dtype=np.float32)
        
        # Robot link lengths (approximate values in meters)
        self.link_lengths = [
            0.132,  # Base to joint 1
            0.110,  # Joint 1 to joint 2
            0.096,  # Joint 2 to joint 3
            0.060,  # Joint 3 to joint 4
            0.065   # Joint 4 to end effector
        ]
        
        # Joint limits
        self.joint_limits = [
            (-math.pi, math.pi),      # Joint 1 limits
            (-math.pi/2, math.pi/2),  # Joint 2 limits
            (-math.pi, math.pi),      # Joint 3 limits
            (-math.pi, math.pi),      # Joint 4 limits
            (-math.pi/2, math.pi/2),  # Joint 5 limits
            (-math.pi, math.pi)       # Joint 6 limits
        ]
        
        # Time scale for actions (to slow down movements)
        self.time_scale = time_scale
        
        # Set up action and observation spaces
        # Action space: joint velocities (normalized between -1 and 1)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_joints,), 
            dtype=np.float32
        )
        
        # Observation space: current joint positions, joint velocities, end effector position, target position
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_joints + self.num_joints + 3 + 3,), 
            dtype=np.float32
        )
        
        # Set up ROS2 action client
        self.arm_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # Wait for action server
        self.node.get_logger().info('Waiting for action server...')
        self.arm_client.wait_for_server()
        self.node.get_logger().info('Action server connected!')
        
        # Subscribe to joint states
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10)
        
        # Wait for first joint state message
        self.received_joint_state = False
        while not self.received_joint_state:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.1)
        
        # Initialize state
        self.reset()
    
    def _joint_state_callback(self, msg):
        """
        Callback for joint state messages.
        """
        # Extract the arm joint positions
        arm_positions = {}
        arm_velocities = {}
        
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                arm_positions[name] = msg.position[i]
                if len(msg.velocity) > i:
                    arm_velocities[name] = msg.velocity[i]
                else:
                    arm_velocities[name] = 0.0
        
        # Store the positions and velocities in order
        if len(arm_positions) == len(self.joint_names):
            self.joint_positions = np.array([arm_positions[name] for name in self.joint_names], dtype=np.float32)
            self.joint_velocities = np.array([arm_velocities[name] for name in self.joint_names], dtype=np.float32)
            
            # Calculate end effector position
            self._update_end_effector_position()
            
            self.received_joint_state = True
    
    def _update_end_effector_position(self):
        """
        Calculate the current end effector position using forward kinematics.
        """
        try:
            # Get current joint angles
            joint1, joint2, joint3, joint4, joint5, joint6 = self.joint_positions
            
            # Calculate forward kinematics (simplified model)
            # Calculate the position after joint 1 (base rotation)
            x0 = 0
            y0 = 0
            z0 = self.link_lengths[0]  # Base height
            
            # Apply joint 1 rotation (around Z axis)
            c1 = math.cos(joint1)
            s1 = math.sin(joint1)
            
            # Calculate position after joint 2
            c2 = math.cos(joint2)
            s2 = math.sin(joint2)
            
            # Length of upper arm
            l1 = self.link_lengths[1]
            
            # Position after joint 2
            x1 = l1 * s2 * c1
            y1 = l1 * s2 * s1
            z1 = z0 + l1 * c2
            
            # Calculate position after joint 3
            c3 = math.cos(joint3)
            s3 = math.sin(joint3)
            
            # Length of forearm
            l2 = self.link_lengths[2]
            
            # Position after joint 3
            x2 = x1 + l2 * s3 * c1
            y2 = y1 + l2 * s3 * s1
            z2 = z1 + l2 * c3
            
            # Calculate position after joint 4
            c4 = math.cos(joint4)
            s4 = math.sin(joint4)
            
            # Length to wrist
            l3 = self.link_lengths[3]
            
            # Position after joint 4
            x3 = x2 + l3 * s4 * c1
            y3 = y2 + l3 * s4 * s1
            z3 = z2 + l3 * c4
            
            # Calculate final end effector position
            l4 = self.link_lengths[4]
            
            # Final position
            x_final = x3 + l4 * s4 * c1
            y_final = y3 + l4 * s4 * s1
            z_final = z3 + l4 * c4
            
            self.end_effector_position = np.array([x_final, y_final, z_final], dtype=np.float32)
            
        except Exception as e:
            self.node.get_logger().error(f'Error calculating forward kinematics: {str(e)}')
    
    def _get_observation(self):
        """
        Get the current observation (state).
        """
        obs = np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.end_effector_position,
            self.target_position
        ])

        # Check for and handle NaN values
        if np.isnan(obs).any():
            self.node.get_logger().error("NaN detected in observation vector!")
            self.node.get_logger().error(f"  - Joint Positions: {self.joint_positions}")
            self.node.get_logger().error(f"  - Joint Velocities: {self.joint_velocities}")
            self.node.get_logger().error(f"  - End Effector Position: {self.end_effector_position}")
            # Replace NaNs with zeros to prevent crash
            obs = np.nan_to_num(obs)
            self.node.get_logger().warn("Replaced NaN values with 0 to prevent a crash.")

        return obs
    
    def _get_reward(self):
        """
        Calculate reward based on distance to target.
        """
        # Calculate distance to target
        distance = np.linalg.norm(self.end_effector_position - self.target_position)
        
        # Base reward is negative distance (closer is better)
        reward = -distance
        
        # Bonus reward for being very close to target
        if distance < 0.05:  # Within 5cm
            reward += 10.0
            
        # Penalty for joint limits
        joint_limit_penalty = 0
        for i, (pos, (lower, upper)) in enumerate(zip(self.joint_positions, self.joint_limits)):
            if pos <= lower or pos >= upper:
                joint_limit_penalty -= 1.0
        
        reward += joint_limit_penalty
        
        return reward
    
    def _is_done(self):
        """
        Check if the episode is done.
        """
        # Episode is done if we're close enough to target
        distance = np.linalg.norm(self.end_effector_position - self.target_position)
        if distance < 0.03:  # Within 3cm
            return True
            
        # Or if we've hit joint limits severely
        for i, (pos, (lower, upper)) in enumerate(zip(self.joint_positions, self.joint_limits)):
            if pos < lower - 0.1 or pos > upper + 0.1:  # Some tolerance
                return True
                
        return False
    
    def _send_joint_command(self, positions, duration=1.0):
        """
        Send a command to move the robot arm to specified joint positions.
        """
        # Create a trajectory point with the target positions
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.time_from_start = Duration(sec=int(duration), 
                                         nanosec=int((duration % 1) * 1e9))

        # Create and send the goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = [point]

        # Send the goal and wait for it to complete
        self.arm_client.send_goal_async(goal_msg)
        
        # Wait for the movement to complete
        time.sleep(duration)
        
        # Update joint positions and end effector position
        rclpy.spin_once(self.node, timeout_sec=0.1)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        # Fix: Use the correct super().reset() call for gymnasium
        if seed is not None:
            super().reset(seed=seed)
        
        # Move to home position (all joints at 0)
        home_position = np.zeros(self.num_joints, dtype=np.float32)
        self._send_joint_command(home_position, duration=2.0)
        
        # Update state
        self.joint_positions = np.copy(home_position)
        self.joint_velocities = np.zeros_like(self.joint_positions)
        self._update_end_effector_position()
        
        # Return observation
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute action and return new state, reward, done, and info.
        """
        # Scale action to joint velocities (actions are between -1 and 1)
        # Convert to position increments
        position_increments = action * 0.1  # Scale factor to limit movement
        
        # Calculate new joint positions
        new_positions = self.joint_positions + position_increments
        
        # Clip to joint limits
        for i, (lower, upper) in enumerate(self.joint_limits):
            new_positions[i] = max(min(new_positions[i], upper), lower)
        
        # Send command to robot
        self._send_joint_command(new_positions, duration=self.time_scale)
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if done
        done = self._is_done()
        
        # Get new observation
        observation = self._get_observation()
        
        # Return step information
        return observation, reward, done, False, {}
    
    def render(self):
        """
        Render the current state (print information).
        """
        self.node.get_logger().info(f"Current position: {self.end_effector_position}")
        self.node.get_logger().info(f"Target position: {self.target_position}")
        self.node.get_logger().info(f"Distance: {np.linalg.norm(self.end_effector_position - self.target_position):.4f}")
        self.node.get_logger().info(f"Joint positions: {self.joint_positions}")


# Main training function
def train(target_position, episodes=1000, render=False, save_path="ppo_model.pth"):
    """
    Train the PPO agent to reach the target position.
    
    Args:
        target_position (list): Target position [x, y, z] in meters
        episodes (int): Number of training episodes
        render (bool): Whether to render the environment
        save_path (str): Path to save the trained model
    """
    # Initialize ROS2 node
    rclpy.init()
    node = rclpy.create_node('rl_training_node')
    
    try:
        # Create environment
        env = MyCobotEnv(target_position, node)
        
        # Set up PPO agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = PPO(state_dim, action_dim, action_std_init=0.6, lr=0.0003, gamma=0.99)
        memory = RolloutBuffer()
        
        # Training loop
        time_step = 0
        i_episode = 0
        
        # Action standard deviation decay
        action_std_decay_freq = int(episodes / 10)
        
        # Log directory
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(log_file, "w") as f:
            f.write("Episode,Reward,Steps,Distance\n")
        
        # Estimate total training time
        avg_steps_per_episode = 50  # Estimated average steps per episode
        avg_time_per_step = 0.5  # Estimated time per step in seconds (depends on time_scale)
        estimated_training_time = episodes * avg_steps_per_episode * avg_time_per_step
        estimated_hours = estimated_training_time / 3600
        
        node.get_logger().info(f"Starting training for {episodes} episodes")
        node.get_logger().info(f"Estimated training time: {estimated_hours:.1f} hours")
        node.get_logger().info(f"Target position: {target_position}")
        
        while i_episode < episodes:
            state, _ = env.reset()
            current_ep_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Select action
                action, action_logprob = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action)
                
                # Store in memory
                memory.states.append(torch.FloatTensor(state))
                memory.actions.append(torch.FloatTensor(action))
                memory.logprobs.append(action_logprob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # Update state
                state = next_state
                current_ep_reward += reward
                time_step += 1
                steps += 1
                
                # Render if requested
                if render:
                    env.render()
                
                # Limit steps per episode to avoid very long episodes
                if steps >= 100:  # Maximum 100 steps per episode
                    break
            
            # Update PPO agent
            agent.update(memory)
            memory.clear()
            
            # Decay action standard deviation
            if (i_episode + 1) % action_std_decay_freq == 0:
                agent.decay_action_std()
            
            # Log progress
            distance = np.linalg.norm(env.end_effector_position - env.target_position)
            node.get_logger().info(f"Episode: {i_episode+1}/{episodes}, Reward: {current_ep_reward:.2f}, Steps: {steps}, Distance: {distance:.4f}")
            
            with open(log_file, "a") as f:
                f.write(f"{i_episode+1},{current_ep_reward:.2f},{steps},{distance:.4f}\n")
            
            # Save model periodically
            if (i_episode + 1) % 100 == 0:
                torch.save(agent.policy.state_dict(), save_path)
                node.get_logger().info(f"Model saved to {save_path}")
            
            i_episode += 1
        
        # Save final model
        torch.save(agent.policy.state_dict(), save_path)
        node.get_logger().info(f"Training complete. Final model saved to {save_path}")
        
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


# Function to test a trained model
def test(target_position, model_path="ppo_model.pth", render=True):
    """
    Test a trained PPO agent.
    
    Args:
        target_position (list): Target position [x, y, z] in meters
        model_path (str): Path to the trained model
        render (bool): Whether to render the environment
    """
    # Initialize ROS2 node
    rclpy.init()
    node = rclpy.create_node('rl_testing_node')
    
    try:
        # Create environment
        env = MyCobotEnv(target_position, node)
        
        # Set up PPO agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = PPO(state_dim, action_dim)
        
        # Load trained model
        agent.policy.load_state_dict(torch.load(model_path))
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        
        node.get_logger().info(f"Model loaded from {model_path}")
        
        # Test loop
        for i_episode in range(5):  # Run 5 test episodes
            state, _ = env.reset()
            current_ep_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:  # Limit to 100 steps
                # Select action
                action, _ = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, _, _ = env.step(action)
                
                # Update state
                state = next_state
                current_ep_reward += reward
                steps += 1
                
                # Render if requested
                if render:
                    env.render()
                    time.sleep(0.1)  # Slow down for visualization
            
            # Log results
            distance = np.linalg.norm(env.end_effector_position - env.target_position)
            node.get_logger().info(f"Test Episode: {i_episode+1}, Reward: {current_ep_reward:.2f}, Steps: {steps}, Distance: {distance:.4f}")
            
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    """
    Main function to parse arguments and start training or testing.
    """
    parser = argparse.ArgumentParser(description='Train or test a RL agent for myCobot control')
    
    parser.add_argument('--target', nargs=3, type=float, default=[0.3, 0.2, 0.5],
                        help='Target position [x, y, z] in meters')
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training')
    
    parser.add_argument('--model', type=str, default="ppo_model.pth",
                        help='Path to save/load the model')
    
    parser.add_argument('--test', action='store_true',
                        help='Test mode (use trained model)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run training or testing
    if args.test:
        test(args.target, model_path=args.model, render=args.render)
    else:
        train(args.target, episodes=args.episodes, render=args.render, save_path=args.model)


if __name__ == "__main__":
    main() 