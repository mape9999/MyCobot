#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tf2_ros

import time
from gazebo_marker_manager import GazeboMarkerManager

class MyCobotEnvImproved3D(gym.Env):
    """Entorno Gymnasium MEJORADO 3D para MyCobot (controlando 3 articulaciones)
    Controla: link1_to_link2, link2_to_link3, link4_to_link5
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, target_position=(0.06062, 0.0, 0.3), max_episode_steps=200, success_threshold=0.05, show_markers=True, random_start=False):
        super().__init__()

        # Inicializar ROS2 si no está ya inicializado
        if not rclpy.ok():
            rclpy.init()

        # Crear nodo ROS2
        self.node = rclpy.create_node('mycobot_env_improved_3d')

        # Parámetros del robot
        self.all_joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'
        ]
        # Ahora controlamos 3 DOF
        self.controlled_indices = [0, 1, 3]  # link1_to_link2, link2_to_link3, link4_to_link5
        self.controlled_joints = [self.all_joint_names[i] for i in self.controlled_indices]

        # Límites de articulaciones
        self.joint_min = -2.879793
        self.joint_max = 2.879793
        self.max_delta = 0.3

        # Parámetros del entorno
        self.target_position = np.array(target_position, dtype=np.float32)
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.show_markers = show_markers
        self.random_start = random_start

        # Rangos de inicio aleatorio
        if self.random_start:
            self.start_joint_range = [
                (-1.5, 1.5),  # link1_to_link2
                (-1.5, 1.5),  # link2_to_link3
                (-1.5, 1.5)   # link4_to_link5
            ]
            print("🎲 Modo inicio aleatorio 3D: ACTIVADO")
            print(f"   🤖 Rangos articulaciones: {self.start_joint_range}")
        else:
            print("📍 Modo inicio fijo 3D: [0.0, 0.0, 0.0] articulaciones")

        # Tiempos de espera 
        self.movement_wait_cycles = 15 #25
        self.reset_wait_cycles = 30 #40
        self.ros_spin_cycles = 3 #5

        # Estado del entorno
        self.current_step = 0
        self.current_joint_positions = np.zeros(6)
        self.current_ee_position = np.zeros(3)

        # Historial
        self.distance_history = []
        self.best_distance = float('inf')
        self.consecutive_improvements = 0
        self.stuck_counter = 0

        # Detección de precisión
        self.min_distance_reached = float('inf')
        self.consecutive_away_steps = 0
        self.precision_mode_threshold = 0.05
        self.away_step_limit = 3
        self.alejamiento_percentage = 1.05

        # TF2 para obtener posición del EE
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # Publisher y subscribers
        self.trajectory_pub = self.node.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        time.sleep(0.5)

        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Gestor de marcadores
        if self.show_markers:
            self.marker_manager = GazeboMarkerManager()
            print("🔵 Marcadores visuales: ACTIVADOS")
        else:
            self.marker_manager = None
            print("🚫 Marcadores visuales: DESACTIVADOS")

        # Espacios de acción y observación
        self.action_space = spaces.Box(
            low=-self.max_delta,
            high=self.max_delta,
            shape=(3,),  # 3 articulaciones controladas
            dtype=np.float32
        )

        # Observación: [ee_pos(3), target_pos(3), joint_pos(3), distance(1), prev_distance(1)] = 11
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        print("🤖 MyCobotEnv MEJORADO 3D inicializado (sin colisiones)")
        print(f"🎯 Objetivo: {self.target_position}")
        print(f"📏 Success threshold: {self.success_threshold}m")
        print(f"⚙️ Articulaciones controladas: {self.controlled_joints}")
        print(f"🎛️ Max delta: {self.max_delta}")

    def joint_state_callback(self, msg):
        try:
            joint_positions = {}
            for i, name in enumerate(msg.name):
                if name in self.all_joint_names:
                    joint_positions[name] = msg.position[i]
            for i, joint_name in enumerate(self.all_joint_names):
                if joint_name in joint_positions:
                    self.current_joint_positions[i] = joint_positions[joint_name]
        except Exception as e:
            self.node.get_logger().warn(f"Error en joint_state_callback: {e}")

    def get_ee_position(self):
        try:
            #frames_to_try = ['gripper_base', 'link6_flange', 'link6']
            frames_to_try = ['EE_link']
            for frame in frames_to_try:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link', frame, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    x = transform.transform.translation.x
                    y = transform.transform.translation.y
                    z = transform.transform.translation.z
                    if self.current_step % 50 == 0 or self.current_step == 0:
                        print(f"🔍 TF2 - {frame}: [{x:.6f}, {y:.6f}, {z:.6f}]")
                    #if frame == 'gripper_base':
                    if frame == "EE_link":
                        return np.array([x, y, z])
                    elif frame == frames_to_try[0]:
                        return np.array([x, y, z])
                except Exception as frame_error:
                    if self.current_step == 0:
                        print(f"⚠️ Frame '{frame}' no disponible: {frame_error}")
                    continue
            print("⚠️ Ningún frame TF2 disponible, usando cinemática simplificada 3D")
            return self.forward_kinematics_simplified()
        except Exception as e:
            print(f"⚠️ TF2 falló completamente, usando cinemática simplificada 3D: {e}")
            return self.forward_kinematics_simplified()

    def forward_kinematics_simplified(self):
        """Cinemática simplificada considerando q1 (rotación Yaw base) de forma aproximada (fallback)."""
        q1 = self.current_joint_positions[0]  # link1_to_link2 (yaw)
        q2 = self.current_joint_positions[1]  # link2_to_link3
        q4 = self.current_joint_positions[3]  # link4_to_link5

        base_x = 0.060607
        base_y = 0.0797747
        base_z = 0.410075

        L_effective = 0.15
        local_y = base_y + L_effective * np.sin(q2) * np.cos(q4)
        local_z = base_z + L_effective * (np.cos(q2) - 1) + L_effective * np.sin(q4)
        local_x = base_x

        cos1, sin1 = np.cos(q1), np.sin(q1)
        x = local_x * cos1 - local_y * sin1
        y = local_x * sin1 + local_y * cos1
        z = local_z
        return np.array([x, y, z])

    def send_joint_command(self, joint_positions):
        """Envía comando de articulaciones al robot (3 controladas)"""
        try:
            pos0 = float(joint_positions[0])  # link1_to_link2
            pos1 = float(joint_positions[1])  # link2_to_link3
            pos2 = float(joint_positions[2])  # link4_to_link5

            if self.current_step % 20 == 0:
                print(f"📤 Enviando comando 3D: [{pos0:.3f}, {pos1:.3f}, {pos2:.3f}]")

            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp.sec = 0
            trajectory_msg.header.stamp.nanosec = 0
            trajectory_msg.header.frame_id = ''
            trajectory_msg.joint_names = self.all_joint_names

            point = JointTrajectoryPoint()
            full_positions = [pos0, pos1, 0.0, pos2, 0.0, 0.0]
            point.positions = full_positions
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6

            duration = 2.5
            point.time_from_start = Duration(sec=int(duration), nanosec=0)

            trajectory_msg.points = [point]
            self.trajectory_pub.publish(trajectory_msg)
        except Exception as e:
            self.node.get_logger().error(f"Error enviando comando: {e}")
            print(f"❌ Error enviando comando: {e}")

    def calculate_reward_improved(self, current_distance, previous_distance):
        """Función de recompensa."""
        # 1) Éxito tradicional
        if current_distance < self.success_threshold:
            success_reward = 200.0
            print(f"🎉 ¡ÉXITO! Episodio completado en step {self.current_step}")
            print(f"🏆 Reward: +{success_reward} | 📏 Distancia final: {current_distance*1000:.1f}mm")
            return success_reward

        # 2) Éxito de precisión
        precision_success = self.check_precision_success(current_distance)
        if precision_success:
            precision_bonus = max(0, 100 - (self.min_distance_reached * 1000))
            precision_reward = 200.0 + precision_bonus
            print(f"🎯 ¡ÉXITO DE PRECISIÓN! Episodio completado en step {self.current_step}")
            print(f"🏆 Reward: +{precision_reward} | 📏 Mejor distancia: {self.min_distance_reached*1000:.1f}mm | 🎯 Bonus precisión: +{precision_bonus}")
            return precision_reward

        # 3) Recompensa por mejora
        improvement_reward = 0.0
        if previous_distance is not None:
            improvement = previous_distance - current_distance
            improvement_reward = improvement * 100.0
            if improvement > 0:
                self.consecutive_improvements += 1
                if self.consecutive_improvements >= 3:
                    improvement_reward += 10.0
            else:
                self.consecutive_improvements = 0

        # 4) Recompensa por proximidad
        proximity_reward = 0.0
        if current_distance < 0.1:
            proximity_reward = np.exp(-current_distance * 10) * 20.0

        # 5) Penalización por estancamiento
        stagnation_penalty = 0.0
        if len(self.distance_history) >= 5:
            recent_distances = self.distance_history[-5:]
            if max(recent_distances) - min(recent_distances) < 0.005:
                self.stuck_counter += 1
                stagnation_penalty = -self.stuck_counter * 2.0
            else:
                self.stuck_counter = 0

        # 6) Penalización por tiempo
        time_penalty = -0.05

        # 7) Bonus por nuevo mejor resultado
        best_bonus = 0.0
        if current_distance < self.best_distance:
            self.best_distance = current_distance
            best_bonus = 15.0

        total_reward = (improvement_reward + proximity_reward + best_bonus +
                        time_penalty + stagnation_penalty)

        if self.current_step % 15 == 0:
            print(f"📊 Step {self.current_step}: Dist={current_distance*1000:.1f}mm, "
                  f"Mejora={improvement_reward:.2f}, Prox={proximity_reward:.2f}, "
                  f"Total={total_reward:.2f}")
        return total_reward

    def check_precision_success(self, current_distance):
        if self.min_distance_reached >= self.precision_mode_threshold:
            return False
        if self.consecutive_away_steps < self.away_step_limit:
            return False
        if current_distance <= self.min_distance_reached * self.alejamiento_percentage:
            return False
        return True

    def update_precision_tracking(self, current_distance):
        if current_distance < self.min_distance_reached:
            self.min_distance_reached = current_distance
            self.consecutive_away_steps = 0
            if current_distance < self.precision_mode_threshold:
                print(f"🎯 Nueva mejor distancia: {current_distance*1000:.1f}mm (modo precisión activo)")
        elif current_distance > self.min_distance_reached:
            self.consecutive_away_steps += 1
            if self.min_distance_reached < self.precision_mode_threshold:
                print(f"📈 Alejándose: {self.consecutive_away_steps} pasos consecutivos (distancia: {current_distance*1000:.1f}mm)")

    def get_observation_improved(self):
        # Actualizar posición del end effector
        self.current_ee_position = self.get_ee_position()

        # Posiciones de articulaciones controladas
        controlled_positions = self.current_joint_positions[self.controlled_indices]

        # Distancias
        current_distance = np.linalg.norm(self.current_ee_position - self.target_position)
        prev_distance = self.distance_history[-1] if self.distance_history else current_distance

        # Historial
        self.distance_history.append(current_distance)
        if len(self.distance_history) > 10:
            self.distance_history.pop(0)

        observation = np.concatenate([
            self.current_ee_position,      # 3
            self.target_position,          # 3
            controlled_positions,          # 3
            [current_distance],            # 1
            [prev_distance]                # 1
        ]).astype(np.float32)
        return observation

    def step(self, action):
        """Ejecuta una acción en el entorno 3D (sin colisiones)."""
        self.current_step += 1

        # ROS spin
        for _ in range(self.ros_spin_cycles):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        old_distance = np.linalg.norm(self.current_ee_position - self.target_position)

        # Acciones y nuevas posiciones
        current_controlled = self.current_joint_positions[self.controlled_indices]
        action = np.clip(action, -self.max_delta, self.max_delta)
        new_controlled_positions = current_controlled + action

        # Límites
        for i in range(3):
            new_controlled_positions[i] = np.clip(new_controlled_positions[i], self.joint_min, self.joint_max)

        # Enviar comando
        self.send_joint_command(new_controlled_positions)

        # Esperar movimiento
        for _ in range(self.movement_wait_cycles):
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.1)

        # Observación
        observation = self.get_observation_improved()

        # Distancia actual y precisión
        current_distance = np.linalg.norm(self.current_ee_position - self.target_position)
        self.update_precision_tracking(current_distance)

        # Recompensa
        reward = self.calculate_reward_improved(current_distance, old_distance)

        # Terminación
        terminated = False
        truncated = False
        if current_distance < self.success_threshold:
            terminated = True
            success = True
        elif self.check_precision_success(current_distance):
            terminated = True
            success = True
        elif self.current_step >= self.max_episode_steps:
            truncated = True
            success = False
            print(f"⏰ ¡FALLO POR TIEMPO AGOTADO! Máximo de {self.max_episode_steps} steps alcanzado")
            print(f"📏 Distancia final: {current_distance*1000:.1f}mm (objetivo: {self.success_threshold*1000:.1f}mm)")
            print(f"🎯 Mejor distancia alcanzada: {self.min_distance_reached*1000:.1f}mm")
        else:
            success = False

        info = {
            'success': success,
            'distance_to_target': current_distance,
            'step': self.current_step,
            'best_distance': self.best_distance,
            'consecutive_improvements': self.consecutive_improvements,
            'min_distance_reached': self.min_distance_reached,
            'consecutive_away_steps': self.consecutive_away_steps,
            'precision_mode_active': self.min_distance_reached < self.precision_mode_threshold,
            'ee_position': self.current_ee_position.astype(float).tolist(),
            'controlled_joint_positions': self.current_joint_positions[self.controlled_indices].astype(float).tolist(),
            'target_position': self.target_position.astype(float).tolist(),
            'reward': float(reward)
        }
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset de estado
        self.current_step = 0
        self.distance_history = []
        self.best_distance = float('inf')
        self.consecutive_improvements = 0
        self.stuck_counter = 0

        # Precisión
        self.min_distance_reached = float('inf')
        self.consecutive_away_steps = 0

        # Posición inicial
        if self.random_start:
            max_attempts = 10
            for attempt in range(max_attempts):
                j0 = np.random.uniform(*self.start_joint_range[0])
                j1 = np.random.uniform(*self.start_joint_range[1])
                j2 = np.random.uniform(*self.start_joint_range[2])

                print(f"🎲 Posición inicial aleatoria 3D: [{j0:.3f}, {j1:.3f}, {j2:.3f}]")
                self.send_joint_command([j0, j1, j2])
                break
        else:
            print("📍 Posición inicial fija 3D: [0.0, 0.0, 0.0]")
            self.send_joint_command([0.0, 0.0, 0.0])

        extended_wait_cycles = self.reset_wait_cycles * 2
        print(f"⏳ Esperando {extended_wait_cycles * 0.1:.1f}s para posición inicial 3D...")
        for i in range(extended_wait_cycles):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.1)
            if i % 20 == 0 and i > 0:
                current_pos = self.get_ee_position()
                print(f"⏳ Progreso reset ({i}/{extended_wait_cycles}): EE en [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")

        observation = self.get_observation_improved()

        if self.show_markers and self.marker_manager is not None:
            self.marker_manager.update_target_marker(
                self.target_position[0], self.target_position[1], self.target_position[2]
            )

        diff_vector = self.current_ee_position - self.target_position
        initial_distance = np.linalg.norm(diff_vector)
        manual_distance = np.sqrt(diff_vector[0]**2 + diff_vector[1]**2 + diff_vector[2]**2)

        info = {
            'distance_to_target': initial_distance,
            'step': 0
        }

        print(f"🔍 DEBUG - Posición EE: [{self.current_ee_position[0]:.6f}, {self.current_ee_position[1]:.6f}, {self.current_ee_position[2]:.6f}]")
        print(f"🔍 DEBUG - Target: [{self.target_position[0]:.6f}, {self.target_position[1]:.6f}, {self.target_position[2]:.6f}]")
        print(f"🔍 DEBUG - Diferencia: [{diff_vector[0]:.6f}, {diff_vector[1]:.6f}, {diff_vector[2]:.6f}]")
        print(f"🔍 DEBUG - Distancia numpy: {initial_distance:.6f}m ({initial_distance*1000:.1f}mm)")
        print(f"🔍 DEBUG - Distancia manual: {manual_distance:.6f}m ({manual_distance*1000:.1f}mm)")
        print(f"🔍 DEBUG - Articulaciones: {self.current_joint_positions}")
        print(f"🔄 Reset completado (3D). Distancia inicial: {initial_distance*1000:.1f}mm")

        return observation, info

    def close(self):
        try:
            if self.marker_manager is not None:
                self.marker_manager.cleanup()
            self.node.destroy_node()
        except Exception as e:
            print(f"Error cerrando entorno 3D: {e}")