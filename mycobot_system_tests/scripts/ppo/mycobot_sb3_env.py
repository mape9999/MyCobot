#!/usr/bin/env python3

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import sys


class MyCobotSB3Env(gym.Env):
    """
    Entorno Gym para el MyCobot de 6 DOF compatible con Stable Baselines3.
    Solo controla 2 articulaciones específicas: link2_to_link3 y link4_to_link5.
    Utiliza espacios de estados y acciones continuos para mayor precisión.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, target_position=(0.0, 0.0, 0.191), max_episode_steps=200, fast_mode=False, render_mode=None):
        super(MyCobotSB3Env, self).__init__()
        
        # Parámetros del robot MyCobot (según tu comando ROS2)
        # Todas las articulaciones del robot (necesarias para el JointTrajectory)
        self.all_joint_names = ['link1_to_link2', 'link2_to_link3', 'link3_to_link4', 'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange']

        # Solo controlamos 2 articulaciones específicas
        self.controlled_joints = ['link2_to_link3', 'link4_to_link5']  # Nombres de las articulaciones que controlamos
        self.controlled_indices = [1, 3]  # Índices en el array completo de articulaciones

        # Posiciones fijas para las articulaciones no controladas (valores por defecto seguros)
        self.fixed_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Posiciones iniciales seguras
        
        # Transformaciones REALES extraídas del XACRO del MyCobot 280
        self.joint_transforms = {
            'base_to_link1': [0, 0, 0],           # Origen del robot
            'link1_to_link2': [0, 0, 0.13156],   # 131.56mm hacia arriba
            'link2_to_link3': [0, 0, -0.001],    # Prácticamente en el mismo lugar
            'link3_to_link4': [-0.1104, 0, 0],   # 110.4mm hacia atrás (en X)
            'link4_to_link5': [-0.096, 0, 0.06062],  # 96mm hacia atrás, 60.62mm hacia arriba
            'link5_to_link6': [0, -0.07318, 0],  # 73.18mm hacia la izquierda (en Y)
            'link6_to_flange': [0, 0.0456, 0]    # 45.6mm hacia la derecha (en Y)
        }

        # Límites de articulaciones (del XACRO)
        self.joint_limits_real = {
            'link1_to_link2': [-2.879793, 2.879793],    # ±165°
            'link2_to_link3': [-2.879793, 2.879793],    # ±165°
            'link3_to_link4': [-2.879793, 2.879793],    # ±165°
            'link4_to_link5': [-2.879793, 2.879793],    # ±165°
            'link5_to_link6': [-2.879793, 2.879793],    # ±165°
            'link6_to_link6_flange': [-3.05, 3.05]      # ±175°
        }
        
        # Modo rápido para entrenamiento acelerado
        self.fast_mode = fast_mode
        self.render_mode = render_mode
        
        # Inicializar ROS2 solo si no estamos en modo rápido
        if not self.fast_mode:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = Node('mycobot_sb3_env')
            
            # Publisher para comandos de trayectoria (según tu interfaz ROS2)
            self.trajectory_pub = self.node.create_publisher(
                JointTrajectory,
                '/arm_controller/joint_trajectory',
                10
            )
            
            # Subscriber para el estado de las articulaciones
            self.joint_state_sub = self.node.create_subscription(
                JointState,
                '/joint_states',
                self.joint_state_callback,
                10
            )
            
            # Esperar a que los publishers y subscribers estén listos
            self.has_received_state = False
            self._wait_for_state(timeout=3.0)
        else:
            # En modo rápido, no necesitamos esperar por ROS
            self.has_received_state = True
        
        # Estado actual de las articulaciones (solo las 2 controladas)
        self.current_joint_positions = np.zeros(2)
        self.current_joint_velocities = np.zeros(2)
        self.current_end_effector_position = np.zeros(3)  # 3D position (x, y, z)
        
        # Posición objetivo en coordenadas 3D (x, y, z)
        if isinstance(target_position, (tuple, list)) and len(target_position) == 3:
            self._target_position = (float(target_position[0]), float(target_position[1]), float(target_position[2]))
        else:
            self._target_position = (0.0, 0.0, 0.191)  # Default target (centro del espacio de trabajo real)
        
        # Límites de las articulaciones controladas (del XACRO real)
        self.joint_min = -2.879793  # -165 grados (límite real del MyCobot 280)
        self.joint_max = 2.879793   # +165 grados (límite real del MyCobot 280)
        
        # Espacio de observación: [joint1_pos, joint2_pos, joint1_vel, joint2_vel, target_x, target_y, target_z, ee_x, ee_y, ee_z]
        # Basado en el análisis real del espacio de trabajo: X=[-0.233, 0.227], Y=[-0.233, 0.233], Z=[0.191, 0.191]
        obs_low = np.array([
            self.joint_min,  # joint1_pos (link2_to_link3)
            self.joint_min,  # joint2_pos (link4_to_link5)
            -5.0,           # joint1_vel
            -5.0,           # joint2_vel
            -0.25,          # target_x (espacio de trabajo real + margen)
            -0.25,          # target_y (espacio de trabajo real + margen)
            0.15,           # target_z (espacio de trabajo real + margen)
            -0.25,          # ee_x (espacio de trabajo real + margen)
            -0.25,          # ee_y (espacio de trabajo real + margen)
            0.15            # ee_z (espacio de trabajo real + margen)
        ], dtype=np.float32)

        obs_high = np.array([
            self.joint_max,  # joint1_pos
            self.joint_max,  # joint2_pos
            5.0,            # joint1_vel
            5.0,            # joint2_vel
            0.25,           # target_x (espacio de trabajo real + margen)
            0.25,           # target_y (espacio de trabajo real + margen)
            0.22,           # target_z (espacio de trabajo real + margen)
            0.25,           # ee_x (espacio de trabajo real + margen)
            0.25,           # ee_y (espacio de trabajo real + margen)
            0.22            # ee_z (espacio de trabajo real + margen)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Espacio de acción: [delta_joint1, delta_joint2] - cambios en las posiciones de las articulaciones
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([0.2, 0.2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Control de episodios
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Umbral de éxito (en metros)
        self.success_threshold = 0.05
        
        # Historial de distancias para calcular mejora
        self.previous_distance = None
        
        # Para normalización de recompensas (basado en espacio de trabajo real)
        # Espacio de trabajo: X=[-0.233, 0.227], Y=[-0.233, 0.233], Z=[0.171, 0.211]
        self.max_distance = np.sqrt(0.46**2 + 0.466**2 + 0.04**2)  # Distancia máxima real posible
    
    @property
    def target_position(self):
        """Getter para la posición objetivo"""
        return self._target_position
        
    @target_position.setter
    def target_position(self, value):
        """Setter para la posición objetivo"""
        if isinstance(value, (tuple, list)) and len(value) == 3:
            self._target_position = (float(value[0]), float(value[1]), float(value[2]))
        else:
            raise ValueError("target_position debe ser una tupla o lista de 3 elementos (x, y, z)")
    
    def joint_state_callback(self, msg):
        """Callback para el estado de las articulaciones"""
        # Verificar que el mensaje tiene suficientes datos
        if len(msg.name) < 6 or len(msg.position) < 6:
            print(f"[WARN] Mensaje de estado de articulaciones incompleto: {len(msg.name)} joints, {len(msg.position)} positions")
            return

        try:
            # Actualizar posiciones fijas con el estado actual (para mantener coherencia)
            for i, joint_name in enumerate(self.all_joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.fixed_joint_positions[i] = msg.position[idx]

            # Extraer solo las posiciones de las articulaciones controladas
            joint_positions = []
            joint_velocities = []

            for joint_name in self.controlled_joints:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    joint_positions.append(msg.position[idx])
                    if len(msg.velocity) > idx:
                        joint_velocities.append(msg.velocity[idx])
                    else:
                        joint_velocities.append(0.0)
                else:
                    print(f"[WARN] No se encontró la articulación {joint_name} en el mensaje")
                    return

            # Convertir a numpy arrays
            self.current_joint_positions = np.array(joint_positions, dtype=np.float32)
            self.current_joint_velocities = np.array(joint_velocities, dtype=np.float32)

        except (ValueError, IndexError) as e:
            print(f"[ERROR] Error procesando joint_states: {e}")
            return

        # Calcular posición del efector final
        self.current_end_effector_position = self.forward_kinematics(self.current_joint_positions)

        # Marcar que hemos recibido el estado
        self.has_received_state = True
    
    def _wait_for_state(self, timeout=3.0):
        """Espera a recibir el estado de las articulaciones"""
        start_time = time.time()
        rate = 0.01  # 100 Hz
        
        while not self.has_received_state and time.time() - start_time < timeout:
            # Procesar callbacks de ROS
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(rate)
        
        if not self.has_received_state:
            print(f"[WARN] Timeout esperando el estado de las articulaciones", file=sys.stderr)
    
    def publish_joint_commands(self, controlled_joint_positions):
        """
        Publica comandos de trayectoria para todas las articulaciones del robot.
        Solo actualiza las 2 articulaciones controladas, mantiene las demás fijas.

        Args:
            controlled_joint_positions: Array con las posiciones de las 2 articulaciones controladas
        """
        # Crear mensaje de trayectoria
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.all_joint_names

        # Crear punto de trayectoria
        point = JointTrajectoryPoint()

        # Inicializar con posiciones fijas
        point.positions = self.fixed_joint_positions.copy()

        # Actualizar solo las articulaciones controladas
        for i, controlled_idx in enumerate(self.controlled_indices):
            point.positions[controlled_idx] = float(controlled_joint_positions[i])

        # Configurar tiempo de ejecución (3 segundos como en tu ejemplo)
        point.time_from_start = Duration(sec=3, nanosec=0)

        # Añadir punto a la trayectoria
        trajectory_msg.points = [point]

        # Publicar comando
        self.trajectory_pub.publish(trajectory_msg)

        # Procesar callbacks para recibir actualizaciones más rápido
        rclpy.spin_once(self.node, timeout_sec=0.01)

        print(f"[DEBUG] Enviado comando: {self.controlled_joints[0]}={controlled_joint_positions[0]:.3f}, {self.controlled_joints[1]}={controlled_joint_positions[1]:.3f}")
    
    def forward_kinematics(self, controlled_joint_positions):
        """
        Calcula la posición del efector final usando la cinemática REAL del MyCobot 280.

        Args:
            controlled_joint_positions: Array con los ángulos de las 2 articulaciones controladas
                                      [theta_link2_to_link3, theta_link4_to_link5]

        Returns:
            Array con la posición del efector final [x, y, z] en metros
        """
        # Crear array completo de 6 articulaciones con posiciones fijas
        full_joint_angles = self.fixed_joint_positions.copy()

        # Actualizar solo las articulaciones controladas
        for i, controlled_idx in enumerate(self.controlled_indices):
            full_joint_angles[controlled_idx] = controlled_joint_positions[i]

        # Cinemática directa completa usando las transformaciones reales del XACRO
        T = np.eye(4)  # Matriz de transformación inicial

        # Aplicar todas las transformaciones secuencialmente
        transforms_and_rotations = [
            # Base a Link1 (fijo)
            (self.joint_transforms['base_to_link1'], 0.0),

            # Link1 a Link2 (rotación Z + traslación)
            (self.joint_transforms['link1_to_link2'], full_joint_angles[0]),

            # Link2 a Link3 (rotación Z + traslación) - CONTROLADA
            (self.joint_transforms['link2_to_link3'], full_joint_angles[1]),

            # Link3 a Link4 (rotación Z + traslación)
            (self.joint_transforms['link3_to_link4'], full_joint_angles[2]),

            # Link4 a Link5 (rotación Z + traslación) - CONTROLADA
            (self.joint_transforms['link4_to_link5'], full_joint_angles[3]),

            # Link5 a Link6 (rotación Z + traslación)
            (self.joint_transforms['link5_to_link6'], full_joint_angles[4]),

            # Link6 a Flange (rotación Z + traslación)
            (self.joint_transforms['link6_to_flange'], full_joint_angles[5])
        ]

        # Aplicar cada transformación
        for translation, rotation_angle in transforms_and_rotations:
            # Matriz de traslación
            T_trans = np.eye(4)
            T_trans[:3, 3] = translation

            # Matriz de rotación en Z
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            T_rot = np.eye(4)
            T_rot[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]

            # Aplicar transformación: primero traslación, luego rotación
            T = T @ T_trans @ T_rot

        # Extraer posición del efector final
        end_effector_position = T[:3, 3]

        return np.array(end_effector_position, dtype=np.float32)

    def calculate_reward(self, distance, previous_distance=None):
        """
        Calcula la recompensa basada en la distancia al objetivo y la mejora respecto a la distancia anterior.

        Args:
            distance: Distancia actual al objetivo
            previous_distance: Distancia anterior al objetivo (opcional)

        Returns:
            Recompensa calculada
        """
        # Normalizar la distancia para que esté en el rango [0, 1]
        normalized_distance = distance / self.max_distance

        # Recompensa base inversamente proporcional a la distancia
        # Usamos una función exponencial para dar más recompensa cuando estamos cerca del objetivo
        reward = -normalized_distance**2

        # Bonificación por éxito
        if distance < self.success_threshold:
            reward += 10.0  # Bonificación grande por alcanzar el objetivo

        # Bonificación/penalización por mejora/empeoramiento
        if previous_distance is not None:
            # Normalizar la mejora
            improvement = (previous_distance - distance) / self.max_distance
            reward += improvement * 5.0  # Recompensar mejora o penalizar empeoramiento

        # Pequeña penalización por cada paso para fomentar soluciones rápidas
        reward -= 0.01

        return reward

    def get_observation(self):
        """
        Construye el vector de observación para el agente.

        Returns:
            Array con la observación completa
        """
        target_position = np.array(self.target_position, dtype=np.float32)

        observation = np.concatenate([
            self.current_joint_positions,
            self.current_joint_velocities,
            target_position,
            self.current_end_effector_position
        ]).astype(np.float32)

        return observation

    def step(self, action):
        """
        Ejecuta una acción en el entorno y devuelve el nuevo estado, recompensa, etc.

        Args:
            action: Array con los cambios en las posiciones de las articulaciones [delta_joint1, delta_joint2]

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Obtener posición actual
        joint1_pos = float(self.current_joint_positions[0])
        joint2_pos = float(self.current_joint_positions[1])

        # Aplicar acción (cambios en las posiciones de las articulaciones)
        joint1_pos += float(action[0])
        joint2_pos += float(action[1])

        # Limitar ángulos
        joint1_pos = np.clip(joint1_pos, self.joint_min, self.joint_max)
        joint2_pos = np.clip(joint2_pos, self.joint_min, self.joint_max)

        # Guardar posición inicial para verificar movimiento
        initial_positions = np.copy(self.current_joint_positions)
        initial_ee_position = np.copy(self.current_end_effector_position)

        # Actualizar posición
        if not self.fast_mode:
            # Publicar comandos
            self.publish_joint_commands([joint1_pos, joint2_pos])

            # Esperar y procesar callbacks para obtener el estado actualizado
            movement_detected = False
            max_attempts = 30

            for attempt in range(max_attempts):
                # Procesar callbacks de ROS
                rclpy.spin_once(self.node, timeout_sec=0.05)
                time.sleep(0.05)

                # Verificar si ha habido movimiento
                position_diff = np.abs(self.current_joint_positions - initial_positions).sum()
                ee_diff = np.linalg.norm(self.current_end_effector_position - initial_ee_position)

                if position_diff > 0.01 or ee_diff > 0.01:  # Umbral de movimiento
                    movement_detected = True
                    break

            if not movement_detected:
                print(f"[WARN] No se detectó movimiento después de {max_attempts} intentos")
        else:
            # En modo rápido, actualizamos directamente el estado interno
            self.current_joint_positions = np.array([joint1_pos, joint2_pos], dtype=np.float32)
            # En modo rápido, simulamos velocidades basadas en el cambio de posición
            self.current_joint_velocities = np.array([action[0], action[1]], dtype=np.float32) / 0.1  # dt estimado
            self.current_end_effector_position = self.forward_kinematics(self.current_joint_positions)

        # Calcular distancia al objetivo
        distance = np.linalg.norm(self.current_end_effector_position - np.array(self.target_position))

        # Calcular recompensa
        reward = self.calculate_reward(distance, self.previous_distance)

        # Verificar si hemos terminado
        terminated = distance < self.success_threshold
        truncated = self.current_step >= self.max_episode_steps

        # Actualizar historial de distancias
        self.previous_distance = distance

        # Obtener observación
        observation = self.get_observation()

        # Información adicional
        info = {
            'joint_positions': self.current_joint_positions,
            'joint_velocities': self.current_joint_velocities,
            'end_effector_position': self.current_end_effector_position,
            'distance_to_target': distance,
            'target_position': self.target_position
        }

        if terminated:
            print(f"[INFO] ¡Éxito! Objetivo alcanzado. " +
                  f"Posición final: joints=[{self.current_joint_positions[0]:.2f}, {self.current_joint_positions[1]:.2f}] rad, " +
                  f"end_effector=({self.current_end_effector_position[0]:.2f}, {self.current_end_effector_position[1]:.2f}, {self.current_end_effector_position[2]:.2f}), " +
                  f"Distancia: {distance:.4f}, Pasos: {self.current_step}")
        elif truncated:
            print(f"[INFO] Episodio terminado por límite de pasos. " +
                  f"Posición final: joints=[{self.current_joint_positions[0]:.2f}, {self.current_joint_positions[1]:.2f}] rad, " +
                  f"end_effector=({self.current_end_effector_position[0]:.2f}, {self.current_end_effector_position[1]:.2f}, {self.current_end_effector_position[2]:.2f}), " +
                  f"Distancia: {distance:.4f}, Pasos: {self.current_step}")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno y devuelve el estado inicial.

        Args:
            seed: Semilla para la generación de números aleatorios
            options: Opciones adicionales (no usado)

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Reiniciar contador de pasos
        self.current_step = 0

        # Establecer posición inicial aleatoria para las 2 articulaciones controladas
        joint1_pos = np.random.uniform(self.joint_min, self.joint_max)
        joint2_pos = np.random.uniform(self.joint_min, self.joint_max)

        # Guardar posición anterior para verificar movimiento
        if not self.fast_mode and hasattr(self, 'current_joint_positions'):
            previous_positions = np.copy(self.current_joint_positions)
        else:
            previous_positions = np.array([0.0, 0.0])

        # Establecer nueva posición
        self.current_joint_positions = np.array([joint1_pos, joint2_pos], dtype=np.float32)
        self.current_joint_velocities = np.zeros(2, dtype=np.float32)

        # Publicar posición inicial si no estamos en modo rápido
        if not self.fast_mode:
            self.publish_joint_commands([joint1_pos, joint2_pos])

            # Esperar a que el robot se mueva a la posición inicial
            movement_detected = False
            max_attempts = 30

            for attempt in range(max_attempts):
                # Procesar callbacks de ROS
                rclpy.spin_once(self.node, timeout_sec=0.05)
                time.sleep(0.05)

                # Verificar si ha habido movimiento
                position_diff = np.abs(self.current_joint_positions - previous_positions).sum()

                if position_diff > 0.01:
                    movement_detected = True
                    break

            if not movement_detected:
                print(f"[WARN] No se detectó movimiento en reset después de {max_attempts} intentos")

        # Calcular posición del efector final
        self.current_end_effector_position = self.forward_kinematics(self.current_joint_positions)

        # Reiniciar historial de distancias
        self.previous_distance = np.linalg.norm(self.current_end_effector_position - np.array(self.target_position))

        # Obtener observación
        observation = self.get_observation()

        # Información adicional
        info = {
            'initial_joints': self.current_joint_positions,
            'joint_positions': self.current_joint_positions,
            'joint_velocities': self.current_joint_velocities,
            'end_effector_position': self.current_end_effector_position,
            'distance_to_target': self.previous_distance,
            'target_position': self.target_position
        }

        # Solo mostrar información al inicio de cada episodio
        print(f"[INFO] Nuevo episodio. Target: {self.target_position}, " +
              f"Posición inicial: joints=[{joint1_pos:.2f}, {joint2_pos:.2f}] rad, " +
              f"end_effector=({self.current_end_effector_position[0]:.2f}, {self.current_end_effector_position[1]:.2f}, {self.current_end_effector_position[2]:.2f}), " +
              f"Distancia: {self.previous_distance:.4f}")

        return observation, info

    def render(self):
        """
        Renderiza el entorno (no implementado, se usa visualización externa).

        Returns:
            None
        """
        # La visualización se realiza externamente (Gazebo, Rviz)
        pass

    def close(self):
        """
        Cierra el entorno y libera recursos.

        Returns:
            None
        """
        if not self.fast_mode and hasattr(self, 'node'):
            self.node.destroy_node()
            # No cerramos rclpy.shutdown() para permitir múltiples instancias


if __name__ == "__main__":
    # Prueba simple del entorno
    env = MyCobotSB3Env(target_position=(0.0, 0.0, 0.191), fast_mode=True)
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}, Reward: {reward}, Done: {terminated or truncated}")

        if terminated or truncated:
            break

    env.close()
