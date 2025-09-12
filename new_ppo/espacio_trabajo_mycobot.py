#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: espacio_trabajo_mycobot.py

Objetivo:
- Definir y muestrear el espacio de trabajo 3D del MyCobot (EE_link) variando 3 articulaciones:
  link1_to_link2 (J1), link2_to_link3 (J2), link4_to_link5 (J4 en código, índice 3).
- Ejecuta movimientos reales vía ROS2 publicando JointTrajectory.
- Lee la posición del EE con TF2 (base_link -> EE_link) y genera un scatter 3D.

Notas de seguridad/realismo:
- Respeta límites articulares del URDF: [-2.879793, 2.879793] para J1, J2 y J4.
- Controla la velocidad promedio imponiendo time_from_start dinámico según el delta:
  T >= max(|Δq|) / 2.792527 (v_max del URDF). Además aplica un mínimo configurable (por defecto 0.3s).
- Por defecto fija el resto de articulaciones a 0.0.

Uso típico:
  python3 espacio_trabajo_mycobot.py --resolution 10 --min-duration 0.3 --show

Requisitos:
- ROS2 en ejecución con robot/sim lanzado, publicando TF de EE_link y escuchando en /arm_controller/joint_trajectory
- Python deps: numpy, matplotlib
"""

import argparse
import math
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros

import matplotlib
matplotlib.use('Agg')  # Para guardar figuras sin display; --show las muestra si hay entorno gráfico
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D


class WorkspaceMapper3D(Node):
    """Nodo ROS2 para barrer el espacio de trabajo y graficar puntos EE_link."""

    def __init__(self,
                 resolution: int = 10,
                 jmin: float = -2.879793,
                 jmax: float = 2.879793,
                 min_duration: float = 0.3,
                 max_velocity: float = 2.792527,
                 settle_margin: float = 0.05,
                 spin_timeout: float = 0.05):
        super().__init__('workspace_mapper_3d')

        self.get_logger().info('🗺️  MAPEADOR ESPACIO DE TRABAJO 3D - MyCobot 280 (EE_link)')

        # Parámetros
        self.resolution = max(2, int(resolution))
        self.joint_min = float(jmin)
        self.joint_max = float(jmax)
        self.min_duration = max(0.1, float(min_duration))
        self.max_velocity = max(0.1, float(max_velocity))
        self.settle_margin = max(0.0, float(settle_margin))  # tiempo extra tras T para estabilizar
        self.spin_timeout = float(spin_timeout)

        # Joints del brazo (orden fijo)
        self.all_joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'
        ]
        # Indices controlados (J1, J2, J4)
        self.controlled_indices = [0, 1, 3]

        # Publisher de trayectoria
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )

        # TF2 para leer EE_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Datos recolectados
        self.points_xyz = []  # lista de [x,y,z]
        self.points_joints = []  # lista de [j1,j2,j4]

        # Último comando para estimar delta y ajustar T
        self.last_cmd = np.zeros(3, dtype=np.float32)

    # -------------------------- Utilidades ROS/TF -------------------------- #
    def _float_to_duration(self, seconds: float) -> Duration:
        if seconds < 0:
            seconds = 0.0
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def get_ee_position(self) -> np.ndarray:
        """Lee TF base_link->EE_link. Devuelve np.array([x,y,z]) o None si falla."""
        try:
            # Espera corta para TF
            trans = self.tf_buffer.lookup_transform(
                'base_link', 'EE_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            return np.array([x, y, z], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"TF2 no disponible (base_link->EE_link): {e}")
            return None

    def send_joint_command(self, j_cmd: np.ndarray, duration_s: float):
        """Publica un JointTrajectory a /arm_controller/joint_trajectory.
        j_cmd: np.array([j1, j2, j4]) en rad.
        duration_s: tiempo objetivo para el movimiento (s)."""
        j1, j2, j4 = float(j_cmd[0]), float(j_cmd[1]), float(j_cmd[2])

        traj = JointTrajectory()
        traj.joint_names = self.all_joint_names

        point = JointTrajectoryPoint()
        # Posiciones completas (las no controladas se fijan a 0.0)
        point.positions = [j1, j2, 0.0, j4, 0.0, 0.0]
        point.velocities = [0.0] * 6
        point.accelerations = [0.0] * 6
        point.time_from_start = self._float_to_duration(duration_s)

        traj.points = [point]
        self.trajectory_pub.publish(traj)

    # -------------------------- Lógica de barrido -------------------------- #
    def compute_duration_for_step(self, target_cmd: np.ndarray) -> float:
        """Calcula un tiempo de movimiento seguro en función del delta y v_max."""
        delta = np.max(np.abs(target_cmd - self.last_cmd))
        t_min_by_speed = delta / self.max_velocity if self.max_velocity > 0 else 0.0
        return max(self.min_duration, t_min_by_speed)

    def sweep_workspace(self):
        """Barre el espacio 3D de J1, J2, J4 y captura EE_link en cada punto."""
        self.get_logger().info(
            f"Barrido 3D con resolution={self.resolution}, jmin={self.joint_min}, jmax={self.joint_max}"
        )

        # Grid de joints
        grid = np.linspace(self.joint_min, self.joint_max, self.resolution, dtype=np.float32)

        total = self.resolution ** 3
        count = 0
        start_time = time.time()

        for j1 in grid:
            for j2 in grid:
                for j4 in grid:
                    target = np.array([j1, j2, j4], dtype=np.float32)

                    # Calcular duración segura según delta respecto al último comando
                    duration_s = self.compute_duration_for_step(target)

                    # Enviar comando
                    self.send_joint_command(target, duration_s)

                    # Esperar movimiento: T + margen
                    end_t = time.time() + duration_s + self.settle_margin
                    while time.time() < end_t:
                        rclpy.spin_once(self, timeout_sec=self.spin_timeout)
                        time.sleep(max(0.0, self.spin_timeout / 2.0))

                    # Leer EE
                    ee = self.get_ee_position()
                    if ee is not None:
                        self.points_xyz.append(ee.tolist())
                        self.points_joints.append(target.tolist())
                    else:
                        # Si TF falla, no guardamos el punto
                        pass

                    # Actualizar last_cmd
                    self.last_cmd = target.copy()

                    count += 1
                    if count % 50 == 0 or count == 1:
                        elapsed = time.time() - start_time
                        self.get_logger().info(
                            f"Progreso: {count}/{total} ({count/total*100:.1f}%) | puntos válidos: {len(self.points_xyz)} | t={elapsed:.1f}s"
                        )

        self.get_logger().info(
            f"Barrido finalizado. Puntos válidos: {len(self.points_xyz)} de {total}."
        )

    # -------------------------- Plot/Export -------------------------- #
    def save_and_plot(self, show: bool = False):
        if not self.points_xyz:
            self.get_logger().error('No hay puntos para graficar.')
            return

        pts = np.array(self.points_xyz)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Guardar CSV simple
        csv_path = f"workspace_3d_points_{ts}.csv"
        np.savetxt(csv_path, pts, delimiter=',', header='x,y,z', comments='')
        self.get_logger().info(f"Puntos guardados en: {csv_path}")

        # Gráfica 3D
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=6, alpha=0.8)
        ax.set_title('Espacio de trabajo 3D (EE_link) - MyCobot 280')
        ax.set_xlabel('X (m) - base_link')
        ax.set_ylabel('Y (m) - base_link')
        ax.set_zlabel('Z (m) - base_link')
        ax.view_init(elev=25, azim=35)
        ax.grid(True)

        png_path = f"workspace_3d_plot_{ts}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=220)
        self.get_logger().info(f"Gráfica guardada en: {png_path}")

        if show:
            try:
                plt.show()
            except Exception:
                self.get_logger().warn('No se pudo mostrar la figura (entorno sin display).')


# ------------------------------ Main ------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Barrido del espacio de trabajo 3D (EE_link) del MyCobot 280')
    parser.add_argument('--resolution', type=int, default=8,
                        help='Número de muestras por articulación (total = res^3). Ej: 8 => 512 puntos')
    parser.add_argument('--jmin', type=float, default=-2.879793, help='Límite inferior (rad) para J1,J2,J4')
    parser.add_argument('--jmax', type=float, default=2.879793, help='Límite superior (rad) para J1,J2,J4')
    parser.add_argument('--min-duration', type=float, default=0.3, help='Duración mínima por paso (s)')
    parser.add_argument('--max-velocity', type=float, default=2.792527, help='Velocidad máxima (rad/s)')
    parser.add_argument('--settle-margin', type=float, default=0.05, help='Margen extra tras el movimiento (s)')
    parser.add_argument('--spin-timeout', type=float, default=0.05, help='Timeout de spin_once (s)')
    parser.add_argument('--show', action='store_true', help='Mostrar ventana de la gráfica (si hay display)')

    args = parser.parse_args()

    rclpy.init()
    try:
        node = WorkspaceMapper3D(
            resolution=args.resolution,
            jmin=args.jmin,
            jmax=args.jmax,
            min_duration=args.min_duration,
            max_velocity=args.max_velocity,
            settle_margin=args.settle_margin,
            spin_timeout=args.spin_timeout,
        )

        # Pequeña espera para que TF2 tenga buffers
        node.get_logger().info('Inicializando TF2...')
        startup_t = time.time() + 1.0
        while time.time() < startup_t:
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.05)

        node.sweep_workspace()
        node.save_and_plot(show=args.show)

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()