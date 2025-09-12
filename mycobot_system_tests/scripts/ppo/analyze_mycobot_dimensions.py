#!/usr/bin/env python3

"""
Análisis de las dimensiones reales del MyCobot 280 basado en el XACRO.
Calcula el espacio de trabajo real y actualiza los parámetros del entorno.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MyCobotKinematics:
    """Cinemática del MyCobot 280 basada en el XACRO real"""
    
    def __init__(self):
        # Dimensiones REALES extraídas del XACRO del MyCobot 280
        self.joint_transforms = {
            'base_to_link1': [0, 0, 0],  # Origen del robot
            'link1_to_link2': [0, 0, 0.13156],  # 131.56mm hacia arriba
            'link2_to_link3': [0, 0, -0.001],   # Prácticamente en el mismo lugar
            'link3_to_link4': [-0.1104, 0, 0],  # 110.4mm hacia atrás (en X)
            'link4_to_link5': [-0.096, 0, 0.06062],  # 96mm hacia atrás, 60.62mm hacia arriba
            'link5_to_link6': [0, -0.07318, 0],  # 73.18mm hacia la izquierda (en Y)
            'link6_to_flange': [0, 0.0456, 0]   # 45.6mm hacia la derecha (en Y)
        }
        
        # Nombres de articulaciones en orden
        self.joint_names = [
            'link1_to_link2',
            'link2_to_link3', 
            'link3_to_link4',
            'link4_to_link5',
            'link5_to_link6',
            'link6_to_link6_flange'
        ]
        
        # Límites de articulaciones (del XACRO)
        self.joint_limits = {
            'link1_to_link2': [-2.879793, 2.879793],    # ±165°
            'link2_to_link3': [-2.879793, 2.879793],    # ±165°
            'link3_to_link4': [-2.879793, 2.879793],    # ±165°
            'link4_to_link5': [-2.879793, 2.879793],    # ±165°
            'link5_to_link6': [-2.879793, 2.879793],    # ±165°
            'link6_to_link6_flange': [-3.05, 3.05]      # ±175°
        }
        
        # Articulaciones que controlamos
        self.controlled_joints = ['link2_to_link3', 'link4_to_link5']
        self.controlled_indices = [1, 3]  # Índices en el array de 6 articulaciones
    
    def forward_kinematics_full(self, joint_angles):
        """
        Cinemática directa completa del MyCobot 280.
        
        Args:
            joint_angles: Array de 6 ángulos de articulaciones [rad]
            
        Returns:
            Posición del efector final [x, y, z] en metros
        """
        if len(joint_angles) != 6:
            raise ValueError("Se requieren exactamente 6 ángulos de articulaciones")
        
        # Matriz de transformación inicial (identidad)
        T = np.eye(4)
        
        # Transformaciones fijas y rotaciones de articulaciones
        transforms = [
            # Base a Link1 (fijo)
            self._translation_matrix(self.joint_transforms['base_to_link1']),
            
            # Link1 a Link2 (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link1_to_link2']) @ 
            self._rotation_z_matrix(joint_angles[0]),
            
            # Link2 a Link3 (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link2_to_link3']) @ 
            self._rotation_z_matrix(joint_angles[1]),
            
            # Link3 a Link4 (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link3_to_link4']) @ 
            self._rotation_z_matrix(joint_angles[2]),
            
            # Link4 a Link5 (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link4_to_link5']) @ 
            self._rotation_z_matrix(joint_angles[3]),
            
            # Link5 a Link6 (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link5_to_link6']) @ 
            self._rotation_z_matrix(joint_angles[4]),
            
            # Link6 a Flange (rotación Z + traslación)
            self._translation_matrix(self.joint_transforms['link6_to_flange']) @ 
            self._rotation_z_matrix(joint_angles[5])
        ]
        
        # Aplicar todas las transformaciones
        for transform in transforms:
            T = T @ transform
        
        # Extraer posición del efector final
        return T[:3, 3]
    
    def forward_kinematics_2dof(self, controlled_angles, fixed_angles=None):
        """
        Cinemática directa para solo las 2 articulaciones controladas.
        
        Args:
            controlled_angles: Array de 2 ángulos [link2_to_link3, link4_to_link5]
            fixed_angles: Array de 6 ángulos con posiciones fijas para las no controladas
            
        Returns:
            Posición del efector final [x, y, z] en metros
        """
        if fixed_angles is None:
            # Posiciones fijas por defecto (posición "home" segura)
            fixed_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Actualizar solo las articulaciones controladas
        full_angles = fixed_angles.copy()
        for i, controlled_idx in enumerate(self.controlled_indices):
            full_angles[controlled_idx] = controlled_angles[i]
        
        return self.forward_kinematics_full(full_angles)
    
    def _translation_matrix(self, translation):
        """Crea matriz de transformación de traslación"""
        T = np.eye(4)
        T[:3, 3] = translation
        return T
    
    def _rotation_z_matrix(self, angle):
        """Crea matriz de transformación de rotación en Z"""
        c, s = np.cos(angle), np.sin(angle)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return T
    
    def calculate_workspace_2dof(self, num_samples=50):
        """
        Calcula el espacio de trabajo para las 2 articulaciones controladas.
        
        Args:
            num_samples: Número de muestras por articulación
            
        Returns:
            Array de posiciones alcanzables [N, 3]
        """
        # Límites de las articulaciones controladas
        joint1_limits = self.joint_limits['link2_to_link3']
        joint2_limits = self.joint_limits['link4_to_link5']
        
        # Generar muestras
        joint1_angles = np.linspace(joint1_limits[0], joint1_limits[1], num_samples)
        joint2_angles = np.linspace(joint2_limits[0], joint2_limits[1], num_samples)
        
        positions = []
        for j1 in joint1_angles:
            for j2 in joint2_angles:
                try:
                    pos = self.forward_kinematics_2dof([j1, j2])
                    positions.append(pos)
                except:
                    continue
        
        return np.array(positions)
    
    def analyze_workspace(self):
        """Analiza el espacio de trabajo y muestra estadísticas"""
        print("=" * 60)
        print("ANÁLISIS DEL MYCOBOT 280 - DIMENSIONES REALES")
        print("=" * 60)
        
        # Mostrar transformaciones
        print("Transformaciones entre links (del XACRO):")
        for name, transform in self.joint_transforms.items():
            print(f"  {name}: {transform} m")
        
        print(f"\nArticulaciones controladas: {self.controlled_joints}")
        print(f"Índices controlados: {self.controlled_indices}")
        
        # Calcular espacio de trabajo
        print("\nCalculando espacio de trabajo...")
        workspace = self.calculate_workspace_2dof(num_samples=30)
        
        if len(workspace) > 0:
            # Estadísticas del espacio de trabajo
            min_pos = np.min(workspace, axis=0)
            max_pos = np.max(workspace, axis=0)
            
            print(f"\nEspacio de trabajo alcanzable (2 DOF):")
            print(f"  X: [{min_pos[0]:.3f}, {max_pos[0]:.3f}] m")
            print(f"  Y: [{min_pos[1]:.3f}, {max_pos[1]:.3f}] m") 
            print(f"  Z: [{min_pos[2]:.3f}, {max_pos[2]:.3f}] m")
            
            # Comparar con valores estimados en el código actual
            print(f"\n📊 COMPARACIÓN CON CÓDIGO ACTUAL:")
            print(f"  Código actual - X: [-0.169, 0.169] m")
            print(f"  Real calculado - X: [{min_pos[0]:.3f}, {max_pos[0]:.3f}] m")
            print(f"  Código actual - Y: [0.000, 0.000] m (FIJO)")
            print(f"  Real calculado - Y: [{min_pos[1]:.3f}, {max_pos[1]:.3f}] m")
            print(f"  Código actual - Z: [0.066, 0.404] m")
            print(f"  Real calculado - Z: [{min_pos[2]:.3f}, {max_pos[2]:.3f}] m")
            
            # Verificar objetivo por defecto
            default_target = [0.15, 0.0, 0.3]
            print(f"\n🎯 VERIFICACIÓN OBJETIVO POR DEFECTO {default_target}:")
            x_ok = min_pos[0] <= default_target[0] <= max_pos[0]
            y_ok = min_pos[1] <= default_target[1] <= max_pos[1]
            z_ok = min_pos[2] <= default_target[2] <= max_pos[2]
            
            print(f"  X = {default_target[0]:.2f}: {'✅' if x_ok else '❌'}")
            print(f"  Y = {default_target[1]:.2f}: {'✅' if y_ok else '❌'}")
            print(f"  Z = {default_target[2]:.2f}: {'✅' if z_ok else '❌'}")
            
            overall_ok = x_ok and y_ok and z_ok
            print(f"  RESULTADO: {'✅ ALCANZABLE' if overall_ok else '❌ NO ALCANZABLE'}")
            
            return workspace, (min_pos, max_pos)
        else:
            print("❌ No se pudo calcular el espacio de trabajo")
            return None, None


def main():
    """Función principal"""
    kinematics = MyCobotKinematics()
    workspace, bounds = kinematics.analyze_workspace()
    
    if workspace is not None and len(workspace) > 0:
        print(f"\n💡 RECOMENDACIONES:")
        min_pos, max_pos = bounds
        
        # Sugerir objetivos seguros
        safe_margin = 0.02  # 2cm de margen
        safe_x_range = [min_pos[0] + safe_margin, max_pos[0] - safe_margin]
        safe_y_range = [min_pos[1] + safe_margin, max_pos[1] - safe_margin]
        safe_z_range = [min_pos[2] + safe_margin, max_pos[2] - safe_margin]
        
        print(f"  Rangos seguros (con margen de {safe_margin*1000:.0f}mm):")
        print(f"    X: [{safe_x_range[0]:.3f}, {safe_x_range[1]:.3f}] m")
        print(f"    Y: [{safe_y_range[0]:.3f}, {safe_y_range[1]:.3f}] m")
        print(f"    Z: [{safe_z_range[0]:.3f}, {safe_z_range[1]:.3f}] m")
        
        # Generar objetivos seguros
        print(f"\n  Objetivos seguros sugeridos:")
        safe_targets = [
            [np.mean(safe_x_range), np.mean(safe_y_range), np.mean(safe_z_range)],
            [safe_x_range[0], safe_y_range[0], safe_z_range[1]],
            [safe_x_range[1], safe_y_range[1], safe_z_range[0]],
            [0.0, np.mean(safe_y_range), np.mean(safe_z_range)]
        ]
        
        for i, target in enumerate(safe_targets, 1):
            print(f"    {i}. ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
