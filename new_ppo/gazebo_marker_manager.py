
#!/usr/bin/env python3
"""
Gazebo Marker Manager para crear marcadores visuales que representen objetivos.
Permite spawnar esferas en Gazebo Harmony para visualizar las posiciones objetivo.
"""

import subprocess
import time
import random

class GazeboMarkerManager:
    """Gestor de marcadores en Gazebo para visualizar objetivos"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.target_marker_name = None
    
    def spawn_target_marker(self, x, y, z, radius=0.05):
        """Spawna una esfera verde en la posición objetivo"""
        if self.target_marker_name:
            self.remove_target_marker()
        
        self.target_marker_name = f"target_marker_{int(time.time())}"
        
        cmd = [
            "gz", "service", "-s", "/world/default/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", f'sdf: "<sdf version=\\"1.6\\"><model name=\\"{self.target_marker_name}\\"><pose>{x} {y} {z} 0 0 0</pose><link name=\\"link\\"><visual name=\\"visual\\"><geometry><sphere><radius>{radius}</radius></sphere></geometry><material><ambient>0 1 0 0.8</ambient><diffuse>0 1 0 0.8</diffuse></material></visual></link><static>true</static></model></sdf>"'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Marcador objetivo spawneado en ({x:.3f}, {y:.3f}, {z:.3f})")
                return True
            else:
                print(f"❌ Error spawneando marcador objetivo: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error ejecutando comando objetivo: {e}")
            return False
    
    def update_target_marker(self, x, y, z, radius=0.05):
        """Actualiza la posición del marcador objetivo"""
        # Si ya existe un marcador, lo eliminamos y creamos uno nuevo
        if self.target_marker_name:
            self.remove_target_marker()
            time.sleep(0.1)  # Pequeña pausa para asegurar que se elimine
        
        # Crear nuevo marcador en la nueva posición
        return self.spawn_target_marker(x, y, z, radius)
    
    def remove_target_marker(self):
        """Elimina el marcador objetivo"""
        if not self.target_marker_name:
            return
        
        cmd = [
            "gz", "service", "-s", "/world/default/remove",
            "--reqtype", "gz.msgs.Entity",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", f'name: "{self.target_marker_name}"'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            print(f"🗑️ Target marker eliminado: {self.target_marker_name}")
        except Exception:
            pass
        
        self.target_marker_name = None
    
    def remove_marker(self):
        """Elimina todos los marcadores"""
        self.remove_target_marker()
    
    def cleanup(self):
        """Limpia todos los marcadores al cerrar"""
        self.remove_marker()

def test_marker_manager():
    """Función de prueba para el gestor de marcadores."""
    manager = GazeboMarkerManager()
    test_position = (0.1, 0.0, 0.191)
    print(f"🧪 Probando spawn de marcador en {test_position}")
    
    if manager.spawn_target_marker(test_position):
        print("✅ Marcador creado exitosamente")
        time.sleep(3)
        manager.cleanup()
    else:
        print("❌ Error creando marcador")

if __name__ == "__main__":
    test_marker_manager()
