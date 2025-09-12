#!/usr/bin/env python3

"""
Script de prueba para verificar la comunicación ROS2 con el MyCobot.
Prueba el envío de comandos de trayectoria y la recepción de estados.
"""

import time
import numpy as np
from mycobot_sb3_env import MyCobotSB3Env


def test_ros_communication():
    """Prueba la comunicación ROS2 con el MyCobot"""
    
    print("=" * 60)
    print("PRUEBA DE COMUNICACIÓN ROS2 - MyCobot")
    print("=" * 60)
    print("IMPORTANTE: Asegúrate de que:")
    print("1. El robot MyCobot esté ejecutándose en Gazebo")
    print("2. Los controladores estén activos")
    print("3. El tópico /joint_states esté publicando")
    print("4. El tópico /arm_controller/joint_trajectory esté disponible")
    print()
    
    # Verificar tópicos disponibles
    print("Verificando tópicos ROS2...")
    import subprocess
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=5)
        topics = result.stdout.strip().split('\n')
        
        joint_states_available = '/joint_states' in topics
        trajectory_available = '/arm_controller/joint_trajectory' in topics
        
        print(f"  /joint_states: {'✅ DISPONIBLE' if joint_states_available else '❌ NO ENCONTRADO'}")
        print(f"  /arm_controller/joint_trajectory: {'✅ DISPONIBLE' if trajectory_available else '❌ NO ENCONTRADO'}")
        
        if not joint_states_available or not trajectory_available:
            print("\n⚠️  ADVERTENCIA: Algunos tópicos no están disponibles.")
            print("   Asegúrate de que el robot esté ejecutándose correctamente.")
            print("   La prueba continuará pero puede fallar.")
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  No se pudo verificar tópicos (ros2 no disponible o timeout)")
    
    print("\n" + "-" * 60)
    
    # Crear entorno SIN modo rápido (para usar ROS2)
    print("Creando entorno MyCobot (modo ROS2)...")
    try:
        env = MyCobotSB3Env(target_position=(0.15, 0.0, 0.3), fast_mode=False)
        print("✅ Entorno creado exitosamente")
    except Exception as e:
        print(f"❌ Error creando entorno: {e}")
        return False
    
    print("\nEsperando conexión con ROS2...")
    time.sleep(2)
    
    # Verificar si recibimos estados de articulaciones
    if env.has_received_state:
        print("✅ Estados de articulaciones recibidos")
        print(f"   Posiciones actuales: {env.current_joint_positions}")
        print(f"   Velocidades actuales: {env.current_joint_velocities}")
    else:
        print("⚠️  No se han recibido estados de articulaciones")
        print("   Esto puede indicar problemas de comunicación")
    
    print("\n" + "-" * 60)
    print("INICIANDO PRUEBAS DE MOVIMIENTO...")
    print("-" * 60)
    
    # Prueba 1: Reset del entorno
    print("\n1. Probando reset del entorno...")
    try:
        obs, info = env.reset()
        print("✅ Reset exitoso")
        print(f"   Observación shape: {obs.shape}")
        print(f"   Posición inicial EE: {info['end_effector_position']}")
        print(f"   Distancia al objetivo: {info['distance_to_target']:.4f}m")
    except Exception as e:
        print(f"❌ Error en reset: {e}")
        env.close()
        return False
    
    # Esperar a que el robot se mueva
    print("   Esperando movimiento del robot...")
    time.sleep(4)  # Tiempo para que se ejecute la trayectoria
    
    # Prueba 2: Ejecutar algunos pasos
    print("\n2. Probando pasos de acción...")
    
    test_actions = [
        np.array([0.1, -0.1]),   # Mover articulaciones en direcciones opuestas
        np.array([-0.05, 0.15]), # Movimiento más pequeño
        np.array([0.0, -0.1]),   # Solo mover una articulación
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n   Paso {i+1}: Acción {action}")
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   ✅ Paso ejecutado")
            print(f"      Recompensa: {reward:.4f}")
            print(f"      Posición EE: ({info['end_effector_position'][0]:.3f}, {info['end_effector_position'][1]:.3f}, {info['end_effector_position'][2]:.3f})")
            print(f"      Distancia: {info['distance_to_target']:.4f}m")
            print(f"      Terminado: {terminated}, Truncado: {truncated}")
            
            # Esperar a que se ejecute la trayectoria
            print("      Esperando ejecución...")
            time.sleep(4)
            
        except Exception as e:
            print(f"   ❌ Error en paso {i+1}: {e}")
            break
    
    print("\n" + "-" * 60)
    print("RESUMEN DE LA PRUEBA")
    print("-" * 60)
    
    # Estado final
    if env.has_received_state:
        print("✅ Comunicación ROS2 funcionando")
        print(f"   Posiciones finales de articulaciones controladas: {env.current_joint_positions}")
        print(f"   Posición final del efector final: {env.current_end_effector_position}")
        print(f"   Todas las posiciones de articulaciones: {env.fixed_joint_positions}")
    else:
        print("❌ Problemas de comunicación ROS2")
    
    # Cerrar entorno
    env.close()
    print("\n✅ Prueba completada. Entorno cerrado.")
    
    return True


def main():
    """Función principal"""
    print("Iniciando prueba de comunicación ROS2...")
    
    try:
        success = test_ros_communication()
        if success:
            print("\n🎉 PRUEBA EXITOSA")
            print("   El entorno puede comunicarse correctamente con ROS2")
            print("   Puedes proceder a entrenar/probar sin --fast")
        else:
            print("\n❌ PRUEBA FALLIDA")
            print("   Revisa la configuración de ROS2 y el robot")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Prueba interrumpida por el usuario")
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
