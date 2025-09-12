#!/usr/bin/env python3
"""
Script de lanzamiento completo para entrenamiento con detección de colisiones
Este script maneja todo el proceso de forma automática
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import threading
from pathlib import Path

class ProcessManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def add_process(self, process):
        self.processes.append(process)
        
    def cleanup(self):
        print("\n🛑 Cerrando todos los procesos...")
        self.running = False
        
        for process in self.processes:
            if process and process.poll() is None:
                print(f"🔄 Terminando proceso {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚠️ Forzando cierre del proceso {process.pid}...")
                    process.kill()
        
        print("✅ Todos los procesos cerrados.")

def setup_environment():
    """Configura el entorno ROS2"""
    print("🔧 Configurando entorno ROS2...")
    
    # Verificar que el workspace esté compilado
    install_path = Path('/home/migue/mycobot_ws/install')
    if not install_path.exists():
        print("❌ Workspace no compilado. Ejecuta: colcon build")
        return False
    
    # Source del workspace
    setup_script = install_path / 'setup.bash'
    if not setup_script.exists():
        print("❌ setup.bash no encontrado")
        return False
    
    print("✅ Entorno ROS2 configurado")
    return True

def launch_gazebo():
    """Lanza Gazebo con MyCobot"""
    print("🚀 Lanzando Gazebo con MyCobot...")
    
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = '0'
    
    cmd = [
        'bash', '-c',
        'source /home/migue/mycobot_ws/install/setup.bash && '
        'ros2 launch mycobot_gazebo mycobot_gazebo.launch.py'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        print(f"✅ Gazebo lanzado (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Error lanzando Gazebo: {e}")
        return None

def launch_collision_detector():
    """Lanza el detector de colisiones"""
    print("🚀 Lanzando detector de colisiones...")
    
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = '0'
    
    cmd = [
        'bash', '-c',
        'source /home/migue/mycobot_ws/install/setup.bash && '
        'ros2 run collision_detector self_collision_detector'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        print(f"✅ Detector de colisiones lanzado (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Error lanzando detector de colisiones: {e}")
        return None

def wait_for_gazebo():
    """Espera a que Gazebo esté listo"""
    print("⏳ Esperando a que Gazebo esté listo...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            # Verificar si los tópicos de Gazebo están disponibles
            result = subprocess.run(
                ['bash', '-c', 
                 'source /home/migue/mycobot_ws/install/setup.bash && '
                 'ros2 topic list | grep -q "/joint_states"'],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Gazebo está listo")
                return True
                
        except subprocess.TimeoutExpired:
            pass
        
        print(f"⏳ Intento {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("⚠️ Timeout esperando a Gazebo, continuando...")
    return False

def run_training(args):
    """Ejecuta el entrenamiento"""
    print("🚀 Iniciando entrenamiento...")
    
    # Cambiar al directorio correcto
    os.chdir('/home/migue/mycobot_ws/src/mycobot_ros2/new_ppo')
    
    # Configurar entorno Python
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/migue/mycobot_ws/src/mycobot_ros2/new_ppo:' + env.get('PYTHONPATH', '')
    env['ROS_DOMAIN_ID'] = '0'
    
    # Comando de entrenamiento
    cmd = [
        'bash', '-c',
        f'source /home/migue/mycobot_ws/install/setup.bash && '
        f'python3 train_improved.py'
    ]
    
    # Si se especificaron argumentos personalizados, usar el script integrado
    if any([args.timesteps != 3000, args.learning_rate != 3e-4, 
            args.show_markers, args.fast_mode, args.random_start]):
        
        script_args = []
        if args.timesteps != 3000:
            script_args.append(f'--timesteps {args.timesteps}')
        if args.learning_rate != 3e-4:
            script_args.append(f'--learning-rate {args.learning_rate}')
        if args.show_markers:
            script_args.append('--show-markers')
        if args.fast_mode:
            script_args.append('--fast-mode')
        if args.random_start:
            script_args.append('--random-start')
        
        cmd = [
            'bash', '-c',
            f'source /home/migue/mycobot_ws/install/setup.bash && '
            f'python3 scripts/train_with_collision_detection.py --no-collision-detector {" ".join(script_args)}'
        ]
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"✅ Entrenamiento iniciado (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Error iniciando entrenamiento: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Lanzamiento completo de entrenamiento con detección de colisiones')
    
    # Argumentos de entrenamiento
    parser.add_argument('--timesteps', type=int, default=3000,
                       help='Número de timesteps para entrenar (default: 3000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--show-markers', action='store_true',
                       help='Mostrar marcadores visuales en RViz')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Activar modo rápido (menos esperas)')
    parser.add_argument('--random-start', action='store_true',
                       help='Usar posiciones iniciales aleatorias')
    
    # Argumentos de lanzamiento
    parser.add_argument('--no-gazebo', action='store_true',
                       help='No lanzar Gazebo (usar uno existente)')
    parser.add_argument('--no-collision-detector', action='store_true',
                       help='No lanzar detector de colisiones (usar uno existente)')
    parser.add_argument('--wait-time', type=int, default=10,
                       help='Tiempo de espera entre lanzamientos (segundos)')
    
    args = parser.parse_args()
    
    print("🤖 LANZAMIENTO COMPLETO DE ENTRENAMIENTO CON DETECCIÓN DE COLISIONES")
    print("="*80)
    print(f"🎯 Timesteps: {args.timesteps}")
    print(f"📚 Learning rate: {args.learning_rate}")
    print(f"🔵 Marcadores: {'SÍ' if args.show_markers else 'NO'}")
    print(f"⚡ Modo rápido: {'SÍ' if args.fast_mode else 'NO'}")
    print(f"🎲 Inicio aleatorio: {'SÍ' if args.random_start else 'NO'}")
    print("="*80)
    
    # Configurar manejador de procesos
    process_manager = ProcessManager()
    
    def signal_handler(sig, frame):
        process_manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 1. Configurar entorno
        if not setup_environment():
            return 1
        
        # 2. Lanzar Gazebo si es necesario
        if not args.no_gazebo:
            gazebo_process = launch_gazebo()
            if gazebo_process:
                process_manager.add_process(gazebo_process)
                
                # Esperar a que Gazebo esté listo
                time.sleep(args.wait_time)
                wait_for_gazebo()
            else:
                print("❌ No se pudo lanzar Gazebo")
                return 1
        else:
            print("⚠️ Usando Gazebo existente")
        
        # 3. Lanzar detector de colisiones si es necesario
        if not args.no_collision_detector:
            collision_process = launch_collision_detector()
            if collision_process:
                process_manager.add_process(collision_process)
                time.sleep(3)  # Esperar a que el detector se inicialice
            else:
                print("❌ No se pudo lanzar el detector de colisiones")
                return 1
        else:
            print("⚠️ Usando detector de colisiones existente")
        
        # 4. Ejecutar entrenamiento
        training_process = run_training(args)
        if training_process:
            process_manager.add_process(training_process)
            
            # Esperar a que termine el entrenamiento
            training_process.wait()
            print("🎉 ¡Entrenamiento completado!")
        else:
            print("❌ No se pudo iniciar el entrenamiento")
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupción por teclado recibida.")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        return 1
    finally:
        process_manager.cleanup()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())