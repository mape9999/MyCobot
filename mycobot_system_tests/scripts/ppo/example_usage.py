#!/usr/bin/env python3

"""
Ejemplo de uso del sistema de entrenamiento PPO para MyCobot.
Muestra cómo entrenar y probar modelos tanto en modo rápido como con ROS2.
"""

import os
import subprocess
import time


def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {description}")
    print(f"COMANDO: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrumpido por el usuario")
        return False


def main():
    """Función principal con ejemplos de uso"""
    
    print("🤖 SISTEMA DE ENTRENAMIENTO PPO PARA MYCOBOT")
    print("=" * 60)
    print("Este script muestra ejemplos de cómo usar el sistema de entrenamiento.")
    print("Puedes ejecutar los comandos manualmente o dejar que el script los ejecute.")
    print()
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('mycobot_sb3_env.py'):
        print("❌ Error: No se encontró mycobot_sb3_env.py")
        print("   Asegúrate de ejecutar este script desde el directorio ppo/")
        return
    
    examples = [
        {
            'name': 'Verificar alcanzabilidad de objetivo',
            'cmd': 'python3 check_reachability.py --x 0.15 --y 0.0 --z 0.3',
            'description': 'Verifica si el objetivo (0.15, 0.0, 0.3) es alcanzable'
        },
        {
            'name': 'Generar objetivos aleatorios',
            'cmd': 'python3 check_reachability.py --x 0.1 --y 0.0 --z 0.35 --suggest 3',
            'description': 'Genera 3 objetivos aleatorios alcanzables'
        },
        {
            'name': 'Entrenamiento rápido (simulación)',
            'cmd': 'python3 train_mycobot_ppo.py --train --fast --timesteps 10000 --target-x 0.15 --target-y 0.0 --target-z 0.3',
            'description': 'Entrena un modelo en modo rápido (10k pasos para demo)'
        },
        {
            'name': 'Prueba del modelo (simulación)',
            'cmd': 'python3 train_mycobot_ppo.py --test --fast --episodes 3',
            'description': 'Prueba el modelo entrenado en modo rápido (3 episodios)'
        },
        {
            'name': 'Prueba de comunicación ROS2',
            'cmd': 'python3 test_ros_communication.py',
            'description': 'Verifica la comunicación con ROS2 (requiere robot activo)'
        },
        {
            'name': 'Entrenamiento con ROS2 (robot real)',
            'cmd': 'python3 train_mycobot_ppo.py --train --timesteps 5000 --target-x 0.15 --target-y 0.0 --target-z 0.3',
            'description': 'Entrena con robot real (5k pasos para demo, requiere ROS2)'
        },
        {
            'name': 'Prueba con ROS2 (robot real)',
            'cmd': 'python3 train_mycobot_ppo.py --test --episodes 2',
            'description': 'Prueba con robot real (2 episodios, requiere ROS2)'
        }
    ]
    
    while True:
        print("\n📋 EJEMPLOS DISPONIBLES:")
        print("-" * 40)
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['name']}")
            print(f"   {example['description']}")
            print(f"   Comando: {example['cmd']}")
            print()
        
        print("0. Salir")
        print("\n" + "=" * 60)
        
        try:
            choice = input("Selecciona un ejemplo (0-7): ").strip()
            
            if choice == '0':
                print("👋 ¡Hasta luego!")
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    
                    # Advertencias especiales
                    if 'ROS2' in example['name'] or not '--fast' in example['cmd']:
                        print("\n⚠️  ADVERTENCIA: Este ejemplo requiere:")
                        print("   - Robot MyCobot ejecutándose en Gazebo")
                        print("   - Controladores activos")
                        print("   - Tópicos ROS2 disponibles")
                        
                        confirm = input("\n¿Continuar? (y/N): ").strip().lower()
                        if confirm != 'y':
                            print("Ejemplo cancelado.")
                            continue
                    
                    # Ejecutar ejemplo
                    success = run_command(example['cmd'], example['name'])
                    
                    if success:
                        print(f"\n🎉 {example['name']} completado exitosamente!")
                    else:
                        print(f"\n❌ {example['name']} falló.")
                    
                    input("\nPresiona Enter para continuar...")
                    
                else:
                    print("❌ Opción inválida. Selecciona un número entre 0-7.")
            
            except ValueError:
                print("❌ Por favor ingresa un número válido.")
        
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break


def show_quick_start():
    """Muestra una guía de inicio rápido"""
    print("\n🚀 GUÍA DE INICIO RÁPIDO")
    print("=" * 60)
    print("1. VERIFICAR OBJETIVO:")
    print("   python3 check_reachability.py --x 0.15 --y 0.0 --z 0.3")
    print()
    print("2. ENTRENAR MODELO (RÁPIDO):")
    print("   python3 train_mycobot_ppo.py --train --fast --timesteps 100000")
    print()
    print("3. PROBAR MODELO (RÁPIDO):")
    print("   python3 train_mycobot_ppo.py --test --fast --episodes 10")
    print()
    print("4. PROBAR COMUNICACIÓN ROS2:")
    print("   python3 test_ros_communication.py")
    print()
    print("5. ENTRENAR CON ROBOT REAL:")
    print("   python3 train_mycobot_ppo.py --train --timesteps 50000")
    print()
    print("6. PROBAR CON ROBOT REAL:")
    print("   python3 train_mycobot_ppo.py --test --episodes 5")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-start':
        show_quick_start()
    else:
        main()
