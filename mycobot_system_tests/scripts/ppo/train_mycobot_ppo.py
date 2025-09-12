#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from mycobot_sb3_env import MyCobotSB3Env


def make_env(target_position, fast_mode, rank=0, seed=0):
    """
    Función auxiliar para crear un entorno con una semilla específica.
    
    Args:
        target_position: Posición objetivo para el efector final (x, y, z)
        fast_mode: Si es True, se usa el modo rápido sin ROS
        rank: Índice del entorno (para entornos paralelos)
        seed: Semilla para la generación de números aleatorios
        
    Returns:
        Función que crea un entorno
    """
    def _init():
        env = MyCobotSB3Env(target_position=target_position, fast_mode=fast_mode)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(target_position=(0.0, 0.0, 0.191), num_timesteps=100000, n_envs=1, fast_mode=True, save_dir=None):
    """
    Entrena un agente PPO para controlar el MyCobot.
    
    Args:
        target_position: Posición objetivo para el efector final (x, y, z)
        num_timesteps: Número total de pasos de entrenamiento
        n_envs: Número de entornos paralelos (solo funciona con fast_mode=True)
        fast_mode: Si es True, se usa el modo rápido sin ROS
        save_dir: Directorio para guardar los modelos y logs
        
    Returns:
        Modelo entrenado
    """
    # Configurar directorio para guardar modelos y logs
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "models")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    
    # Detectar dispositivo disponible
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    # Crear entornos vectorizados
    if fast_mode and n_envs > 1:
        # Solo usar entornos paralelos en modo rápido
        env = SubprocVecEnv([make_env(target_position, fast_mode, i) for i in range(n_envs)])
    else:
        # En modo normal, usar un solo entorno
        env = DummyVecEnv([make_env(target_position, fast_mode)])
    
    # Configurar callbacks para guardar modelos y evaluar
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="mycobot_ppo_model"
    )
    
    # Crear modelo PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,  # Especificar dispositivo explícitamente
        tensorboard_log=os.path.join(save_dir, "logs"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            )
        )
    )
    
    # Entrenar modelo
    print(f"Iniciando entrenamiento PPO para MyCobot...")
    print(f"Posición objetivo: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f})")
    print(f"Pasos totales: {num_timesteps}")
    print(f"Entornos paralelos: {n_envs}")
    print(f"Modo rápido: {'Activado' if fast_mode else 'Desactivado'}")
    print("-" * 50)
    
    start_time = time.time()
    model.learn(
        total_timesteps=num_timesteps,
        callback=checkpoint_callback,
        tb_log_name="ppo_mycobot"
    )
    total_time = time.time() - start_time
    
    print(f"Entrenamiento completado en {total_time:.2f} segundos")
    
    # Guardar modelo final
    model_path = os.path.join(save_dir, "mycobot_ppo_final.zip")
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")
    
    return model


def test_ppo(model_path, target_position=(0.0, 0.0, 0.191), num_episodes=10, fast_mode=False):
    """
    Prueba un modelo PPO entrenado.
    
    Args:
        model_path: Ruta al modelo entrenado
        target_position: Posición objetivo para el efector final (x, y, z)
        num_episodes: Número de episodios de prueba
        fast_mode: Si es True, se usa el modo rápido sin ROS
        
    Returns:
        None
    """
    # Crear entorno de prueba
    env = MyCobotSB3Env(target_position=target_position, fast_mode=fast_mode)
    
    # Cargar modelo
    model = PPO.load(model_path)
    
    # Estadísticas
    success_count = 0
    total_steps = 0
    distances = []
    
    print(f"\nIniciando {num_episodes} pruebas...")
    print(f"Posición objetivo: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f})")
    print("-" * 50)
    
    for episode in range(1, num_episodes + 1):
        # Reiniciar entorno
        obs, info = env.reset()
        
        # Ejecutar episodio
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            # Seleccionar acción
            action, _ = model.predict(obs, deterministic=True)
            
            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            
            steps += 1
            done = terminated
            
            # Añadir pequeña pausa para visualización
            if not fast_mode:
                time.sleep(0.05)
        
        # Estadísticas
        distance = info['distance_to_target']
        distances.append(distance)
        total_steps += steps
        
        # Comprobar éxito
        success = distance < env.success_threshold
        if success:
            success_count += 1
            result = "ÉXITO"
        else:
            result = "FALLO"
        
        print(f"Episodio {episode}/{num_episodes} - {result} | " +
              f"Distancia: {distance:.4f} m | Pasos: {steps}")
    
    # Resultados finales
    success_rate = success_count / num_episodes * 100
    avg_distance = np.mean(distances)
    avg_steps = total_steps / num_episodes
    
    print("\nResultados finales:")
    print(f"Tasa de éxito: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Distancia promedio: {avg_distance:.4f} m")
    print(f"Pasos promedio: {avg_steps:.1f}")
    
    # Cerrar entorno
    env.close()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenamiento PPO para MyCobot')
    parser.add_argument('--train', action='store_true', help='Modo entrenamiento')
    parser.add_argument('--test', action='store_true', help='Modo prueba')
    parser.add_argument('--target-x', type=float, default=0.0, help='Coordenada X del objetivo')
    parser.add_argument('--target-y', type=float, default=0.0, help='Coordenada Y del objetivo')
    parser.add_argument('--target-z', type=float, default=0.191, help='Coordenada Z del objetivo')
    parser.add_argument('--timesteps', type=int, default=100000, help='Número de pasos de entrenamiento')
    parser.add_argument('--episodes', type=int, default=10, help='Número de episodios para prueba')
    parser.add_argument('--fast', action='store_true', help='Activar modo rápido (sin ROS)')
    parser.add_argument('--n-envs', type=int, default=1, help='Número de entornos paralelos (solo con --fast)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directorio para guardar modelos')
    parser.add_argument('--model', type=str, default=None, help='Ruta al modelo para pruebas')
    
    args = parser.parse_args()
    
    # Configurar posición objetivo
    target_position = (args.target_x, args.target_y, args.target_z)
    
    if args.train:
        # Modo entrenamiento
        print("Iniciando modo entrenamiento...")
        train_ppo(
            target_position=target_position,
            num_timesteps=args.timesteps,
            n_envs=args.n_envs,
            fast_mode=args.fast,
            save_dir=args.save_dir
        )
    elif args.test:
        # Modo prueba
        print("Iniciando modo prueba...")
        
        # Determinar ruta del modelo
        if args.model is None:
            # Buscar en la carpeta models
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "models")
            model_path = os.path.join(models_dir, "mycobot_ppo_final.zip")
        else:
            model_path = args.model
        
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            print(f"Error: No se encontró el modelo en {model_path}")
            return
        
        # Ejecutar pruebas
        test_ppo(
            model_path=model_path,
            target_position=target_position,
            num_episodes=args.episodes,
            fast_mode=args.fast
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
