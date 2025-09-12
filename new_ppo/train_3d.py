#!/usr/bin/env python3

import argparse
import time
import os
import glob
import re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from myCobotEnv_improved_3d import MyCobotEnvImproved3D
from datetime import datetime

class StopTrainingOnTimesteps(BaseCallback):
    """Callback para parar exactamente en el número de timesteps especificado"""
    def __init__(self, max_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.max_timesteps = max_timesteps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_timesteps

class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso, estadísticas y loggear a TensorBoard"""
    def __init__(self, verbose=0, hyperparams: dict | None = None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.best_reward = float('-inf')
        self.last_log_time = time.time()
        self.hyperparams = hyperparams or {}

    def _on_training_start(self) -> None:
        # Log estático de hiperparámetros
        if self.model and hasattr(self.model, 'logger'):
            for k, v in self.hyperparams.items():
                try:
                    self.model.logger.record(f'hparams/{k}', float(v))
                except Exception:
                    # Si no es convertible a float, ignorar
                    pass

    def _on_step(self) -> bool:
        # Log por-step
        infos = self.locals.get('infos', [])
        if infos:
            info = infos[0]
            # Métricas continuas
            if self.model and hasattr(self.model, 'logger'):
                self.model.logger.record('env/distance_to_target', float(info.get('distance_to_target', np.nan)))
                self.model.logger.record('env/min_distance_reached', float(info.get('min_distance_reached', np.nan)))
                self.model.logger.record('env/best_distance', float(info.get('best_distance', np.nan)))
                self.model.logger.record('env/consecutive_improvements', float(info.get('consecutive_improvements', 0)))
                # EE y joints por componente
                ee = info.get('ee_position')
                if isinstance(ee, (list, tuple)) and len(ee) == 3:
                    self.model.logger.record('env/ee_x', float(ee[0]))
                    self.model.logger.record('env/ee_y', float(ee[1]))
                    self.model.logger.record('env/ee_z', float(ee[2]))
                joints = info.get('controlled_joint_positions')
                if isinstance(joints, (list, tuple)) and len(joints) == 3:
                    self.model.logger.record('env/joint_0', float(joints[0]))
                    self.model.logger.record('env/joint_1', float(joints[1]))
                    self.model.logger.record('env/joint_2', float(joints[2]))
                target = info.get('target_position')
                if isinstance(target, (list, tuple)) and len(target) == 3:
                    self.model.logger.record('env/target_x', float(target[0]))
                    self.model.logger.record('env/target_y', float(target[1]))
                    self.model.logger.record('env/target_z', float(target[2]))
                if 'reward' in info:
                    self.model.logger.record('rollout/reward', float(info['reward']))

        # Log por-episodio y consola
        if len(infos) > 0:
            info = infos[0]
            if self.locals.get('dones', [False])[0]:
                self.total_episodes += 1
                episode_reward = self.locals.get('rewards', [0])[0]
                self.episode_rewards.append(episode_reward)
                if info.get('success', False):
                    self.success_count += 1
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                current_time = time.time()
                if (self.total_episodes % 10 == 0 or current_time - self.last_log_time > 30):
                    success_rate = (self.success_count / max(self.total_episodes, 1)) * 100
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    # Log a TB
                    if self.model and hasattr(self.model, 'logger'):
                        self.model.logger.record('eval/success_rate', float(success_rate))
                        self.model.logger.record('eval/avg_reward_last_10', float(avg_reward))
                        self.model.logger.record('eval/best_episode_reward', float(self.best_reward))
                    # Consola
                    print(f"📊 Episodio {self.total_episodes} | Steps: {self.num_timesteps} | "
                          f"Éxitos: {success_rate:.1f}% | Reward promedio: {avg_reward:.2f} | "
                          f"Mejor: {self.best_reward:.2f}")
                    self.last_log_time = current_time
        return True


def train_mycobot_3d(target_position, timesteps, learning_rate, show_markers, random_start, clip_range=0.2, n_steps_override=None):
    """Entrena el modelo en 3D (controlando 3 articulaciones)"""
    print("🚀 ENTRENAMIENTO 3D DE MYCOBOT")
    print(f"🎯 Objetivo: {target_position}")
    print(f"⏱️ Timesteps: {timesteps}")
    print(f"📚 Learning rate: {learning_rate}")
    print(f"🔵 Marcadores: {'SÍ' if show_markers else 'NO'}")
    print(f"🎲 Inicio aleatorio: {'SÍ' if random_start else 'NO'}")
    print(f"🪙 clip_range: {clip_range}")
    print("=" * 60)

    def make_env_func():
        # Env envuelto con Monitor para logging episodico detallado
        return Monitor(MyCobotEnvImproved3D(
            target_position=target_position,
            max_episode_steps=150,
            success_threshold=0.05,
            show_markers=show_markers,
            random_start=random_start
        ))

    env = DummyVecEnv([make_env_func])

    # Calcular n_steps
    if n_steps_override is not None:
        n_steps = int(n_steps_override)
    else:
        n_steps = min(timesteps, 1024)
        if timesteps < 1000:
            n_steps = max(timesteps // 3, 64)
    print(f"🔧 n_steps usado: {n_steps}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=min(256, max(64, n_steps//2)),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_mycobot_improved_tensorboard/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )

    print("🤖 Modelo PPO 3D creado")
    print("🧠 Arquitectura: pi=[128, 128], vf=[128, 128]")
    print(f"📊 Iniciando entrenamiento por {timesteps} timesteps...")

    # Construir nombres coherentes para modelo y logs ANTES del entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_base_name = f"ppo_mycobot_improved_3d_{timesteps}steps_{timestamp}"
    log_dir = "./ppo_mycobot_improved_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    start_time = time.time()
    stop_callback = StopTrainingOnTimesteps(max_timesteps=timesteps)
    progress_callback = ProgressCallback(hyperparams=dict(
        algo='PPO',
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=min(256, max(64, n_steps//2)),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        model_name=model_base_name
    ))

    model.learn(
        total_timesteps=timesteps,
        callback=[stop_callback, progress_callback],
        progress_bar=True,
        tb_log_name=model_base_name
    )

    training_time = time.time() - start_time

    model_name = f"{model_base_name}.zip"
    model.save(model_name)

    success_rate = (progress_callback.success_count / max(progress_callback.total_episodes, 1)) * 100
    avg_reward = np.mean(progress_callback.episode_rewards) if progress_callback.episode_rewards else 0

    print("\n🎉 ENTRENAMIENTO 3D COMPLETADO!")
    print(f"⏱️ Tiempo total: {training_time:.1f} segundos")
    print(f"📊 Episodios completados: {progress_callback.total_episodes}")
    print(f"✅ Tasa de éxito: {success_rate:.1f}%")
    print(f"🏆 Reward promedio: {avg_reward:.2f}")
    print(f"🥇 Mejor reward: {progress_callback.best_reward:.2f}")
    print(f"💾 Modelo guardado como: {model_name}")

    env.close()
    return model_name


def test_mycobot_3d(target_position, episodes, show_markers, random_start, model_steps=None):
    """Prueba el modelo 3D entrenado"""

    if model_steps is not None:
        # Permite formatos: "12505" o "12505-20250831" (con fecha), y tolera sufijos de hora
        ms = str(model_steps)
        if '-' in ms:
            steps_part, date_part = ms.split('-', 1)
            pattern = f"ppo_mycobot_improved_3d_{steps_part}steps_{date_part}*.zip"
        else:
            steps_part = ms
            pattern = f"ppo_mycobot_improved_3d_{steps_part}steps*.zip"
        candidates = glob.glob(pattern)
        if not candidates:
            print(f"❌ No se encontró modelo con patrón: {pattern}")
            print("💡 Ejemplos válidos: --mod-steps 12505 o --mod-steps 12505-20250831")
            return
        candidates.sort(key=os.path.getmtime, reverse=True)
        model_path = candidates[0]
        print(f"🔍 Modelo seleccionado por --mod-steps: {os.path.basename(model_path)}")
    else:
        model_files = glob.glob("ppo_mycobot_improved_3d_*steps*.zip")
        if not model_files:
            print("❌ No se encontraron modelos 3D entrenados")
            print("💡 Usa --mod-steps <número> o <número-fecha> para especificar el modelo")
            return
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        match = re.search(r'ppo_mycobot_improved_3d_(\d+)steps', os.path.basename(model_path))
        if match:
            detected_steps = match.group(1)
            print(f"🔍 Modelo 3D más reciente detectado: {model_path} ({detected_steps} steps)")

    print(f"🧪 PROBANDO MODELO 3D: {model_path}")
    print(f"🎯 Objetivo: {target_position}")
    print(f"📊 Episodios de prueba: {episodes}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo {model_path}")
        return

    env = MyCobotEnvImproved3D(
        target_position=target_position,
        max_episode_steps=150,
        success_threshold=0.05,
        show_markers=show_markers,
        random_start=random_start
    )

    model = PPO.load(model_path)

    successes = 0
    total_rewards = []
    best_distances = []

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        best_distance_episode = float('inf')

        print(f"\n🎮 Episodio {episode + 1}/{episodes}")
        print(f"📏 Distancia inicial: {info['distance_to_target']*1000:.1f}mm")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            current_distance = info['distance_to_target']
            if current_distance < best_distance_episode:
                best_distance_episode = current_distance

            if steps % 20 == 0:
                print(f"   Step {steps}: Dist={current_distance*1000:.1f}mm, "
                      f"Mejor={best_distance_episode*1000:.1f}mm, "
                      f"Reward={episode_reward:.1f}")

            if terminated or truncated:
                if info['success']:
                    successes += 1
                    if current_distance < env.success_threshold:
                        print(f"✅ ¡Éxito Tradicional! Pasos: {steps}, Reward: {episode_reward:.2f}")
                        print(f"   📏 Distancia final: {current_distance*1000:.1f}mm")
                    else:
                        print(f"🎯 ¡Éxito por Precisión! Pasos: {steps}, Reward: {episode_reward:.2f}")
                        print(f"   📏 Mejor distancia alcanzada: {info['min_distance_reached']*1000:.1f}mm")
                        print(f"   📈 Se alejó {info['consecutive_away_steps']} pasos consecutivos")
                else:
                    print(f"❌ Fallo. Pasos: {steps}, Reward: {episode_reward:.2f}")
                    print(f"   📏 Mejor distancia: {best_distance_episode*1000:.1f}mm")
                    if 'min_distance_reached' in info:
                        print(f"   🎯 Mejor distancia del episodio: {info['min_distance_reached']*1000:.1f}mm")
                break

        total_rewards.append(episode_reward)
        best_distances.append(best_distance_episode)

    success_rate = (successes / episodes) * 100
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_best_distance = sum(best_distances) / len(best_distances)

    print("\n📊 RESULTADOS FINALES 3D:")
    print(f"✅ Tasa de éxito: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"🏆 Reward promedio: {avg_reward:.2f}")
    print(f"📏 Mejor distancia promedio: {avg_best_distance*1000:.1f}mm")
    print(f"🎯 Umbral de éxito tradicional: {env.success_threshold*1000:.1f}mm")
    print(f"🎯 Umbral de precisión: {env.precision_mode_threshold*1000:.1f}mm")
    print(f"📈 Sistema de precisión: {env.away_step_limit} pasos alejándose, {env.alejamiento_percentage*100:.1f}% incremento")

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Entrenar o probar MyCobot 3D con PPO')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo 3D')
    parser.add_argument('--test', action='store_true', help='Probar el modelo 3D entrenado')
    parser.add_argument('--timesteps', type=int, default=2000, help='Número de timesteps para entrenar')
    parser.add_argument('--episodes', type=int, default=5, help='Número de episodios para probar')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range (e.g., 0.1, 0.2, 0.3)')
    parser.add_argument('--n-steps', dest='n_steps', type=int, default=None, help='Override de n_steps para PPO (e.g., 256, 512, 1024)')
    parser.add_argument('--target-x', type=float, default=0.06062, help='Posición X objetivo')
    parser.add_argument('--target-y', type=float, default=0.0, help='Posición Y objetivo')
    parser.add_argument('--target-z', type=float, default=0.3, help='Posición Z objetivo')
    parser.add_argument('--no-markers', action='store_true', help='Desactivar marcadores en gazebo')
    parser.add_argument('--random-start', action='store_true', help='Posición inicial aleatoria')
    parser.add_argument('--mod-steps', type=str, help='Modelo a cargar: pasos (e.g., 12505) o pasos-fecha (YYYYMMDD), e.g., 12505-20250831')

    args = parser.parse_args()

    target_position = (args.target_x, args.target_y, args.target_z)
    show_markers = not args.no_markers

    if args.train:
        train_mycobot_3d(
            target_position=target_position,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            show_markers=show_markers,
            random_start=args.random_start,
            clip_range=args.clip_range,
            n_steps_override=args.n_steps
        )
    elif args.test:
        test_mycobot_3d(
            target_position=target_position,
            episodes=args.episodes,
            show_markers=show_markers,
            random_start=args.random_start,
            model_steps=args.mod_steps
        )
    else:
        print("❌ Especifica --train o --test")
        parser.print_help()

if __name__ == "__main__":
    main()