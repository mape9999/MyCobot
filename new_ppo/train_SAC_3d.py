#!/usr/bin/env python3

import argparse
import time
import os
import glob
import re
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from myCobotEnv_improved_3d import MyCobotEnvImproved3D


class StopTrainingOnTimesteps(BaseCallback):
    """Callback para parar exactamente en el número de timesteps especificado"""
    def __init__(self, max_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.max_timesteps = max_timesteps

    def _on_step(self) -> bool:
        return self.num_timesteps < self.max_timesteps


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso y estadísticas y loggear a TensorBoard"""
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
        if self.model and hasattr(self.model, 'logger'):
            for k, v in self.hyperparams.items():
                try:
                    self.model.logger.record(f'hparams/{k}', float(v))
                except Exception:
                    pass

    def _on_step(self) -> bool:
        # Log por-step
        infos = self.locals.get('infos', [])
        if infos:
            info = infos[0]
            if self.model and hasattr(self.model, 'logger'):
                self.model.logger.record('env/distance_to_target', float(info.get('distance_to_target', np.nan)))
                self.model.logger.record('env/min_distance_reached', float(info.get('min_distance_reached', np.nan)))
                self.model.logger.record('env/best_distance', float(info.get('best_distance', np.nan)))
                self.model.logger.record('env/consecutive_improvements', float(info.get('consecutive_improvements', 0)))
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
                    if self.model and hasattr(self.model, 'logger'):
                        self.model.logger.record('eval/success_rate', float(success_rate))
                        self.model.logger.record('eval/avg_reward_last_10', float(avg_reward))
                        self.model.logger.record('eval/best_episode_reward', float(self.best_reward))
                        self.model.logger.dump(step=int(self.num_timesteps))
                    print(f"📊 Episodio {self.total_episodes} | Steps: {self.num_timesteps} | "
                          f"Éxitos: {success_rate:.1f}% | Reward promedio: {avg_reward:.2f} | "
                          f"Mejor: {self.best_reward:.2f}")
                    self.last_log_time = current_time
        return True


def train_sac_mycobot_3d(target_position, timesteps, learning_rate, random_start):
    """Entrena el modelo SAC en 3D (controlando 3 articulaciones)"""
    print("🚀 ENTRENAMIENTO 3D DE MYCOBOT con SAC")
    print(f"🎯 Objetivo: {target_position}")
    print(f"⏱️ Timesteps: {timesteps}")
    print(f"📚 Learning rate: {learning_rate}")
    print(f"🎲 Inicio aleatorio: {'SÍ' if random_start else 'NO'}")
    print("=" * 60)

    def make_env_func():
        return Monitor(MyCobotEnvImproved3D(
            target_position=target_position,
            max_episode_steps=150,
            success_threshold=0.05,
            random_start=random_start
        ))

    env = DummyVecEnv([make_env_func])

    train_freq = 64 if timesteps >= 1000 else max(16, timesteps // 10)
    gradient_steps = train_freq
    batch_size = 128
    tau = 0.02

    print(f"🔧 train_freq: {train_freq}, gradient_steps: {gradient_steps}, batch_size: {batch_size}, tau: {tau}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=max(100000, timesteps),
        batch_size=batch_size,
        gamma=0.99,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=min(10000, max(1000, timesteps // 10)),
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        tensorboard_log="./sac_mycobot_improved_tensorboard/",
        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )

    print("🤖 Modelo SAC 3D creado")
    print("🧠 Arquitectura: actor/critic=[256, 256]")
    print(f"📊 Iniciando entrenamiento por {timesteps} timesteps...")

    start_time = time.time()
    stop_callback = StopTrainingOnTimesteps(max_timesteps=timesteps)
    progress_callback = ProgressCallback(hyperparams=dict(
        algo='SAC',
        learning_rate=learning_rate,
        buffer_size=max(100000, timesteps),
        batch_size=batch_size,
        gamma=0.99,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=min(10000, max(1000, timesteps // 10))
    ))

    model.learn(
        total_timesteps=timesteps,
        callback=[stop_callback, progress_callback],
        progress_bar=True,
        tb_log_name=f"SAC_{int(time.time())}"
    )

    training_time = time.time() - start_time

    model_name = f"sac_mycobot_improved_3d_{timesteps}steps.zip"
    model.save(model_name)

    success_rate = (progress_callback.success_count / max(progress_callback.total_episodes, 1)) * 100
    avg_reward = np.mean(progress_callback.episode_rewards) if progress_callback.episode_rewards else 0

    print("\n🎉 ENTRENAMIENTO 3D SAC COMPLETADO!")
    print(f"⏱️ Tiempo total: {training_time:.1f} segundos")
    print(f"📊 Episodios completados: {progress_callback.total_episodes}")
    print(f"✅ Tasa de éxito: {success_rate:.1f}%")
    print(f"🏆 Reward promedio: {avg_reward:.2f}")
    print(f"🥇 Mejor reward: {progress_callback.best_reward:.2f}")
    print(f"💾 Modelo guardado como: {model_name}")

    env.close()
    return model_name


def test_sac_mycobot_3d(target_position, episodes, random_start, model_steps=None):
    """Prueba el modelo SAC 3D entrenado"""

    if model_steps is not None:
        model_path = f"sac_mycobot_improved_3d_{model_steps}steps.zip"
    else:
        model_files = glob.glob("sac_mycobot_improved_3d_*steps.zip")
        if not model_files:
            print("❌ No se encontraron modelos 3D SAC entrenados")
            print("💡 Usa --mod-steps <número> para especificar el modelo")
            return
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        match = re.search(r'sac_mycobot_improved_3d_(\d+)steps\.zip', model_path)
        if match:
            detected_steps = match.group(1)
            print(f"🔍 Modelo SAC 3D más reciente detectado: {model_path} ({detected_steps} steps)")

    print(f"🧪 PROBANDO MODELO SAC 3D: {model_path}")
    print(f"🎯 Objetivo: {target_position}")
    print(f"📊 Episodios de prueba: {episodes}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo {model_path}")
        return

    env = MyCobotEnvImproved3D(
        target_position=target_position,
        max_episode_steps=200,
        success_threshold=0.05,
        random_start=random_start
    )

    model = SAC.load(model_path)

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

    print("\n📊 RESULTADOS FINALES 3D (SAC):")
    print(f"✅ Tasa de éxito: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"🏆 Reward promedio: {avg_reward:.2f}")
    print(f"📏 Mejor distancia promedio: {avg_best_distance*1000:.1f}mm")
    print(f"🎯 Umbral de éxito tradicional: {env.success_threshold*1000:.1f}mm")
    print(f"🎯 Umbral de precisión: {env.precision_mode_threshold*1000:.1f}mm")
    print(f"📈 Sistema de precisión: {env.away_step_limit} pasos alejándose, {env.alejamiento_percentage*100:.1f}% incremento")

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Entrenar o probar MyCobot 3D con SAC')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo 3D con SAC')
    parser.add_argument('--test', action='store_true', help='Probar el modelo 3D SAC entrenado')
    parser.add_argument('--timesteps', type=int, default=2000, help='Número de timesteps para entrenar')
    parser.add_argument('--episodes', type=int, default=5, help='Número de episodios para probar')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--target-x', type=float, default=0.06062, help='Posición X objetivo')
    parser.add_argument('--target-y', type=float, default=0.0, help='Posición Y objetivo')
    parser.add_argument('--target-z', type=float, default=0.3, help='Posición Z objetivo')
    parser.add_argument('--random-start', action='store_true', help='Posición inicial aleatoria')
    parser.add_argument('--mod-steps', type=int, help='Especificar modelo por número de steps')

    args = parser.parse_args()

    target_position = (args.target_x, args.target_y, args.target_z)

    if args.train:
        train_sac_mycobot_3d(
            target_position=target_position,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            random_start=args.random_start
        )
    elif args.test:
        test_sac_mycobot_3d(
            target_position=target_position,
            episodes=args.episodes,
            random_start=args.random_start,
            model_steps=args.mod_steps
        )
    else:
        print("❌ Especifica --train o --test")
        parser.print_help()


if __name__ == "__main__":
    main()