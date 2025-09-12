import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Ruta al directorio donde está el archivo .tfevents
#log_dir = "/home/migue/rrbot2_ws/src/rrbot_description/test/ppo/models_origen_simple/logs/ppo_rrbot_origen_simple_4/"

log_dir = "/home/migue/mycobot_ws/src/mycobot_ros2/new_ppo/ppo_mycobot_improved_tensorboard/PPO_21/"

# Cargar el log
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Ver qué métricas hay disponibles
print("Scalars disponibles:", ea.Tags()["scalars"])

# Lista de métricas típicas de PPO en SB3
metrics = [
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/approx_kl"
]

# Graficar cada métrica si existe
plt.figure(figsize=(12,8))
for metric in metrics:
    if metric in ea.Tags()["scalars"]:
        data = ea.Scalars(metric)
        df = pd.DataFrame([(s.step, s.value) for s in data], columns=["timesteps", metric])
        plt.plot(df["timesteps"], df[metric], label=metric)

plt.xlabel("Timesteps")
plt.ylabel("Valor")
plt.title("Evolución de métricas de PPO durante el entrenamiento")
plt.legend()
plt.grid()
plt.show()
