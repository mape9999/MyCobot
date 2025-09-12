from stable_baselines3 import PPO

# Ruta del archivo .zip de tu modelo
model_path = input ("Introduce el nombre del modelo:") 

# Cargar el modelo entrenado
model = PPO.load(model_path)

# Acceder a los hiperparámetros
learning_rate = model.learning_rate
n_steps = model.n_steps
batch_size = model.batch_size
gamma = model.gamma
gae_lambda = model.gae_lambda
clip_range = model.clip_range
timesteps = model.num_timesteps

print(f"Learning Rate: {learning_rate}")
print(f"Número de pasos (n_steps): {n_steps}")
print(f"Tamaño del batch: {batch_size}")
print(f"Factor de descuento (gamma): {gamma}")
print(f"GAE Lambda: {gae_lambda}")
print(f"Rango de clipping (clip_range): {clip_range}")
print(f"Cantidad total de pasos realizados: {timesteps}")
