# Entrenamiento y Test de MyCobot (PPO 3D) con Gazebo + RViz (ROS 2)

Este documento explica cómo lanzar la simulación del robot MyCobot en Gazebo/RViz y cómo entrenar y testear un modelo PPO en 3D usando el script `train_3d.py` de esta carpeta.

## Requisitos

- ROS 2 configurado y el workspace compilado (`/home/migue/mycobot_ws`)
  - Recomendado: `colcon build` y `source /home/migue/mycobot_ws/install/setup.bash` en cada terminal
- Gazebo (Ignition/GZ) funcionando con ROS 2 y el paquete `mycobot_gazebo`
- Python 3 y librerías:
  - `stable-baselines3`, `gymnasium`, `numpy`, `tensorboard`, `torch`
  - (ROS ya provee `rclpy`, `sensor_msgs`, `trajectory_msgs`, `tf2_ros`)

Ejemplo de instalación Python (en un venv):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "stable-baselines3[extra]" gymnasium numpy tensorboard
# torch: según tu plataforma, ej. pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Lanzar la simulación (Gazebo + RViz)

1) Asegúrate de tener compilado y cargado el workspace ROS 2:
```bash
source /home/migue/mycobot_ws/install/setup.bash
```

2) Lanza la simulación. Si tienes el alias configurado, basta con:
```bash
robot
```
Donde el alias es:
```bash
alias robot='bash ~/mycobot_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_gazebo.sh'
```

Como alternativa directa (sin alias), puedes ejecutar el script:
```bash
bash /home/migue/mycobot_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_gazebo.sh
```

Este script lanza `ros2 launch mycobot_gazebo mycobot.gazebo.launch.py` con controladores y RViz.

## Estructura y notas del entrenamiento

- Script principal: `new_ppo/train_3d.py`
- Entorno: `new_ppo/myCobotEnv_improved_3d.py`
  - Publica a: `/arm_controller/joint_trajectory`
  - Lee de: `/joint_states`
  - Usa TF2 para `base_link -> EE_link`
- Marcadores visuales en Gazebo: `new_ppo/gazebo_marker_manager.py` (requiere comando `gz`)
- Modelos entrenados se guardan como: `ppo_mycobot_improved_3d_<steps>steps_<YYYYMMDD_HHMM>.zip`
- Logs de TensorBoard: `new_ppo/ppo_mycobot_improved_tensorboard/`

Importante: `train_3d.py` y `test` usan rutas relativas; ejecuta los comandos desde la carpeta `new_ppo`.

```bash
cd /home/migue/mycobot_ws/src/mycobot_ros2/new_ppo
```

## Entrenar un modelo (PPO 3D)

1) Abre una nueva terminal, source del workspace y ve a `new_ppo`:
```bash
source /home/migue/mycobot_ws/install/setup.bash
cd /home/migue/mycobot_ws/src/mycobot_ros2/new_ppo
```

2) Con la simulación corriendo, lanza el entrenamiento. Ejemplos:

- Entrenamiento básico (por defecto: marcadores ON, objetivo en [0.06062, 0.0, 0.3]):
```bash
python3 train_3d.py --train --timesteps 3000 --learning-rate 3e-4
```

- Cambiar objetivo, usar inicio aleatorio y ajustar PPO:
```bash
python3 train_3d.py --train \
  --timesteps 12000 \
  --learning-rate 1e-3 \
  --clip-range 0.2 \
  --n-steps 1024 \
  --target-x 0.08 --target-y 0.0 --target-z 0.27 \
  --random-start
```

- Si no tienes `gz` disponible para crear marcadores en Gazebo, desactívalos:
```bash
python3 train_3d.py --train --timesteps 3000 --no-markers
```

Parámetros disponibles:
- `--timesteps` (int): timesteps totales de entrenamiento
- `--learning-rate` (float): LR de PPO
- `--clip-range` (float): clip PPO (ej. 0.1, 0.2, 0.3)
- `--n-steps` (int): override de `n_steps` para PPO (si no, se calcula automáticamente)
- `--target-x|y|z` (float): posición objetivo del EE
- `--no-markers`: desactiva marcadores en Gazebo
- `--random-start`: estados iniciales aleatorios de articulaciones

Salida clave:
- Modelo guardado: `ppo_mycobot_improved_3d_<steps>steps_<fecha>.zip`
- Logs TensorBoard en `ppo_mycobot_improved_tensorboard/`

## Probar (test) un modelo entrenado

1) Con Gazebo corriendo, en `new_ppo` ejecuta:

- Probar el último modelo encontrado automáticamente:
```bash
python3 train_3d.py --test --episodes 5
```

- Probar un modelo concreto por número de steps:
```bash
python3 train_3d.py --test --episodes 10 --mod-steps 12512
```

- Probar un modelo por steps-fecha (prefijo):
```bash
python3 train_3d.py --test --episodes 10 --mod-steps 12512-20250903
```

Notas:
- El script busca ficheros `ppo_mycobot_improved_3d_*steps*.zip` en la carpeta actual (`new_ppo`).
- Puedes ajustar el objetivo también en test con `--target-x|y|z`.
- Opciones `--random-start` y `--no-markers` también aplican en test.

## Visualizar entrenamiento en TensorBoard

Desde `new_ppo`:
```bash
tensorboard --logdir ppo_mycobot_improved_tensorboard
```
Abre el enlace que imprime TensorBoard en el navegador.

## Consejos y resolución de problemas

- Gazebo/ROS 2 no listo:
  - Asegúrate de ver `/joint_states` y que los controladores estén cargados.
  - Re-lanza el script `robot` y espera a que RViz/Gazebo terminen de cargar.

- `gz` no encontrado (marcadores):
  - Ejecuta con `--no-markers` o instala GZ/Ignition con el cliente `gz`.

- TF2 frame `EE_link` no disponible:
  - Verifica que el URDF/tf de `mycobot_gazebo` publica `EE_link` (o ajusta el frame en `myCobotEnv_improved_3d.py` en `get_ee_position`).

- Modelos no aparecen en test:
  - Asegúrate de ejecutar desde `new_ppo` y que existen archivos `ppo_mycobot_improved_3d_*steps*.zip`.
  - Usa `--mod-steps` para seleccionar explícitamente.

- Rendimiento/PPO:
  - Ajusta `--n-steps`, `--clip-range`, `--learning-rate` según estabilidad.
  - `n_steps` se auto-ajusta si no lo especificas; para entrenamientos cortos (<1000) se reduce automáticamente.

## Referencia rápida de comandos

- Lanzar simulación:
```bash
source /home/migue/mycobot_ws/install/setup.bash
robot
```

- Entrenar:
```bash
cd /home/migue/mycobot_ws/src/mycobot_ros2/new_ppo
python3 train_3d.py --train --timesteps 3000
```

- Testear:
```bash
python3 train_3d.py --test --episodes 5
```

- TensorBoard:
```bash
tensorboard --logdir ppo_mycobot_improved_tensorboard
```

---

Si necesitas orquestación completa (lanzar Gazebo y entrenamiento desde un solo comando), existe `launch_complete_training.py`. Requiere que tu workspace esté compilado y puede esperar a que Gazebo esté listo automáticamente. En este README nos centramos en `train_3d.py` para mantener el flujo claro.