# MyCobot PPO Training Scripts

Este directorio contiene scripts para entrenar y probar agentes de aprendizaje por refuerzo (PPO) para el robot MyCobot de 6 DOF, controlando específicamente 2 articulaciones: `link2_to_link3` y `link4_to_link5`.

## Archivos

- `mycobot_sb3_env.py`: Entorno Gymnasium para el MyCobot compatible con Stable Baselines3
- `train_mycobot_ppo.py`: Script principal para entrenar y probar modelos PPO
- `check_reachability.py`: Verificador de alcanzabilidad de objetivos
- `test_ros_communication.py`: Script de prueba para comunicación ROS2
- `example_usage.py`: Ejemplos de uso del sistema
- `README.md`: Este archivo de documentación

## Características del Entorno

### Articulaciones Controladas
- **link2_to_link3**: Segunda articulación del brazo
- **link4_to_link5**: Cuarta articulación del brazo

### Espacio de Observación (10 dimensiones)
- Posiciones de las 2 articulaciones controladas
- Velocidades de las 2 articulaciones controladas  
- Posición objetivo 3D (x, y, z)
- Posición actual del efector final 3D (x, y, z)

### Espacio de Acción (2 dimensiones)
- Cambios incrementales en las posiciones de las 2 articulaciones controladas
- Rango: [-0.2, 0.2] radianes por paso

### Modos de Operación
- **Modo Normal**: Se conecta a ROS2 y controla el robot real/simulado
  - Publica en: `/arm_controller/joint_trajectory` (trajectory_msgs/JointTrajectory)
  - Escucha de: `/joint_states` (sensor_msgs/JointState)
  - Compatible con Gazebo y robot real
- **Modo Rápido**: Simulación interna sin ROS para entrenamiento acelerado

### Comunicación ROS2
El sistema usa la misma interfaz que tu comando manual:
```bash
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: ['link1_to_link2', 'link2_to_link3', 'link3_to_link4', 'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'], \
points: [{positions: [0.0, -1.23, 0.0, -0.296, 0.0, 0.0], time_from_start: {sec: 3.0, nanosec: 0}}]}"
```

**Articulaciones controladas**: Solo `link2_to_link3` y `link4_to_link5` se mueven, las demás permanecen fijas.

## Instalación de Dependencias

```bash
pip install stable-baselines3[extra] gymnasium numpy torch tensorboard
```

## Uso

### Verificación Previa

Antes de entrenar, verifica que tu objetivo sea alcanzable:
```bash
# Verificar objetivo
python3 check_reachability.py --x 0.15 --y 0.0 --z 0.3

# Generar objetivos aleatorios válidos
python3 check_reachability.py --x 0.1 --y 0.0 --z 0.35 --suggest 5
```

### Prueba de Comunicación ROS2

Si planeas usar el robot real (sin --fast), prueba la comunicación primero:
```bash
# Verificar comunicación ROS2 (requiere robot activo)
python3 test_ros_communication.py
```

### Entrenamiento

```bash
# Entrenamiento básico en modo rápido
python train_mycobot_ppo.py --train --fast

# Entrenamiento con posición objetivo personalizada
python train_mycobot_ppo.py --train --fast --target-x 0.1 --target-y 0.0 --target-z 0.35

# Entrenamiento con más pasos y entornos paralelos
python train_mycobot_ppo.py --train --fast --timesteps 500000 --n-envs 4

# Entrenamiento con robot real (requiere ROS2 activo)
python train_mycobot_ppo.py --train --target-x 0.15 --target-y 0.0 --target-z 0.3
```

### Pruebas

```bash
# Probar modelo entrenado en modo rápido
python train_mycobot_ppo.py --test --fast

# Probar con posición objetivo diferente
python train_mycobot_ppo.py --test --fast --target-x 0.1 --target-y 0.3 --target-z 0.35

# Probar con robot real
python train_mycobot_ppo.py --test --target-x 0.2 --target-y 0.2 --target-z 0.3

# Probar modelo específico
python train_mycobot_ppo.py --test --model /path/to/model.zip --episodes 20
```

## Parámetros Principales

### Argumentos de Línea de Comandos

- `--train`: Activar modo entrenamiento
- `--test`: Activar modo prueba
- `--target-x/y/z`: Coordenadas del objetivo (metros)
- `--timesteps`: Número de pasos de entrenamiento (default: 100000)
- `--episodes`: Número de episodios para prueba (default: 10)
- `--fast`: Activar modo rápido sin ROS
- `--n-envs`: Número de entornos paralelos (solo con --fast)
- `--save-dir`: Directorio para guardar modelos
- `--model`: Ruta al modelo para pruebas

### Configuración del Entorno

- **Umbral de éxito**: 0.05 metros
- **Máximo pasos por episodio**: 200
- **Límites de articulaciones**: [-π, π] radianes

### Espacio de Trabajo Alcanzable

⚠️ **IMPORTANTE**: Solo controlando 2 articulaciones, el espacio de trabajo es limitado:

- **Rango X**: [-0.169, 0.169] metros
- **Rango Y**: [0.0, 0.0] metros (FIJO - no se puede cambiar)
- **Rango Z**: [0.066, 0.404] metros

**Objetivos válidos de ejemplo:**
- (0.15, 0.0, 0.3) ✅
- (0.1, 0.0, 0.35) ✅
- (-0.12, 0.0, 0.25) ✅
- (0.0, 0.0, 0.4) ✅

**Objetivos NO alcanzables:**
- (0.2, 0.2, 0.3) ❌ (X fuera de rango, Y≠0)
- (0.0, 0.1, 0.3) ❌ (Y≠0)
- (0.18, 0.0, 0.3) ❌ (X fuera de rango)

## Estructura de Archivos Generados

```
models/
├── logs/                          # Logs de TensorBoard
├── checkpoints/                   # Checkpoints durante entrenamiento
└── mycobot_ppo_final.zip         # Modelo final entrenado
```

## Monitoreo del Entrenamiento

```bash
# Visualizar progreso con TensorBoard
tensorboard --logdir models/logs
```

## Personalización

### Modificar Articulaciones Controladas

En `mycobot_sb3_env.py`, cambiar:
```python
self.controlled_joints = ['link2_to_link3', 'link4_to_link5']
```

### Ajustar Parámetros del Robot

Modificar las longitudes de los links en:
```python
self.link_lengths = {
    'base_to_link1': 0.131,
    'link1_to_link2': 0.104,
    # ... etc
}
```

### Personalizar Recompensas

Modificar el método `calculate_reward()` en `mycobot_sb3_env.py`.

## Notas Importantes

1. **Cinemática Simplificada**: El entorno usa una aproximación de la cinemática completa del robot
2. **Calibración Necesaria**: Los parámetros del robot pueden necesitar ajuste según el modelo específico
3. **Tópicos ROS**: Verificar que los tópicos de comando coincidan con la interfaz real del robot
4. **Generalización**: El modelo puede generalizar a diferentes posiciones objetivo

## Solución de Problemas

### Error: "No se encontró la articulación X"
- Verificar nombres de articulaciones en el URDF del robot
- Ajustar `self.controlled_joints` en el entorno

### Entrenamiento muy lento
- Usar `--fast` para modo acelerado
- Aumentar `--n-envs` para paralelización

### Robot no se mueve
- Verificar conexión ROS2
- Comprobar tópicos de comando
- Revisar límites de articulaciones

## Ejemplo Completo

```bash
# 1. Entrenar modelo
python train_mycobot_ppo.py --train --fast --timesteps 200000 --target-x 0.25 --target-y 0.15 --target-z 0.35

# 2. Probar modelo entrenado
python train_mycobot_ppo.py --test --fast --episodes 20 --target-x 0.25 --target-y 0.15 --target-z 0.35

# 3. Probar con objetivo diferente
python train_mycobot_ppo.py --test --fast --target-x 0.1 --target-y 0.3 --target-z 0.4
```
