#!/bin/bash

echo "🚀 INSTALANDO DEPENDENCIAS PARA MYCOBOT PPO TRAINING"
echo "=" * 60

# Actualizar pip
echo "📦 Actualizando pip..."
python3 -m pip install --upgrade pip

# Instalar dependencias principales
echo "📦 Instalando dependencias principales..."
python3 -m pip install gymnasium numpy matplotlib

# Instalar PyTorch (CPU version por defecto, cambiar si tienes GPU)
echo "📦 Instalando PyTorch..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar Stable Baselines3
echo "📦 Instalando Stable Baselines3..."
python3 -m pip install stable-baselines3[extra]

# Instalar TensorBoard para monitoreo
echo "📦 Instalando TensorBoard..."
python3 -m pip install tensorboard

# Verificar instalación
echo ""
echo "✅ VERIFICANDO INSTALACIÓN..."
echo "=" * 60

python3 -c "
import gymnasium as gym
import numpy as np
import torch
import stable_baselines3 as sb3
import tensorboard
print('✅ gymnasium:', gym.__version__)
print('✅ numpy:', np.__version__)
print('✅ torch:', torch.__version__)
print('✅ stable-baselines3:', sb3.__version__)
print('✅ CUDA disponible:', torch.cuda.is_available())
print('')
print('🎉 ¡Todas las dependencias instaladas correctamente!')
"

echo ""
echo "📋 PRÓXIMOS PASOS:"
echo "1. Verificar objetivo: python3 ppo/check_reachability.py --x 0.0 --y 0.0 --z 0.191"
echo "2. Entrenar modelo: python3 ppo/train_mycobot_ppo.py --train --fast --timesteps 10000"
echo "3. Probar modelo: python3 ppo/train_mycobot_ppo.py --test --fast --episodes 5"
echo ""
echo "🔧 Para usar con robot real (sin --fast), asegúrate de que:"
echo "   - El robot esté ejecutándose en Gazebo"
echo "   - Los tópicos /joint_states y /arm_controller/joint_trajectory estén disponibles"
