#!/usr/bin/env python3

import numpy as np
import argparse


def check_reachability(target_x, target_y, target_z):
    """
    Verifica si una posición objetivo es alcanzable con las 2 articulaciones controladas.
    
    Args:
        target_x, target_y, target_z: Coordenadas del objetivo
        
    Returns:
        bool: True si es alcanzable, False si no
    """
    # Rangos REALES calculados del análisis del XACRO del MyCobot 280
    # Basado en el análisis completo de cinemática directa
    min_x, max_x = -0.233, 0.227  # Rango X real
    min_y, max_y = -0.233, 0.233  # Rango Y real (NO es fijo!)
    min_z, max_z = 0.191, 0.191   # Rango Z real (muy limitado con 2 DOF)
    
    print("=" * 60)
    print("VERIFICADOR DE ALCANZABILIDAD - MyCobot 280 (2 DOF)")
    print("=" * 60)
    print(f"Configuración basada en análisis REAL del XACRO:")
    print(f"  Articulaciones controladas: link2_to_link3, link4_to_link5")
    print(f"  Cinemática completa de 6 DOF con 4 articulaciones fijas")
    print()

    print(f"Espacio de trabajo alcanzable (REAL):")
    print(f"  X: [{min_x:.3f}, {max_x:.3f}]m")
    print(f"  Y: [{min_y:.3f}, {max_y:.3f}]m (¡NO es fijo!)")
    print(f"  Z: [{min_z:.3f}, {max_z:.3f}]m (muy limitado con 2 DOF)")
    print()
    
    # Verificar cada coordenada
    print(f"Verificando objetivo: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
    print("-" * 40)

    x_ok = min_x <= target_x <= max_x
    y_ok = min_y <= target_y <= max_y
    z_ok = min_z <= target_z <= max_z

    print(f"X = {target_x:.3f}m: {'✅ ALCANZABLE' if x_ok else '❌ FUERA DE RANGO'}")
    if not x_ok:
        print(f"  Rango válido: [{min_x:.3f}, {max_x:.3f}]m")

    print(f"Y = {target_y:.3f}m: {'✅ ALCANZABLE' if y_ok else '❌ FUERA DE RANGO'}")
    if not y_ok:
        print(f"  Rango válido: [{min_y:.3f}, {max_y:.3f}]m")

    print(f"Z = {target_z:.3f}m: {'✅ ALCANZABLE' if z_ok else '❌ FUERA DE RANGO'}")
    if not z_ok:
        print(f"  Rango válido: [{min_z:.3f}, {max_z:.3f}]m")
    
    print()
    overall_ok = x_ok and y_ok and z_ok
    
    if overall_ok:
        print("🎯 RESULTADO: ✅ OBJETIVO ALCANZABLE")
        print("   Puedes usar este objetivo para entrenar/probar el modelo.")
    else:
        print("🚫 RESULTADO: ❌ OBJETIVO NO ALCANZABLE")
        print("   Ajusta las coordenadas para que estén dentro del rango.")
        
        # Sugerir alternativas
        print("\n💡 Sugerencias de objetivos alcanzables (REALES):")
        suggestions = [
            (0.0, 0.0, 0.191),      # Centro del espacio de trabajo
            (0.1, 0.1, 0.191),      # Hacia adelante y derecha
            (-0.1, -0.1, 0.191),    # Hacia atrás y izquierda
            (0.2, 0.0, 0.191),      # Máximo alcance en X
            (-0.2, 0.2, 0.191)      # Combinación X negativo, Y positivo
        ]
        
        for i, (sx, sy, sz) in enumerate(suggestions, 1):
            print(f"   {i}. ({sx:.2f}, {sy:.1f}, {sz:.2f})")
    
    print("=" * 60)
    return overall_ok


def suggest_random_targets(n=5):
    """Genera objetivos aleatorios alcanzables"""
    # Rangos REALES del MyCobot 280
    min_x, max_x = -0.233, 0.227
    min_y, max_y = -0.233, 0.233
    min_z, max_z = 0.191, 0.191  # Z es prácticamente fijo con 2 DOF

    print(f"\n🎲 {n} OBJETIVOS ALEATORIOS ALCANZABLES (REALES):")
    print("-" * 40)

    for i in range(n):
        # Generar coordenadas aleatorias dentro del rango válido con margen de seguridad
        safety_margin = 0.02  # 2cm de margen
        x = np.random.uniform(min_x + safety_margin, max_x - safety_margin)
        y = np.random.uniform(min_y + safety_margin, max_y - safety_margin)
        z = 0.191  # Z es prácticamente constante

        print(f"   {i+1}. ({x:.3f}, {y:.3f}, {z:.3f})")

    print("\nPuedes usar cualquiera de estos con:")
    print("python train_mycobot_ppo.py --train --fast --target-x X --target-y Y --target-z Z")


def main():
    parser = argparse.ArgumentParser(description='Verificar si un objetivo es alcanzable para MyCobot')
    parser.add_argument('--x', type=float, required=True, help='Coordenada X del objetivo')
    parser.add_argument('--y', type=float, required=True, help='Coordenada Y del objetivo')
    parser.add_argument('--z', type=float, required=True, help='Coordenada Z del objetivo')
    parser.add_argument('--suggest', type=int, default=0, help='Generar N objetivos aleatorios alcanzables')
    
    args = parser.parse_args()
    
    # Verificar objetivo especificado
    is_reachable = check_reachability(args.x, args.y, args.z)
    
    # Generar sugerencias si se solicita
    if args.suggest > 0:
        suggest_random_targets(args.suggest)
    
    return 0 if is_reachable else 1


if __name__ == "__main__":
    exit(main())
