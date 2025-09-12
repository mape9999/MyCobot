#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import time
import json
from datetime import datetime

class CollisionMovementTester(Node):
    def __init__(self):
        super().__init__('collision_movement_tester')
        
        # Publisher para comandos de trayectoria
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        
        # Suscriptores para monitorear el sistema
        self.collision_alert_sub = self.create_subscription(
            String,
            '/mycobot_280/self_collision_alert',
            self.collision_alert_callback,
            10
        )
        
        self.collision_details_sub = self.create_subscription(
            String,
            '/mycobot_280/collision_details',
            self.collision_details_callback,
            10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Suscriptores para sensores de contacto individuales (para debugging)
        self.contact_subscribers = {}
        contact_topics = [
            '/mycobot_280/link1_contact/contacts',
            '/mycobot_280/link2_contact/contacts',
            '/mycobot_280/link3_contact/contacts',
            '/mycobot_280/link4_contact/contacts',
            '/mycobot_280/link5_contact/contacts',
            '/mycobot_280/link6_contact/contacts',
            '/mycobot_280/gripper_contact/contacts'
        ]
        
        for topic in contact_topics:
            sub = self.create_subscription(
                String,
                topic,
                lambda msg, t=topic: self.contact_sensor_callback(msg, t),
                10
            )
            self.contact_subscribers[topic] = sub
        
        # Variables de estado
        self.current_joint_positions = None
        self.collision_count = 0
        self.movement_count = 0
        self.test_results = []
        
        # Configuraciones de prueba que DEBERÍAN causar auto-colisiones
        self.test_configurations = [
            {
                'name': 'Posición Inicial Segura',
                'positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'duration': 3.0,
                'expected_collision': False,
                'description': 'Posición neutral - no debería haber colisiones'
            },
            {
                'name': 'Colisión Link2-Link4 (Configuración 1)',
                'positions': [0.0, 1.5, -1.8, 1.8, 0.0, 0.0],
                'duration': 4.0,
                'expected_collision': True,
                'description': 'Link2 y Link4 deberían colisionar'
            },
            {
                'name': 'Colisión Link3-Link5 (Configuración 2)',
                'positions': [0.0, 0.8, 2.2, -2.5, 0.0, 0.0],
                'duration': 4.0,
                'expected_collision': True,
                'description': 'Link3 y Link5 deberían colisionar'
            },
            {
                'name': 'Configuración Extrema (Múltiples Colisiones)',
                'positions': [3.14, 1.8, -1.8, 1.8, -1.8, 3.14],
                'duration': 5.0,
                'expected_collision': True,
                'description': 'Configuración extrema con múltiples posibles colisiones'
            },
            {
                'name': 'Colisión Base-Link1',
                'positions': [0.0, -1.8, 0.5, 0.0, 0.0, 0.0],
                'duration': 4.0,
                'expected_collision': True,
                'description': 'Link1 hacia atrás para colisionar con base'
            },
            {
                'name': 'Vuelta a Posición Segura',
                'positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'duration': 3.0,
                'expected_collision': False,
                'description': 'Regreso a posición segura'
            }
        ]
        
        self.current_test = 0
        self.test_start_time = None
        self.waiting_for_movement = False
        
        # Timer para ejecutar pruebas
        self.test_timer = self.create_timer(1.0, self.run_test_sequence)
        
        self.get_logger().info("🤖 Collision Movement Tester iniciado")
        self.get_logger().info(f"📋 Se ejecutarán {len(self.test_configurations)} configuraciones de prueba")
        self.get_logger().info("🔍 Monitoreando topics de colisión y sensores de contacto")
        
        # Esperar un poco antes de empezar
        self.startup_delay = 3
        
    def collision_alert_callback(self, msg):
        """Callback para alertas de colisión"""
        self.collision_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        self.get_logger().warn(f"🚨 [{timestamp}] COLLISION ALERT #{self.collision_count}: {msg.data}")
        
        # Registrar en resultados de prueba
        if self.current_test < len(self.test_configurations):
            test_config = self.test_configurations[self.current_test]
            self.test_results.append({
                'test_name': test_config['name'],
                'collision_detected': True,
                'timestamp': timestamp,
                'alert_message': msg.data,
                'joint_positions': self.current_joint_positions.copy() if self.current_joint_positions else None
            })
    
    def collision_details_callback(self, msg):
        """Callback para detalles de colisión"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        try:
            details = json.loads(msg.data)
            self.get_logger().info(f"📋 [{timestamp}] COLLISION DETAILS:")
            self.get_logger().info(f"    - Tipo: {details.get('type', 'unknown')}")
            self.get_logger().info(f"    - Colisión 1: {details.get('collision1', 'unknown')}")
            self.get_logger().info(f"    - Colisión 2: {details.get('collision2', 'unknown')}")
            self.get_logger().info(f"    - Fuente: {details.get('source', 'unknown')}")
            self.get_logger().info(f"    - Contador: {details.get('count', 0)}")
        except json.JSONDecodeError:
            self.get_logger().info(f"📋 [{timestamp}] COLLISION DETAILS (raw): {msg.data}")
    
    def contact_sensor_callback(self, msg, topic):
        """Callback para sensores de contacto individuales"""
        if msg.data and len(msg.data) > 10:  # Solo mostrar si hay contenido significativo
            sensor_name = topic.split('/')[-2]  # Extraer nombre del sensor
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.get_logger().debug(f"📡 [{timestamp}] {sensor_name}: {msg.data[:100]}...")
    
    def joint_state_callback(self, msg):
        """Callback para estados de articulaciones"""
        if len(msg.position) >= 6:
            self.current_joint_positions = list(msg.position[:6])
    
    def send_trajectory_command(self, positions, duration_sec, test_name):
        """Envía comando de trayectoria al robot"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        trajectory_msg.joint_names = [
            'link1_to_link2',
            'link2_to_link3', 
            'link3_to_link4',
            'link4_to_link5',
            'link5_to_link6',
            'link6_to_link6_flange'
        ]
        
        # Crear punto de trayectoria
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        
        trajectory_msg.points = [point]
        
        # Publicar comando
        self.trajectory_pub.publish(trajectory_msg)
        
        # Log del comando enviado
        positions_str = [f"{pos:.3f}" for pos in positions]
        self.get_logger().info(f"🎯 Enviando comando: {test_name}")
        self.get_logger().info(f"    Posiciones: {positions_str}")
        self.get_logger().info(f"    Duración: {duration_sec}s")
        
        # Mostrar comando ROS2 equivalente
        cmd_str = f"ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
        cmd_str += f"\"{{joint_names: {trajectory_msg.joint_names}, "
        cmd_str += f"points: [{{positions: {positions}, time_from_start: {{sec: {int(duration_sec)}, nanosec: {int((duration_sec % 1) * 1e9)}}}}}]}}\""
        self.get_logger().info(f"💻 Comando equivalente:")
        self.get_logger().info(f"    {cmd_str}")
    
    def run_test_sequence(self):
        """Ejecuta la secuencia de pruebas"""
        # Delay inicial
        if self.startup_delay > 0:
            self.startup_delay -= 1
            if self.startup_delay == 0:
                self.get_logger().info("🚀 Iniciando secuencia de pruebas...")
            return
        
        # Verificar si hemos terminado todas las pruebas
        if self.current_test >= len(self.test_configurations):
            if not hasattr(self, 'final_report_shown'):
                self.show_final_report()
                self.final_report_shown = True
            return
        
        # Si estamos esperando que termine un movimiento
        if self.waiting_for_movement:
            if self.test_start_time and (time.time() - self.test_start_time) > (self.test_configurations[self.current_test]['duration'] + 2):
                self.waiting_for_movement = False
                self.current_test += 1
                self.get_logger().info("⏭️  Movimiento completado, continuando con siguiente prueba...\n")
            return
        
        # Ejecutar prueba actual
        test_config = self.test_configurations[self.current_test]
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"🧪 PRUEBA {self.current_test + 1}/{len(self.test_configurations)}: {test_config['name']}")
        self.get_logger().info(f"📝 {test_config['description']}")
        self.get_logger().info(f"🎯 Colisión esperada: {'SÍ' if test_config['expected_collision'] else 'NO'}")
        self.get_logger().info(f"{'='*60}")
        
        # Enviar comando
        self.send_trajectory_command(
            test_config['positions'],
            test_config['duration'],
            test_config['name']
        )
        
        # Marcar que estamos esperando
        self.waiting_for_movement = True
        self.test_start_time = time.time()
        self.movement_count += 1
    
    def show_final_report(self):
        """Muestra reporte final de las pruebas"""
        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info("📊 REPORTE FINAL DE PRUEBAS DE AUTO-COLISIÓN")
        self.get_logger().info(f"{'='*80}")
        
        self.get_logger().info(f"📈 Estadísticas generales:")
        self.get_logger().info(f"    - Pruebas ejecutadas: {len(self.test_configurations)}")
        self.get_logger().info(f"    - Movimientos enviados: {self.movement_count}")
        self.get_logger().info(f"    - Colisiones detectadas: {self.collision_count}")
        self.get_logger().info(f"    - Resultados registrados: {len(self.test_results)}")
        
        self.get_logger().info(f"\n📋 Resumen por prueba:")
        for i, test_config in enumerate(self.test_configurations):
            expected = "SÍ" if test_config['expected_collision'] else "NO"
            
            # Buscar si hubo colisión detectada para esta prueba
            detected = False
            for result in self.test_results:
                if result['test_name'] == test_config['name']:
                    detected = True
                    break
            
            detected_str = "SÍ" if detected else "NO"
            status = "✅" if (test_config['expected_collision'] == detected) else "❌"
            
            self.get_logger().info(f"    {status} Prueba {i+1}: {test_config['name']}")
            self.get_logger().info(f"        Esperada: {expected} | Detectada: {detected_str}")
        
        if self.test_results:
            self.get_logger().info(f"\n🔍 Detalles de colisiones detectadas:")
            for result in self.test_results:
                self.get_logger().info(f"    - {result['test_name']} ({result['timestamp']})")
                self.get_logger().info(f"      Mensaje: {result['alert_message']}")
                if result['joint_positions']:
                    pos_str = [f"{pos:.3f}" for pos in result['joint_positions']]
                    self.get_logger().info(f"      Posiciones: {pos_str}")
        
        self.get_logger().info(f"\n💡 Para uso en PPO:")
        self.get_logger().info(f"    - Topic de alertas: /mycobot_280/self_collision_alert")
        self.get_logger().info(f"    - Topic de detalles: /mycobot_280/collision_details")
        self.get_logger().info(f"    - Sensores individuales: /mycobot_280/linkX_contact/contacts")
        self.get_logger().info(f"    - Puedes usar las alertas como recompensa negativa (-1.0)")
        self.get_logger().info(f"    - Los detalles JSON contienen información específica de la colisión")
        
        self.get_logger().info(f"\n🎯 Comandos útiles para debugging:")
        self.get_logger().info(f"    ros2 topic echo /mycobot_280/self_collision_alert")
        self.get_logger().info(f"    ros2 topic echo /mycobot_280/collision_details")
        self.get_logger().info(f"    ros2 topic list | grep contact")
        
        self.get_logger().info(f"{'='*80}")
        self.get_logger().info("✅ Pruebas completadas. Presiona Ctrl+C para salir.")
        
        # Cancelar timer
        self.test_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CollisionMovementTester()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Pruebas interrumpidas por el usuario")
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()