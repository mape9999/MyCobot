#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import json
from datetime import datetime

class SimpleCollisionDetector(Node):
    def __init__(self):
        super().__init__('simple_collision_detector')
        
        # Parámetros
        self.declare_parameter('robot_name', 'mycobot_280')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        
        # Suscriptor para estados de articulaciones
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publishers para alertas de colisión
        self.collision_pub = self.create_publisher(
            String, 
            f'/{self.robot_name}/self_collision_alert', 
            10
        )
        
        self.collision_details_pub = self.create_publisher(
            String,
            f'/{self.robot_name}/collision_details',
            10
        )
        
        # Variables de estado
        self.collision_count = 0
        self.last_collision_time = 0
        
        # Rangos de articulaciones que causan colisiones (muy simples)
        self.dangerous_ranges = {
            'link2_to_link3': [(1.2, 2.5), (-2.5, -1.2)],
            'link3_to_link4': [(-2.5, -1.2), (1.2, 2.5)],
            'link4_to_link5': [(1.2, 2.5), (-2.5, -1.2)]
        }
        
        self.get_logger().info(f"🤖 Simple collision detector initialized for {self.robot_name}")
        self.get_logger().info("🔍 Monitoring joint positions for dangerous configurations")
    
    def joint_state_callback(self, msg):
        """Callback para estados de articulaciones"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Evitar spam de detecciones
        if current_time - self.last_collision_time < 2.0:
            return
        
        # Buscar articulaciones peligrosas
        dangerous_joints = []
        
        for i, (joint_name, position) in enumerate(zip(msg.name, msg.position)):
            if joint_name in self.dangerous_ranges:
                ranges = self.dangerous_ranges[joint_name]
                for range_min, range_max in ranges:
                    if range_min <= position <= range_max:
                        dangerous_joints.append(f"{joint_name}({position:.3f})")
                        break
        
        # Si encontramos articulaciones peligrosas, reportar colisión
        if dangerous_joints:
            self.report_collision(dangerous_joints, msg.position)
            self.last_collision_time = current_time
    
    def report_collision(self, dangerous_joints, all_positions):
        """Reporta una colisión detectada"""
        self.collision_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        collision_description = f"Dangerous joint positions: {', '.join(dangerous_joints)}"
        
        # Publicar alerta simple
        alert_msg = String()
        alert_msg.data = f"COLLISION DETECTED: {collision_description}"
        self.collision_pub.publish(alert_msg)
        
        # Publicar detalles
        details_msg = String()
        details_msg.data = json.dumps({
            "type": "simple_geometric_collision",
            "dangerous_joints": dangerous_joints,
            "all_positions": list(all_positions),
            "timestamp": timestamp,
            "count": self.collision_count,
            "description": collision_description
        })
        self.collision_details_pub.publish(details_msg)
        
        # Log
        self.get_logger().warn(
            f"🚨 COLLISION #{self.collision_count}: {collision_description}"
        )
        self.get_logger().info(
            f"📐 All positions: {[f'{p:.3f}' for p in all_positions]}"
        )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimpleCollisionDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in simple collision detector: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()