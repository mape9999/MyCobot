import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'  # Frame de referencia
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'my_namespace'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # 💡 Cambia aquí la posición deseada:
        marker.pose.position.x = -0.03595982491970062
        marker.pose.position.y = -0.13671913743019104
        marker.pose.position.z = 0.4176342785358429

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.1  # Diámetro de la esfera
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Rojo
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        self.get_logger().info('Published marker at (-0.03595982491970062, -0.13671913743019104, 0.4176342785358429)')

def main():
    rclpy.init()
    node = MarkerPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
