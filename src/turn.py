import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
import math

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('joint_trajectory_publisher')

    publisher = node.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)

    joint_trajectory = JointTrajectory()
    joint_trajectory.header = Header()
    joint_trajectory.header.frame_id = 'base_link'
    joint_trajectory.joint_names = ['camera_1_joint']

    duration_between_points = Duration(sec=2, nanosec=0)  # İstenen süreyi burada belirtin

    for angle_deg in range(0, 181, 10):
        angle_rad = math.radians(angle_deg)
        point = JointTrajectoryPoint()
        point.positions = [angle_rad]  # Radyan cinsinden açıyı burada belirtin
        point.time_from_start = duration_between_points

        joint_trajectory.points.append(point)

    node.get_logger().info('Publishing joint trajectory...')

    while rclpy.ok():
        publisher.publish(joint_trajectory)
        node.get_logger().info('Published joint trajectory')
        rclpy.spin_once(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()