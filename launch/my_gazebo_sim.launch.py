import launch
from launch_ros.actions import Node


def generate_launch_description():
    return launch.LaunchDescription([
        Node(
            package='my_bot',
            executable='gazebo_sim_node',
            name='gazebo_sim_node'
        )
    ])


if __name__ == '__main__':
    generate_launch_description()