import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from pathlib import Path


def generate_launch_description():
    pkg_share = Path(get_package_share_directory('simulation_control'))
    pkg_gazebo = Path(get_package_share_directory('gazebo_ros'))
    
    worlds_dir = pkg_share / Path('worlds')
    models_dir = pkg_share / Path('models')
    
    world_path = os.path.join(get_package_share_directory('your_package_name'), 'worlds', 'my_nao_world.sdf')

    gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={'world': world_path}.items(),
        )
    return LaunchDescription([
        gazebo,
        # You might want to launch other ROS 2 nodes here, like controllers for the NAO
    ])

# def generate_launch_description():
#     pkg_share = Path(get_package_share_directory('simulation_control'))
#     pkg_gazebo = Path(get_package_share_directory('gazebo_ros'))
    
#     worlds_dir = pkg_share / Path('worlds')
#     models_dir = pkg_share / Path('models')
    
#     # Launch Gazebo with the custom world
#     gazebo = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(pkg_gazebo, 'launch', 'gazebo.launch.py')
#         ),
#         launch_arguments={'world': world_file}.items()
#     )
    
#     # Spawn the humanoid robot using its URDF file
#     spawn_robot = Node(
#         package='gazebo_ros',
#         executable='spawn_entity.py',
#         arguments=['-entity', 'humanoid_robot', '-file', os.path.join(pkg_share, 'urdf', 'humanoid_robot.urdf')],
#         output='screen'
#     )

#     # Spawn the cubby locker
#     spawn_locker = Node(
#         package='gazebo_ros',
#         executable='spawn_entity.py',
#         arguments=['-entity', 'cubby_locker', '-file', os.path.join(pkg_share, 'urdf', 'cubby_locker.urdf')],
#         output='screen'
#     )

#     # Spawn the coat check stand
#     spawn_stand = Node(
#         package='gazebo_ros',
#         executable='spawn_entity.py',
#         arguments=['-entity', 'coat_check_stand', '-file', os.path.join(pkg_share, 'urdf', 'coat_check_stand.urdf')],
#         output='screen'
#     )

#     return LaunchDescription([
#         gazebo,
#         spawn_robot,
#         spawn_locker,
#         spawn_stand,
#     ])