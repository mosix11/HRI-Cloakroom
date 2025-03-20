import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('simulation_control')
    world_file = os.path.join(pkg_share, 'gazebo', 'my_world.world')
    
    # Launch Gazebo with the custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )
    
    # Spawn the humanoid robot using its URDF file
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'humanoid_robot', '-file', os.path.join(pkg_share, 'urdf', 'humanoid_robot.urdf')],
        output='screen'
    )

    # Spawn the cubby locker
    spawn_locker = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'cubby_locker', '-file', os.path.join(pkg_share, 'urdf', 'cubby_locker.urdf')],
        output='screen'
    )

    # Spawn the coat check stand
    spawn_stand = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'coat_check_stand', '-file', os.path.join(pkg_share, 'urdf', 'coat_check_stand.urdf')],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_robot,
        spawn_locker,
        spawn_stand,
    ])