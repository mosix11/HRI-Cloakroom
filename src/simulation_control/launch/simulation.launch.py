import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import AppendEnvironmentVariable
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


from pathlib import Path


def generate_launch_description():
    pkg_share = FindPackageShare('simulation_control') 
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    gz_launch_path = PathJoinSubstitution([ros_gz_sim, 'launch', 'gz_sim.launch.py'])

    # pkg_gazebo = Path(get_package_share_directory('gazebo_ros'))
    
    worlds_dir = pkg_share / Path('worlds')
    models_dir = pkg_share / Path('models')
    
    world_path = worlds_dir / Path('nao_world.sdf')
    
    IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': [PathJoinSubstitution([pkg_share, 'worlds/empty_world.sdf'])],  # Replace with your own world file
                'on_exit_shutdown': 'True'
            }.items(),
        ),
    
    # gzserver_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
    #     ),
    #     launch_arguments={'gz_args': ['-r -s -v4 ', str(world_path.absolute())], 'on_exit_shutdown': 'true'}.items()
    # )
    
    # gzclient_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
    #     ),
    #     launch_arguments={'gz_args': '-g -v4 '}.items()
    # )
    
    # set_env_vars_resources = AppendEnvironmentVariable(
    #     'GZ_SIM_RESOURCE_PATH',
    #     str(models_dir)
    # )
    
    # set_env_vars_resources = AppendEnvironmentVariable(
    #     'GZ_SIM_RESOURCE_PATH',
    #     f'{models_dir}:{os.environ.get("GZ_SIM_RESOURCE_PATH", "")}'
    # )


    return LaunchDescription([
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            PathJoinSubstitution([pkg_share, 'models'])
        ),
        SetEnvironmentVariable(
            'GZ_SIM_PLUGIN_PATH',
            PathJoinSubstitution([pkg_share, 'plugins'])
        ),
        
        # gzserver_cmd,
        # gzclient_cmd
        # You might want to launch other ROS 2 nodes here, like controllers for the NAO
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/example_imu_topic@sensor_msgs/msg/Imu@gz.msgs.IMU',],
            remappings=[('/example_imu_topic',
                         '/remapped_imu_topic'),],
            output='screen'
        ),
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