import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
# from launch.substitutions import PathJoinSubstitution, LaunchConfiguration # Not used currently
# from launch_ros.substitutions import FindPackageShare # Not used currently
# from pathlib import Path # Not used currently

def generate_launch_description():
    pkg_simulation_share = get_package_share_directory('simulation')
    pkg_simulation_control_share = get_package_share_directory('simulation_control') # Add this
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Define the world file path
    world = os.path.join(
        pkg_simulation_share,
        'worlds',
        'cloakroom.world'
    )

    # Set Gazebo resource path (good practice)
    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(pkg_simulation_share, 'models')) # Assuming models are in simulation/models

    # --- Gazebo Simulation ---
    # Include the Gazebo launch file to start gzserver
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # Pass world file and other arguments to gzserver
        # -r: run simulation on start
        # -s: headless mode (no GUI automatically started by server)
        # -v4: verbosity level
        launch_arguments={'gz_args': ['-r -s -v4 ', world], 'on_exit_shutdown': 'true'}.items()
    )

    # Include the Gazebo launch file to start gzclient (GUI)
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # -g: start GUI
        # -v4: verbosity level
        launch_arguments={'gz_args': '-g -v4 '}.items()
    )

    # --- ROS <-> Gazebo Bridge ---
    # Bridge topics needed for controlling the NAO's pose and getting feedback
    # gz_bridge = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     arguments=[
    #         # Command: ROS 2 topic -> Gazebo Topic (gz::msgs::Pose)
    #         '/model/NAO/pose@geometry_msgs/msg/Pose]gz.msgs.Pose',
    #         # Feedback: Gazebo Topic -> ROS 2 Topic (geometry_msgs::msg::Pose)
    #         # Using the model-specific pose topic if available, otherwise use pose/info
    #         '/world/cloakroom/model/NAO/pose@geometry_msgs/msg/Pose[gz.msgs.Pose'
    #         # Alternative if model-specific topic doesn't work:
    #         # '/world/cloakroom/pose/info@sensor_msgs/msg/PoseArray[gz.msgs.Pose_V'
    #     ],
    #     output='screen',
    #     # Remap the ROS 2 topic names if desired (optional)
    #     # remappings=[
    #     #     ('/model/NAO/pose', '/nao/command_pose'),
    #     #     ('/world/cloakroom/model/NAO/pose', '/nao/current_pose')
    #     # ]
    # )

    # --- Your Motion Controller Node ---
    # motion_controller_node = Node(
    #     package='simulation_control',
    #     executable='motion_controller.py', # Make sure this matches your setup.py entry_points
    #     name='motion_controller',         # Optional: Explicitly name the node
    #     output='screen'
    # )

    # --- Launch Description Assembly ---
    ld = LaunchDescription()

    ld.add_action(set_env_vars_resources)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    # ld.add_action(gz_bridge) # Add the bridge
    # ld.add_action(motion_controller_node) # Add your controller node

    return ld