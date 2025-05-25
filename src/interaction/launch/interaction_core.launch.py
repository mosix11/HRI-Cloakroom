import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


from pathlib import Path



def generate_launch_description():
    pkg_share = get_package_share_directory('interaction')
    TTS_action_server_node = Node(
            package='interaction',  
            executable='TTS_action_server', 
            name='TTS_action_server',  
            output='log',  
        )
    
    interaction_node = Node(
            package='interaction',  
            executable='interaction_node',  
            name='interaction_node',  
            output='screen', 
        )
    
    return LaunchDescription([
        TTS_action_server_node,
        interaction_node
    ])