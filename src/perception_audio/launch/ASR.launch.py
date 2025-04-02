import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import requests
from tqdm import tqdm

from pathlib import Path


def generate_launch_description():
    pkg_share = get_package_share_directory('perception_audio')
    
    ASR_sever_node = Node(
            package='perception_audio',  # The name of your ROS 2 package
            executable='ASR_sever_node',  # The Python executable for TTS_action_server
            name='ASR_sever_node',  # Optional, just for identification in logs
            output='log',  # Show the output in the terminal
        )
    
    ASR_client_node = Node(
            package='perception_audio',  # The name of your ROS 2 package
            executable='ASR_client_node',  # The Python executable for interaction_node
            name='ASR_client_node',  # Optional, just for identification in logs
            output='screen',  # Show the output in the terminal
        )
    
    
    return LaunchDescription([
        ASR_sever_node,
        ASR_client_node
    ])