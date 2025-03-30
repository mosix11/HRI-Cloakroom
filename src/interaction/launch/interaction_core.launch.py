import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import requests
from tqdm import tqdm

from pathlib import Path

def download_file_fast(url, output_path):
    """Downloads a file with a progress bar and handles retries."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=16384,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))

def generate_launch_description():
    pkg_share = get_package_share_directory('interaction')
    
    base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
    interaction_pkg_dir = base_dir / Path('interaction')
    interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
    
    llm_model_url = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q6_K.gguf"
    llm_model_filename = "unsloth_Phi-4-mini-instruct-Q6_K.gguf"
    llm_model_path = interaction_pkg_dir / Path('models') / llm_model_filename
    llm_model_path.parent.mkdir(exist_ok=True, parents=True)
    
    if not llm_model_path.exists():
        download_file_fast(llm_model_url, llm_model_path)
    
    host = '127.0.0.1'
    port = '9585'
    n_gpu_layers = '-1'
    n_ctx = '16384'
    verbose = 'False'
    llama_server_process = ExecuteProcess(
            cmd=['python3', '-m', 'llama_cpp.server', '--host', host, '--port', port, '--model', str(llm_model_path.absolute()), '--n_gpu_layers', n_gpu_layers, '--n_ctx', n_ctx, '--verbose', verbose],
            output='screen',
            name='llama_cpp_server'  # Optional, just for identification in logs
        )
    
    # 'python3 -m llama_cpp.server --host 127.0.0.1 --port 9585 --model ~/.hri_cloakroom/interaction/models/unsloth_Phi-4-mini-instruct-Q6_K.gguf --n_gpu_layers -1 --n_ctx 16384 --verbose False'
    
    TTS_action_server_node = Node(
            package='interaction',  # The name of your ROS 2 package
            executable='TTS_action_server',  # The Python executable for TTS_action_server
            name='TTS_action_server',  # Optional, just for identification in logs
            output='log',  # Show the output in the terminal
        )
    
    interaction_node = Node(
            package='interaction',  # The name of your ROS 2 package
            executable='interaction_node',  # The Python executable for interaction_node
            name='interaction_node',  # Optional, just for identification in logs
            output='screen',  # Show the output in the terminal
        )
    
    
    return LaunchDescription([
        llama_server_process,
        TTS_action_server_node,
        interaction_node
    ])