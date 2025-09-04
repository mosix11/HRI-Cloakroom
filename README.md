# Implementation of the Interaction Module for Cloakroom Robot (HRI & RBC Course Joint Project)
The project is written in ROS2 Jazzy and is only runnable on Ubuntu version 24.04. In order to run the project, run the following commands in order:

1. Install system-wide packages:
    ```bash
    chmod +x setup_sys.sh
    ./setup_sys.sh
    source ~/.bashrc
    ```
2. Build python virtual environments and install required packages for each module:
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

3. Build packages in order:
    ```bash
    source ./build_interfaces.sh
    source ./build_perception_video_pkg.sh
    source ./build_perception_audio_pkg.sh
    source ./build_interaction_pkg.sh
    ```
4. If you were asked for Hugging Face api token, for downloading models:
    ```bash
    tocuh .env
    echo 'HF_TOKEN="<Your HuggingFace API Token>"' >> .env
    ```

5. Run packages in order and in different terminal windows (make sure you source `install/setup.bash` in each terminal window first):
    ```bash
    ros2 run perception_video vision_node
    ros2 run perception_audio ASR_server_node
    ros2 run perception_audio ASR_client_node
    ros2 launch interaction interaction_core.launch.py
    ```

6. Start interacting with the robot. You can ask question about the museum or give it your items to store in the lockers.

