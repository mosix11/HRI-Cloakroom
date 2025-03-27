export PYTHONPATH="/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/perception-video/lib/python3.12/site-packages"
colcon build --packages-select perception_video
source install/setup.bash
export PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages"
export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/install/interfaces/lib/python3.12/site-packages"
export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/install/perception_video/lib/python3.12/site-packages"
export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/perception-video/lib/python3.12/site-packages"