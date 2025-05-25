# export PYTHONPATH="/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/interaction/lib/python3.12/site-packages"
# colcon build --packages-select interaction
# source install/setup.bash
# export PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages"
# export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/install/interfaces/lib/python3.12/site-packages"
# export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/install/interaction/lib/python3.12/site-packages"
# export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/interaction/lib/python3.12/site-packages"

python -m colcon build --packages-select interaction
source install/setup.bash

# export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/perception-audio/lib/python3.12/site-packages"