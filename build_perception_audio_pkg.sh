#!/bin/bash
python -m colcon build --packages-select perception_audio
source install/setup.bash

# export PYTHONPATH="$PYTHONPATH:/home/mosix11/Projects/HRI/HRI-Cloakroom/venvs/perception-audio/lib/python3.12/site-packages"
