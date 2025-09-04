#!/bin/bash
python -m colcon build --packages-select perception_video
source install/setup.bash