#!/bin/bash
python -m colcon build --packages-select interaction
source install/setup.bash
