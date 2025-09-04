#!/bin/bash
mkdir venvs
cd venvs
python3 -m venv perception-audio
python3 -m venv perception-video
python3 -m venv interaction
touch COLCON_IGNORE

source perception-video/bin/activate
pip install rospkg
pip install -U colcon-common-extensions
pip install torch torchvision
pip install opencv-python
pip install onnxruntime-gpu
pip install insightface
pip install python-dotenv tqdm
pip install ultralytics
pip install lmdb
pip install gdown
pip install pygame
deactivate

source perception-audio/bin/activate
pip install rospkg
pip install -U colcon-common-extensions
pip install whisper-live
# pip install torch torchaudio
# pip install tqdm python-dotenv
deactivate

source interaction/bin/activate
pip install rospkg
pip install -U colcon-common-extensions
pip install torch torchaudio
# pip install flash-attn --no-build-isolation
pip install lmdb
pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"
# pip install 'llama-cpp-python[server]' -C cmake.args="-DGGML_CUDA=on" -C force_cmake=1
pip install tqdm soundfile sounddevice python-dotenv
pip install kokoro
pip install opencv-python
pip install pillow
deactivate

