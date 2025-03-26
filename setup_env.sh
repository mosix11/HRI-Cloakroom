#!/bin/bash
mkdir venvs
cd venvs
python3 -m venv --system-site-packages perception-audio
python3 -m venv --system-site-packages perception-video
python3 -m venv --system-site-packages interaction

source perception-video/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install opencv-python
pip install onnxruntime-gpu
pip install insightface
pip install python-dotenv tqdm
pip install ultralytics
deactivate

source perception-audio/bin/activate
pip install torch torchaudio
pip install tqdm python-dotenv
git clone git@github.com:mosix11/WhisperLive-Text.git
cd WhisperLive-Text
pip install -e .
# pip install .
cd ..
# rm -rf WhisperLive-Text
deactivate

source interaction/bin/activate
pip install torch torchaudio
pip install flash-attn --no-build-isolation
pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"
pip install tqdm soundfile sounddevice python-dotenv
pip install kokoro
deactivate

