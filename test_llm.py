import llama_cpp
import langchain
print("Llama-Cpp Version:", llama_cpp.__version__)
print("LangChain Version:", langchain.__version__)

# pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=on"

import os
import requests
from tqdm import tqdm
import time
from llama_cpp import Llama

# import gc
# import torch

# gc.collect()
# torch.cuda.empty_cache()

# Constants
# model_url = "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q6_K_L.gguf"
# model_filename = "microsoft_Phi-4-mini-instruct-Q6_K_L.gguf"
model_url = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q6_K.gguf"
model_filename = "unsloth_Phi-4-mini-instruct-Q6_K.gguf"
model_path = os.path.join("models", model_filename)

# Make sure models directory exists
os.makedirs("models", exist_ok=True)

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

# Download model if not exists
if not os.path.exists(model_path):
    print("Model file not found. Downloading...")
    download_file_fast(model_url, model_path)
    print("Download complete.")

# Load model with llama-cpp
print("Loading model...")
llm = Llama(
    model_path=model_path,   
    n_gpu_layers=-1,
    verbose=False,
    n_ctx=4096
)  # adjust context window if needed


response = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ],
    max_tokens=1024
)

print(response['choices'][0]['message']['content'])

# def busy_wait(seconds):
#     end_time = time.perf_counter() + seconds
#     while time.perf_counter() < end_time:
#         pass  # Keep the CPU busy

# # Example: busy wait for 2 seconds
# busy_wait(10)

# # Run a sample prompt
# output = llm("Q: What is the capital of France?\nA:", max_tokens=32, stop=["Q:", "\n"])
# print(output["choices"][0]["text"].strip())


# pip install huggingface-hub

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
#     filename="*q8_0.gguf",
#     verbose=False
# )
