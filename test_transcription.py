import pyaudio
import numpy as np
import whisper
import threading
import queue
import time

# Configuration parameters
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Transcribe in mono (downmix if necessary)
RATE = 16000              # Whisper models are tuned for 16kHz audio
CHUNK = 1024              # Buffer size (samples per read)
SEGMENT_DURATION = 3      # seconds of audio per segment

# Calculate the number of bytes for a segment
# 2 bytes per sample (16-bit)
segment_byte_length = RATE * SEGMENT_DURATION * 2

# Queue to hold audio chunks
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    # Push raw audio data into the queue
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Initialize PyAudio (you can integrate your device selection logic here)
p = pyaudio.PyAudio()
# Here, you could iterate over devices to choose your Astro A50 device if needed.
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

stream.start_stream()

# Load Whisper model (choose "tiny" or "base" for faster inference)
model = whisper.load_model("tiny.en")  # or "base" for slightly better quality

def transcribe_stream():
    buffer = b""
    while True:
        try:
            # Wait for new audio data from the queue
            data = audio_queue.get(timeout=1)
            buffer += data

            # Once enough data is buffered, process the segment
            if len(buffer) >= segment_byte_length:
                # Convert raw bytes to a numpy array of float32
                audio_np = np.frombuffer(buffer, np.int16).astype(np.float32)
                # Normalize audio to range [-1, 1]
                audio_np = audio_np / 32768.0

                # If you are recording more than one channel, downmix to mono:
                # For example, if your original channels > 1:
                # audio_np = audio_np.reshape(-1, original_channels).mean(axis=1)

                # Transcribe the segment (you can adjust parameters as needed)
                result = model.transcribe(audio_np, fp16=False)
                print("Transcription:", result.get("text", "").strip())

                # Clear the buffer (if you want overlapping segments, consider keeping part of it)
                buffer = b""
        except queue.Empty:
            pass

# Run the transcription in a separate thread so audio capture is uninterrupted
transcription_thread = threading.Thread(target=transcribe_stream, daemon=True)
transcription_thread.start()

try:
    # Main thread can perform other tasks or just wait
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
