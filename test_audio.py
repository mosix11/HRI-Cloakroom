import pyaudio
import wave

# Initialize PyAudio
p = pyaudio.PyAudio()

# Automatically find the Astro A50 device with the highest number of input channels
astro_device_index = None
astro_channels = 0

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    device_name = info.get("name", "")
    max_channels = info.get("maxInputChannels", 0)
    if "Astro A50" in device_name and max_channels > 0:
        # Choose the device with the most input channels
        if max_channels > astro_channels:
            astro_device_index = i
            astro_channels = max_channels

if astro_device_index is None:
    print("Astro A50 device not found!")
    p.terminate()
    exit()

# Retrieve device info to get its default sample rate
device_info = p.get_device_info_by_index(astro_device_index)
RATE = int(device_info.get("defaultSampleRate", 44100))

print(f"Found Astro A50 device at index {astro_device_index} with {astro_channels} input channel(s), using sample rate: {RATE} Hz.")

# Configuration for recording
FORMAT = pyaudio.paInt16       # 16-bit resolution
FRAMES_PER_BUFFER = 1024
RECORD_SECONDS = 5             # Duration of recording

try:
    # Open the audio stream using the found device and its sample rate
    stream = p.open(format=FORMAT,
                    channels=astro_channels,
                    rate=RATE,
                    input=True,
                    input_device_index=astro_device_index,
                    frames_per_buffer=FRAMES_PER_BUFFER)
except Exception as e:
    print(f"Error opening stream: {e}")
    p.terminate()
    exit()

print("Recording audio from Astro A50...")

frames = []
for _ in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)

print("Recording finished.")

# Stop and close the stream, then terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a WAV file
output_filename = "output.wav"
wf = wave.open(output_filename, 'wb')
wf.setnchannels(astro_channels)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Audio saved to {output_filename}.")
