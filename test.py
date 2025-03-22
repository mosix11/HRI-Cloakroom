import cv2
import urllib.request
import numpy as np
import sounddevice as sd
import requests
import soundfile as sf
import io


base_url = 'http://192.168.1.55:8080/'
vid_url = base_url + 'video'
audio_url = base_url + 'audio.wav'


# cap = cv2.VideoCapture(vid_url)

# while True:
#     ret, frame = cap.read()

#     cv2.imshow("frame", frame)
#     cv2.waitKey(1)
    
    
# Send a GET request to stream audio
response = requests.get(audio_url, stream=True)

# Wrap the response in a file-like object
audio_stream = io.BytesIO()

try:
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            audio_stream.write(chunk)
            audio_stream.seek(0)
            try:
                data, samplerate = sf.read(audio_stream, dtype='float32')
                sd.play(data, samplerate)
            except:
                pass
            audio_stream = io.BytesIO()  # reset buffer


            # Optionally, process the audio in real-time here
            # For example, every N seconds, you can read the buffer
            # and process it using soundfile or numpy
except KeyboardInterrupt:
    print("Stopped.")

# # Inside the same loop:
