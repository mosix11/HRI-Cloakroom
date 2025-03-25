



from whisper_live.client import TranscriptionClient
import os
import time
import subprocess
import torch






last_processed_index = 0
last_text_segment = ""
unprocessed_text = ""

def sample_callback(text, is_final):
  global client
  global unprocessed_text
  global last_processed_index
  # global last_text_segment
  
  # if last_text_segment != text[-1]:
  #   last_text_segment = text[-1]
    
  if len(text)-2 == last_processed_index:
    unprocessed_text += text[-2]
    # last_text_segment = text[-2]
    # print(last_text_segment)
    last_processed_index += 1
    
  if is_final:
    client.paused = True
    unprocessed_text += text[-1]
    # last_text_segment = text[-1]
    last_processed_index += 1
    print('client paused')
    print(unprocessed_text)
    
    # command = ["python", "kroko_test.py", "--text", unprocessed_text]
    # subprocess.run(command, check=True)
    
    unprocessed_text = ""
    print('back to ASR')
    client.paused = False
    print('client activated')



client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  model="distil-large-v3",
  use_vad=True,
  max_clients=1,
  max_connection_time=600,
  mute_audio_playback=False,
  callback=sample_callback
)

client()
