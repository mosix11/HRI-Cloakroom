



from whisper_live.client import TranscriptionClient
import os
import time
import subprocess
import torch
from datetime import datetime

def merge_segments(segments):
    if not segments:
        return None

    # Sort by start time (just to be safe)
    segments = sorted(segments, key=lambda x: float(x['start']))

    # Concatenate text with '. ' in between
    combined_text = '. '.join(segment['text'].strip().rstrip('.') for segment in segments)

    # Use start of first and end of last
    start = segments[0]['start']
    end = segments[-1]['end']
    time_added = segments[0]['time_added']

    return {
        'start': start,
        'end': end,
        'text': combined_text,
        'time_added': time_added,
    }

last_processed_index = -1

def my_transcription_handler(transcript):
  global last_processed_index
  
  if len(transcript) - 1 > last_processed_index:
    new_transcript = []
    for idx in range(last_processed_index+1, len(transcript)):
      if (transcript[idx]['text'] == 'Thank you' or transcript[idx]['text'] == 'Thank you.') and (float(transcript[idx]['end']) - float(transcript[idx]['start']) == 2.0 ):
        print('Rejected:', transcript[idx])
        continue
      new_transcript.append(transcript[idx])
      
    if len(new_transcript) > 0:
      print(merge_segments(new_transcript))
    # publish here
    last_processed_index = len(transcript)-1
    
  # print("Final text received:", transcript)

client = TranscriptionClient(
  "localhost",
  9090,
  lang="en",
  translate=False,
  # model="distil-large-v3",
  model="distil-whisper/distil-large-v3.5-ct2",
  use_vad=True,
  max_clients=1,
  max_connection_time=600,
  save_output_recording=False,
  mute_audio_playback=False,
  log_transcription=False,
  on_final_transcription=my_transcription_handler
)

client()

