# from whisper_live.client import TranscriptionClient
# client = TranscriptionClient(
#   "localhost",
#   9090,
#   lang="en",
#   translate=False,
#   model="small",                                      # also support hf_model => `Systran/faster-whisper-small`
#   use_vad=False,
# #   save_output_recording=True,                         # Only used for microphone input, False by Default
# #   output_recording_filename="./output_recording.wav", # Only used for microphone input
#   max_clients=1,
#   max_connection_time=600,
#   mute_audio_playback=False,                          # Only used for file input, False by Default
# )


# client()


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
    print(unprocessed_text)
    
    command = ["python", "kroko_test.py", "--text", unprocessed_text]
    subprocess.run(command, check=True)
    
    unprocessed_text = ""
    print('back to ASR')
    client.paused = False
    print('client activated')


  # if is_final and text != last_text:
  #   print(unprocessed_text)
  #   last_text = text
  #   client.paused = True
  #   # # Define the command to be run
  #   # command = f'echo "{text[-1]}" | piper --model en_US-lessac-medium --output-raw | aplay -r 22050 -f S16_LE -t raw -'
  #   # # Run the command
  #   # subprocess.run(command, shell=True, check=True)
    
    
  #   # generator = pipeline(
  #   #   text[-1], voice='am_fenrir', # <= change voice here
  #   #   speed=1, split_pattern=r'\n+'
  #   #   )

  #   # print(generator)
  #   # for i, (gs, ps, audio) in enumerate(generator):
  #   #     print(i)  # i => index
  #   #     print(gs) # gs => graphemes/text
  #   #     print(ps) # ps => phonemes
  #   #     print('here')
  #   #     sd.play(audio, samplerate=24000)
  #   #     print('here2')
  #   #     sd.wait()
  #   #     print('here3')
  #   #     sf.write(f'{i}.wav', audio, 24000) # save each audio file


  #   client.paused = False
  #   unprocessed_text = ""
  # else:
  #   if text != last_text:
  #     unprocessed_text += text[-1]
  #     text = last_text


# last_added_segment = ""
# unprocessed_text = ""
# def sample_callback(text, is_final):
#   global client
#   global unprocessed_text
#   global last_added_segment
  
#   last_segment = text[-1]
#   print(text)
  
#   if last_added_segment != last_segment:
#     unprocessed_text += last_segment
#     last_added_segment = last_segment
#     # print(unprocessed_text)

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
