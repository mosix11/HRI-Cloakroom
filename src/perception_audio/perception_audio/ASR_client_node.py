#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.msg import ASRTranscript
from whisper_live.client import TranscriptionClient
import threading

from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

class ASRClient(Node):
    
    def __init__(self):
        super().__init__('ASR_client_node')
        self.get_logger().info("Starting ASR Client Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        audio_pkg_dir = base_dir / Path('audio')
        audio_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        self.publisher = self.create_publisher(ASRTranscript, 'audio/transcription', 10)
        self.last_processed_index = -1
        
        def my_transcription_handler(transcript):
            # Check if there are new transcripts
            if len(transcript) - 1 > self.last_processed_index:
                new_transcript = []
                for idx in range(self.last_processed_index+1, len(transcript)):
                    if (transcript[idx]['text'] == 'Thank you' or transcript[idx]['text'] == 'Thank you.') and (float(transcript[idx]['end']) - float(transcript[idx]['start']) == 2.0 ):
                        print('Fake transcription rejected:', transcript[idx])
                        continue
                    new_transcript.append(transcript[idx])
                    
                if len(new_transcript) > 0: 
                    merged_transcript = self.merge_segments(new_transcript)
                    # publish here
                    trnsc_msg = ASRTranscript()
                    trnsc_msg.transcript = merged_transcript['text']
                    trnsc_msg.start = float(merged_transcript['start'])
                    trnsc_msg.end = float(merged_transcript['end'])
                    trnsc_msg.timestamp = merged_transcript['time_added']
                    
                    self.publisher.publish(trnsc_msg)
                    self.get_logger().info(f"Published transcript: {merged_transcript} on `audio/transcription` topic.")
                    
                self.last_processed_index = len(transcript) - 1

        # Initialize the TranscriptionClient from whisper-live with the callback
        self.client = TranscriptionClient(
            "localhost",
            9090,
            lang="en",
            translate=False,
            model="distil-large-v3",
            use_vad=True,
            max_clients=1,
            max_connection_time=600,
            save_output_recording=False,
            mute_audio_playback=False,
            log_transcription=False,
            on_final_transcription=my_transcription_handler
        )
        
        self.client()
        
        
    def merge_segments(self, segments):
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
        
def main(args=None):
    rclpy.init(args=args)
    node = ASRClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down ASR client.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()