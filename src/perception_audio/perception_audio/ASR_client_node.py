import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from interfaces.msg import ASRTranscript
from whisper_live.client import TranscriptionClient
import time
import threading
import datetime
import collections


import os
from pathlib import Path

class ASRClient(Node):
    
    def __init__(self):
        super().__init__('ASR_client_node')
        self.get_logger().info("Starting ASR Client Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        audio_pkg_dir = base_dir / Path('audio')
        audio_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE # Text doesn't need to persist for late-joining subscribers
        )
        self.publisher = self.create_publisher(ASRTranscript, 'audio/transcription', qos_profile)
        
        

        
        self._GIBBERISH_WORDS = {
            "you", "you.", "thank you" ,"thank you.", "thanks", "thanks.", "bye", "bye.", "uh", "um", "huh", "hmm", "oh"
        }
        self._FLUSH_INTERVAL = 1.0  # Seconds to wait after last VAD activity/callback before flushing
        
        # --- State Variables ---
        self._last_sent_full_clean_transcript = ""
        self._transcript_buffer = collections.deque() # Using deque for efficient appends/pops
        self._buffer_flush_timer = None
        self._client_thread = None # To run whisper-live client in a separate thread
        self._running = False # Flag to control threads on shutdown

        
        # --- Threading Locks ---
        self._buffer_lock = threading.Lock() # Protects _transcript_buffer and _last_sent_full_clean_transcript
        self._timer_lock = threading.Lock() # Protects _buffer_flush_timer

  
        # Initialize the TranscriptionClient from whisper-live with the callback
        self._client = TranscriptionClient(
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
            transcription_callback=self._transcription_callback
        )
        
        self._running = True
        self._client_thread = threading.Thread(target=self._run_whisper_client)
        self._client_thread.daemon = True # Allow main program to exit even if thread is still running
        self._client_thread.start()
        self.get_logger().info("Whisper Live Client thread started.")
        
        # Start the initial buffer flush timer
        self._reset_flush_timer()
        self.get_logger().info(f"Initial buffer flush timer set for {self._FLUSH_INTERVAL} seconds.")
        

        
    def _run_whisper_client(self):
        """
        Runs the whisper-live client in its own thread.
        """
        self.get_logger().info("Whisper Live Client running...")
        try:
            self._client()
        except Exception as e:
            self.get_logger().error(f"Whisper Live Client encountered an error: {e}")
        finally:
            self.get_logger().info("Whisper Live Client thread finished.")
            
    
    def _flush_buffer_to_llm(self):
        """
        Called when the timer expires to send the buffered text to the LLM.
        """
        # Acquire lock before accessing shared buffer variables
        with self._buffer_lock:
            if not self._transcript_buffer:
                self.get_logger().debug("Buffer is empty, nothing to flush.")
                return

            text_to_send_list = list(self._transcript_buffer)
            self._transcript_buffer.clear() # Clear buffer immediately under lock

            text_to_send = " ".join(text_to_send_list).strip()
            
            if not text_to_send:
                return # Nothing to send after stripping

            final_text_chunk_for_llm = ""
            if not self._last_sent_full_clean_transcript:
                final_text_chunk_for_llm = text_to_send
            else:
                s1 = self._last_sent_full_clean_transcript
                s2 = text_to_send

                best_overlap_length = 0
                for N in range(min(len(s1), len(s2)), 0, -1):
                    suffix_of_s1 = s1[len(s1)-N:]
                    prefix_of_s2 = s2[:N]
                    if suffix_of_s1 == prefix_of_s2:
                        best_overlap_length = N
                        break
                
                if best_overlap_length > 0:
                    final_text_chunk_for_llm = s2[best_overlap_length:].strip()
                else:
                    final_text_chunk_for_llm = text_to_send
            
            if final_text_chunk_for_llm:
                self._publish_ASR_transcript(final_text_chunk_for_llm)
                # Only update last sent transcript if something was actually sent
                self._last_sent_full_clean_transcript = text_to_send
            else:
                self.get_logger().debug(f"No new text to send after diffing. Current: '{text_to_send}', Last: '{self._last_sent_full_clean_transcript}'")
        
    def _reset_flush_timer(self):
        """
        Resets or starts the timer to flush the buffer.
        """
        with self._timer_lock: # Protect access to the timer object
            if self._buffer_flush_timer and self._buffer_flush_timer.is_alive():
                self._buffer_flush_timer.cancel()
                self.get_logger().debug("Cancelled existing flush timer.")

            # Only start a new timer if the node is still running
            if self._running:
                self._buffer_flush_timer = threading.Timer(self._FLUSH_INTERVAL, self._flush_buffer_to_llm)
                self._buffer_flush_timer.daemon = True # Allow app to exit if timer is running
                self._buffer_flush_timer.start()
                self.get_logger().debug(f"New flush timer set for {self._FLUSH_INTERVAL} seconds.")
                
                
    def _transcription_callback(self, full_text_from_library, segments):
        """
        Callback from whisper-live client, running in its own thread.
        """
        # Always reset the timer on callback, indicating activity
        self._reset_flush_timer()

        has_new_completed_segments = False
        current_completed_segment_infos = []

        if segments:
            for seg_idx, seg in enumerate(segments):
                if seg.get('completed', False):
                    text_content = seg['text'].strip()
                    # Add to buffer only if it's new and not gibberish (final check for gibberish at the end)
                    if text_content and text_content not in self._transcript_buffer:
                        current_completed_segment_infos.append({
                            'text': text_content,
                            'is_last_in_stream': seg_idx == len(segments) - 1,
                        })
                        has_new_completed_segments = True

        if not has_new_completed_segments:
            return

        texts_to_consider = [info['text'] for info in current_completed_segment_infos]
        valid_text_count = len(texts_to_consider)

        # Filter trailing gibberish
        for i in range(len(texts_to_consider) - 1, -1, -1):
            original_segment_info = current_completed_segment_infos[i]
            text_content = original_segment_info['text'].strip()
            words_in_segment = text_content.split()
            is_potentially_gibberish = (
                len(words_in_segment) <= 2 and
                text_content.lower() in self._GIBBERISH_WORDS
            )
            if (is_potentially_gibberish and
                original_segment_info['is_last_in_stream'] and
                i == valid_text_count - 1):
                self.get_logger().debug(f"Filtering trailing gibberish segment: '{text_content}'")
                valid_text_count -= 1
            else:
                break
                
        current_clean_texts_list = texts_to_consider[:valid_text_count]
        
        # Acquire lock before modifying the buffer
        with self._buffer_lock:
            for seg_text in current_clean_texts_list:
                stripped_text = seg_text.strip()
                if stripped_text and stripped_text not in self._transcript_buffer:
                    self._transcript_buffer.append(stripped_text)
                    self.get_logger().debug(f"Added to buffer: '{stripped_text}'")
        
    def _publish_ASR_transcript(self, transcript):
        trnsc_msg = ASRTranscript()
        trnsc_msg.transcript = transcript
        current_ros_time = self.get_clock().now()
        seconds_since_epoch = current_ros_time.nanoseconds / 1e9
        dt_object_utc = datetime.datetime.fromtimestamp(seconds_since_epoch, tz=datetime.timezone.utc)
        timestamp_str_custom_utc = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S.%f UTC')
        trnsc_msg.timestamp = timestamp_str_custom_utc
        
        self.publisher.publish(trnsc_msg)
        self.get_logger().info(f"Published transcript: {transcript} with time {timestamp_str_custom_utc} on `audio/transcription` topic.")




    def destroy_node(self):
        """
        Clean up resources when the node is destroyed.
        """
        self.get_logger().info("Shutting down ASR_LLM_Node...")
        self._running = False # Signal threads to stop

        # Cancel any pending timer
        with self._timer_lock:
            if self._buffer_flush_timer and self._buffer_flush_timer.is_alive():
                self._buffer_flush_timer.cancel()
                self.get_logger().info("Cancelled buffer flush timer.")

        # Attempt to gracefully stop the whisper-live client (if it has a stop method)
        # The whisper_live.client.TranscriptionClient doesn't seem to have a direct stop() method.
        # It relies on the client() call to finish, which happens when the connection closes or an error occurs.
        # Setting _running = False is the main way to signal your logic to stop.
        # If the client is stuck blocking, you might need to implement a more forceful shutdown
        # for whisper-live or rely on the daemon thread exiting with the process.

        # Flush any remaining buffer content before exiting
        self._flush_buffer_to_llm()
        
        # Wait for the client thread to finish (optional, but good for clean shutdown)
        if self._client_thread and self._client_thread.is_alive():
            self.get_logger().info("Waiting for Whisper Live client thread to finish...")
            self._client_thread.join(timeout=5) # Wait up to 5 seconds
            if self._client_thread.is_alive():
                self.get_logger().warn("Whisper Live client thread did not terminate cleanly.")
        
        self.get_logger().info("ASR_LLM_Node shutdown complete.")
        super().destroy_node()

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