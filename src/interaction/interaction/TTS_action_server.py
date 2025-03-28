#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from interfaces.action import Speak

from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd






import os
from pathlib import Path
from dotenv import load_dotenv

class TTSActionServer(Node):
    
    def __init__(self):
        super().__init__('TTS_action_server')
        self.get_logger().info("Starting Interaction Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        interaction_pkg_dir = base_dir / Path('interaction')
        interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        self.TTS_pipeline = KPipeline(
            lang_code='a',
            device='cuda',
            repo_id='hexgrad/Kokoro-82M'
        )
        
        self._TTS_action_server = ActionServer(
            self,
            Speak,
            'interaction/speak',
            self.speak_callback
        )
        
        
    def speak_callback(self, goal_handle):
        self.get_logger().info('Executing TTS...')
        
        
        generator = self.TTS_pipeline(
            goal_handle.request.text, voice='am_fenrir',  # change voice here
            speed=1, split_pattern=r'\n+'
        )
        for i, (gs, ps, audio) in enumerate(generator):
            # print(i)     # index
            # print(gs)    # graphemes/text
            # print(ps)    # phonemes
            sd.play(audio, samplerate=24000)
            sd.wait()

            
        goal_handle.succeed()
        result = Speak.Result()
        result.success = True
        return result

            
        

        
        
def main(args=None):
    rclpy.init(args=args)
    node = TTSActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down TTS node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        