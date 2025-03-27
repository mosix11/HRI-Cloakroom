#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interfaces.msg import ASRTranscript, VisionInfo, DetectedFace, DetectedObject, BoundingBox2D
from interfaces.action import Speak
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String

import os
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class InteractionNode(Node):
    
    def __init__(self):
        super().__init__('interaction_node')
        self.get_logger().info("Starting Interaction Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        interaction_pkg_dir = base_dir / Path('interaction')
        interaction_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        # self.tts_pub = self.create_publisher(String, 'interaction/tts', 10)
        self._TTS_action_client = ActionClient(self, Speak, 'interaction/speak')
        
        self.create_subscription(
            ASRTranscript,
            'audio/transcription',
            self.transcription_callback,
            10
        )

        self.create_subscription(
            VisionInfo,
            'vision/vision_info',
            self.vision_callback,
            10
        )
        
        self.SPEAKING_LOCK = False
        
        

    def transcription_callback(self, msg: ASRTranscript):
        self.get_logger().info(f"Received ASR Transcript: '{msg.transcript}' (datetime: {msg.timestamp})")

    def vision_callback(self, msg: VisionInfo):
        self.get_logger().info(f"Received Vision Info: {len(msg.faces)} Faces and {len(msg.objects)} Objects")
        
        
    def send_TTS_goal(self, text):
        self.SPEAKING_LOCK = True
        action_goal = Speak.Goal()
        action_goal.text = text
        self._TTS_action_client.wait_for_server()
        future = self._TTS_action_client.send_goal_async(action_goal)
        future.add_done_callback()
        
    def TTS_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("TTS goal was rejected.")
            self.SPEAKING_LOCK = False
            return
        
        self.get_logger().info("TTS goal accepted, waiting for result...")
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)
    
    def TTS_get_result_callback(self, future):
        result = future.result().result
        self.SPEAKING_LOCK = False
        if result.success:
            self.get_logger().info("TTS finished playing audio. Resuming processing.")
            # Continue with further actions or processing here.
        else:
            self.get_logger().error("TTS failed to play audio.")
        
        
    def download_file_fast(self, url, output_path):
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
        
        
def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down Interaction node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()