import rclpy
from rclpy.node import Node

from whisper_live.server import TranscriptionServer


import os
from pathlib import Path

class ASRServer(Node):
    
    def __init__(self):
        super().__init__('ASR_server_node')
        self.get_logger().info("Starting ASR Server Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        audio_pkg_dir = base_dir / Path('audio')
        audio_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        
        server = TranscriptionServer()
        server.run(
            "0.0.0.0",
            port=9090,
            backend='faster_whisper',
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            single_model=True,
        )
        
        
        
        
def main(args=None):
    rclpy.init(args=args)
    node = ASRServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down ASR server.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()