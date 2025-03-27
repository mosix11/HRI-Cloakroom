#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from dotenv import load_dotenv
from whisper_live.server import TranscriptionServer
load_dotenv()

import os
from pathlib import Path

class ASRServer(Node):
    
    def __init__(self):
        super().__init__('ASR_server_node')
        self.get_logger().info("Starting Vision Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        audio_pkg_dir = base_dir / Path('audio')
        audio_pkg_dir.mkdir(exist_ok=True, parents=True)
        
        
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--port', '-p',
        #                     type=int,
        #                     default=9090,
        #                     help="Websocket port to run the server on.")
        # parser.add_argument('--backend', '-b',
        #                     type=str,
        #                     default='faster_whisper',
        #                     help='Backends from ["tensorrt", "faster_whisper"]')
        # parser.add_argument('--faster_whisper_custom_model_path', '-fw',
        #                     type=str, default=None,
        #                     help="Custom Faster Whisper Model")
        # parser.add_argument('--trt_model_path', '-trt',
        #                     type=str,
        #                     default=None,
        #                     help='Whisper TensorRT model path')
        # parser.add_argument('--trt_multilingual', '-m',
        #                     action="store_true",
        #                     help='Boolean only for TensorRT model. True if multilingual.')
        # parser.add_argument('--omp_num_threads', '-omp',
        #                     type=int,
        #                     default=1,
        #                     help="Number of threads to use for OpenMP")
        # parser.add_argument('--no_single_model', '-nsm',
        #                     action='store_true',
        #                     help='Set this if every connection should instantiate its own model. Only relevant for custom model, passed using -trt or -fw.')
        # args = parser.parse_args()

        # if args.backend == "tensorrt":
        #     if args.trt_model_path is None:
        #         raise ValueError("Please Provide a valid tensorrt model path")

        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(1)

        
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