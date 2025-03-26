#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# from std_msgs.msg import StringMultiArray
from perception_video.msg import DetectedFace, DetectedFaces 
import cv2
import torch
import numpy as np
import pygame
import pickle
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import lmdb
from collections import OrderedDict
import insightface
import os
from pathlib import Path



load_dotenv()


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("Starting Vision Node")
        
        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        vision_node_dir = base_dir / Path('vision')
        vision_node_dir.mkdir(exist_ok=True, parents=True)
        
        # Create publisher for detected face user IDs
        self.face_pub = self.create_publisher(DetectedFaces, 'detected_faces', 10)
        
        # Detection parameters
        self.detection_FPS = 6  # Run detection every 5 frames
        self.visualization_FPS = 30 # Run visualization for all frames 
        self.current_frame = 0
        
        self.vpfn_timer = self.create_timer(1.0 / self.detection_FPS , self.vision_processing_callback)
        self.vis_timer = self.create_timer(1.0 / self.visualization_FPS, self.visualization_callback)
        

        
        # Initialize your models and database
        self.db_env = self.initialize_db(vision_node_dir)
        self.user_embeddings = self.load_user_embeddings(self.db_env)
        
        # Download YOLO model if needed
        yolov12_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'
        self.yolo_model_path = vision_node_dir / Path('models/yolov12.pt')
        self.yolo_model_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.yolo_model_path.exists():
            self.download_file_fast(yolov12_url, self.yolo_model_path)
        
        # Initialize object detection model (YOLO)
        self.obj_model = YOLO(self.yolo_model_path)
        self.obj_model = self.obj_model.eval().to(torch.device('cuda'))
        
        # Initialize face detection model
        self.face_model = insightface.app.FaceAnalysis(name='buffalo_l', root=str(vision_node_dir))
        self.face_model.prepare(ctx_id=0)  # 0 for GPU
        
        # For storing detection annotations
        self.last_face_annotations = []  # List of tuples: (x1, y1, x2, y2, user_id, gender, talking)
        self.last_obj_annotations = []
        self.face_candidates = []  # For temporal consistency
        self.TEMPORAL_THRESHOLD = 3
        self.CANDIDATE_EXPIRY = 20
        self.DETECTION_SCORE_TSH = 0.75
        
        # Initialize video capture and pygame for display (if needed)
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read from webcam.")
            rclpy.shutdown()
        height, width, _ = frame.shape
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        
    

    def vision_processing_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame.")
            return
        
        # self.current_frame += int(30 / self.detection_FPS)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # --- Object Detection ---
        # Update object detections
        self.last_obj_annotations = []
        objects = self.obj_model(frame, verbose=False)
        for result in objects:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int()):
                label = result.names[cls.item()]
                if label == 'person':
                    continue
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                self.last_obj_annotations.append((x1, y1, x2, y2, label, conf.item()))
        
        # --- Face Detection & Recognition ---
        new_faces = self.face_model.get(frame)
        for face in new_faces:
            if face.det_score < self.DETECTION_SCORE_TSH:
                continue
            new_box = (int(face.bbox[0]), int(face.bbox[1]),
                        int(face.bbox[2]), int(face.bbox[3]))
            new_emb = face.embedding
            
            gender = face.gender
            talking = True

            # Match with existing candidate (using IoU)
            matched_candidate = None
            for candidate in self.face_candidates:
                if self.iou(candidate["bbox"], new_box) >= 0.3:
                    matched_candidate = candidate
                    break
            if matched_candidate:
                old_count = matched_candidate["count"]
                new_count = old_count + 1
                matched_candidate["embedding"] = (matched_candidate["embedding"] * old_count + new_emb) / new_count
                matched_candidate["count"] = new_count
                matched_candidate["bbox"] = new_box
                matched_candidate["last_seen"] = self.current_frame
                matched_candidate["gender"] = gender
                matched_candidate["talking"] = talking
            else:
                self.face_candidates.append({
                    "bbox": new_box,
                    "embedding": new_emb,
                    "count": 1,
                    "recognized_label": None,
                    "last_seen": self.current_frame,
                    "gender": gender,
                    "talking": talking
                })

        # Expire old candidates
        self.face_candidates = [
            cand for cand in self.face_candidates
            if self.current_frame - cand["last_seen"] <= self.CANDIDATE_EXPIRY
        ]
        
        # Process candidates that meet the temporal threshold
        for candidate in self.face_candidates:
            if candidate["count"] >= self.TEMPORAL_THRESHOLD and candidate["recognized_label"] is None:
                user_id, score = self.find_matching_user_vectorized(candidate["embedding"], self.user_embeddings, threshold=0.5)
                if user_id is None:
                    new_user_id = self.save_new_embedding(self.db_env, candidate["embedding"])
                    self.user_embeddings[new_user_id] = {"emb": candidate["embedding"], "count": 1}
                    candidate["recognized_label"] = new_user_id
                    self.get_logger().info(f"New user added: {new_user_id}")
                else:
                    record = self.user_embeddings[user_id]
                    new_count = record["count"] + 1
                    updated_emb = (record["emb"] * record["count"] + candidate["embedding"]) / new_count
                    record["emb"] = updated_emb
                    record["count"] = new_count
                    with self.db_env.begin(write=True) as txn:
                        txn.put(user_id.encode("utf-8"), pickle.dumps(record))
                    self.get_logger().info(f"Updated user {user_id}: count {new_count}, similarity {score:.2f}")
                    candidate["recognized_label"] = user_id

        # Update face annotations to be drawn or published.
        self.last_face_annotations = []
        for candidate in self.face_candidates:
            if candidate["recognized_label"] is not None:
                x1, y1, x2, y2 = candidate["bbox"]
                user_id = candidate["recognized_label"]
                gender = candidate['gender']
                talking = candidate['talking']
                self.last_face_annotations.append((x1, y1, x2, y2, user_id, gender, talking))
        
        # --- Publish the Detected Faces ---
        faces_msg = DetectedFaces()
        for _, _, _, _, user_id, gender, talking in self.last_face_annotations:
            face = DetectedFace()
            face.user_id = user_id
            face.gender = gender   # Default/empty until available
            face.talking = talking  # Default value
            faces_msg.faces.append(face)
        
            # Publish the single message containing all detected faces
        self.face_pub.publish(faces_msg)
        self.get_logger().info(f"Published {len(faces_msg.faces)} faces in one message.")

                
                
    def visualization_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.current_frame += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw object detection annotations (red boxes).
        for (x1, y1, x2, y2, label, conf) in self.last_obj_annotations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw face detection annotations (green boxes with recognized labels).
        for (x1, y1, x2, y2, user_label, gender, talking) in self.last_face_annotations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, user_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(frame, (0, 0))
        pygame.display.flip()
        
        # Handle pygame events to allow for closing the window.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.get_logger().info("Exiting...")
                self.destroy_node()
                rclpy.shutdown()        
    

    def initialize_db(self, dir: Path):
        db_path = dir / Path('db/face_embs.lmdb')
        db_path.parent.mkdir(exist_ok=True, parents=True)
        db_env = lmdb.open(str(db_path.absolute()), map_size=int(1e8)) # 0.1GB max size
        return db_env

    # Helper function: Intersection over Union (IoU) for two bounding boxes.
    def iou(self, boxA, boxB):
        # boxA and boxB are tuples: (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

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

    def get_next_user_id(self, txn, increment_counter=False):
        counter_key = b"__counter__"
        raw = txn.get(counter_key)
        current_id = int(pickle.loads(raw)) if raw else 0
        next_id = current_id + 1
        if increment_counter:
            txn.put(counter_key, pickle.dumps(next_id))
        return f"U_{next_id:04d}"  # e.g., U_0001

    def increment_counter(self, txn):
        counter_key = b"__counter__"
        raw = txn.get(counter_key)
        current_id = int(pickle.loads(raw)) if raw else 0
        next_id = current_id + 1
        txn.put(counter_key, pickle.dumps(next_id))
        
    # Save a new embedding record (with a moving average counter)
    def save_new_embedding(self, env, emb):
        with env.begin(write=True) as txn:
            user_id = self.get_next_user_id(txn, increment_counter=True)
            key = user_id.encode("utf-8")
            # Record contains the embedding and an update count (starting at 1)
            record = {"emb": emb, "count": 1}
            txn.put(key, pickle.dumps(record))
        return user_id

    def load_user_embeddings(self, env):
        with env.begin(write=False) as txn:
            counter_key = b"__counter__"
            raw = txn.get(counter_key)
            current_id = int(pickle.loads(raw)) if raw else 0
            embeddings = OrderedDict()
            for i in range(1, current_id + 1):  # IDs start from 1
                key_str = f"U_{i:04d}"  # e.g., U_0001, U_0002, ...
                key = key_str.encode("utf-8")
                value_bytes = txn.get(key)
                if value_bytes is None:
                    print(f"Warning: Missing data for key {key_str}")
                    continue
                record = pickle.loads(value_bytes)
                print(f"{key_str}: embedding shape {record['emb'].shape}, count {record['count']}")
                embeddings[key_str] = record
        return embeddings

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Match a new embedding against all stored user embeddings.
    def find_matching_user_vectorized(self, new_emb: np.ndarray, user_embs: OrderedDict, threshold: float = 0.5):
        if not user_embs:
            return None, None

        user_ids = list(user_embs.keys())
        emb_matrix = np.stack([record["emb"] for record in user_embs.values()])  # shape: (N, D)

        # Normalize embeddings
        new_emb_norm = new_emb / np.linalg.norm(new_emb)
        emb_matrix_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

        sims = emb_matrix_norm @ new_emb_norm  # cosine similarities
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= threshold:
            return user_ids[best_idx], best_score
        else:
            return None, best_score

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        pygame.quit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
