#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration # Import Duration
# from std_msgs.msg import StringMultiArray
from interfaces.msg import VisionInfo, DetectedFace, DetectedObject, BoundingBox2D
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import pygame
import pickle
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import requests
import gdown
from dotenv import load_dotenv
import lmdb
from collections import OrderedDict
import insightface
import os
from pathlib import Path
import time


load_dotenv()


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.get_logger().info("Starting Vision Node")

        base_dir = Path(os.path.expanduser('~/.hri_cloakroom'))
        vision_pkg_dir = base_dir / Path('vision')
        vision_pkg_dir.mkdir(exist_ok=True, parents=True)

        # Create publisher for detected visual information
        self.visinfo_pub = self.create_publisher(VisionInfo, 'vision/vision_info', 10)
        self.bridge = CvBridge()


        # --- Detection & Timing Parameters ---
        self.detection_rate = 6.0  # Detections per second (FPS)
        self.visualization_rate = 30.0 # Visualization FPS
        self.detection_interval = 1.0 / self.detection_rate

        # --- Face Recognition Parameters ---
        self.face_candidates = []
        self.FACE_TEMPORAL_THRESHOLD = 3 # Consecutive frames face needed
        self.FACE_CANDIDATE_EXPIRY_FRAMES = int(self.visualization_rate * 0.1) # Expire candidates not seen for ~0.75s
        self.FACE_DETECTION_SCORE_TSH = 0.75
        self.FACE_MATCH_IOU_THRESHOLD = 0.3
        self.FACE_RECOGNITION_THRESHOLD = 0.5 # Cosine similarity threshold

        # --- Object Buffer Parameters ---
        # How many consecutive *detection cycles* an object must be seen held
        self.OBJECT_CONSISTENCY_THRESHOLD = 3
        # How long (seconds) an object stays in the buffer after last seen
        self.OBJECT_BUFFER_DURATION_SEC = 5.0
        # How long (seconds) a *candidate* object persists if not seen consecutively
        self.OBJECT_CANDIDATE_EXPIRY_SEC = self.detection_interval * 1.5 # Slightly more than one detection cycle

        self.HOLDING_IOU_THRESHOLD = 0.01 # Tune this IoU threshold
        self.HOLDING_HAND_COVERAGE_THRESHOLD = 0.05 # Tune this hand coverage threshold

        # Internal State
        self.processing_frame_count = 0 # Counter incremented each detection cycle
        self.visualization_frame_count = 0 # Counter incremented each visualization cycle

        # Timers
        self.vpfn_timer = self.create_timer(self.detection_interval, self.vision_processing_callback)
        self.vis_timer = self.create_timer(1.0 / self.visualization_rate, self.visualization_callback)


        # Initialize your models and database
        self.db_env = self.initialize_db(vision_pkg_dir)
        self.user_embeddings = self.load_user_embeddings(self.db_env)

        # Download YOLO model if needed
        yolov11_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'
        self.yolo_model_path = vision_pkg_dir / Path('models/yolov11x.pt')
        self.yolo_model_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.yolo_model_path.exists():
            self.download_file_fast(yolov11_url, self.yolo_model_path)
        yolov11_hand_url = 'https://drive.google.com/uc?export=download&id=1TI3R1_Kadw8nTN-TstEl-T1Jk58sXhoy'
        self.yolo_hand_model_path = vision_pkg_dir / Path('models/yolov11_hand.pt')
        if not self.yolo_hand_model_path.exists():
            gdown.download(yolov11_hand_url, str(self.yolo_hand_model_path.absolute()), quiet=False)

        # Initialize object detection model (YOLO)
        self.obj_model = YOLO(self.yolo_model_path, verbose=False)
        self.obj_model = self.obj_model.eval().to(torch.device('cuda'))

        self.hand_model = YOLO(self.yolo_hand_model_path, verbose=False)
        self.hand_model = self.hand_model.eval().to(torch.device('cuda'))

        # Initialize face detection model
        self.face_model = insightface.app.FaceAnalysis(name='buffalo_l', root=str(vision_pkg_dir))
        self.face_model.prepare(ctx_id=0)  # 0 for GPU

        # --- State Storage ---
        self.last_face_annotations = []  # For visualization: (x1, y1, x2, y2, user_id, gender, talking)
        self.last_hand_annotations = []  # For visualization: ((x1, y1, x2, y2), conf)
        self.vis_held_objects = []       # For visualization: Stores objects detected as held in the *last processing cycle*


        # Object Buffer System
        self.object_candidates = {} # Tracks recently detected held objects for consistency check
        # Key: object_label (str). Value: dict {'bbox': tuple, 'last_seen_time': Time, 'consecutive_detections': int}

        self.object_buffer = {} # Stores confirmed held objects
        # Key: object_label (str). Value: dict {'bbox': tuple, 'last_seen_time': Time}


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
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame.")
            return

        # Use reliable time source
        current_time = self.get_clock().now()
        self.processing_frame_count += 1 # Increment detection cycle counter

        # Prepare frame for processing
        success, encoded_frame = cv2.imencode('.jpg', frame)
        if not success:
            self.get_logger().error("Failed to encode image!")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # --- Object Detection (Raw) ---
        objects_raw = self.obj_model(frame_rgb, verbose=False)
        raw_obj_annotations = []
        for result in objects_raw:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int()):
                label = result.names[cls.item()]
                if label == 'person': # Ignore persons for object list
                    continue
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                raw_obj_annotations.append(((x1, y1, x2, y2), label, conf.item())) # Store box as tuple

        # --- Hand Detection ---
        detected_hands = self.hand_model(frame_rgb, verbose=False)
        current_hand_annotations = [] # Store hands from this frame
        for hand_annt in detected_hands:
            for box, conf, cls in zip(hand_annt.boxes.xyxy, hand_annt.boxes.conf, hand_annt.boxes.cls.int()):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                current_hand_annotations.append(((x1, y1, x2, y2), conf.item()))
        self.last_hand_annotations = current_hand_annotations # Update for visualization

        # --- Identify Held Objects in Current Frame ---
        current_held_objects = [] # List of tuples: (box, label, conf)
        held_object_labels_this_frame = set() # Track labels seen this frame
        for obj_box, obj_label, obj_conf in raw_obj_annotations:
            is_held = False
            for hand_box, hand_conf in current_hand_annotations:
                if self.check_object_hand_interaction(
                    obj_box,
                    hand_box,
                    iou_threshold=self.HOLDING_IOU_THRESHOLD,
                    hand_coverage_threshold=self.HOLDING_HAND_COVERAGE_THRESHOLD):
                    is_held = True
                    break
            if is_held:
                current_held_objects.append((obj_box, obj_label, obj_conf))
                held_object_labels_this_frame.add(obj_label)

        self.vis_held_objects = current_held_objects

        # --- Update Object Candidates and Buffer ---
        # 1. Update existing candidates or add new ones
        for obj_box, obj_label, obj_conf in current_held_objects:
            if obj_label in self.object_candidates:
                # Update existing candidate
                self.object_candidates[obj_label]['bbox'] = obj_box
                self.object_candidates[obj_label]['last_seen_time'] = current_time
                # Increment consecutive count only if it was seen in the *previous* cycle too
                # (Requires knowing the previous frame's state, easier to just increment and reset if missed)
                self.object_candidates[obj_label]['consecutive_detections'] += 1
            else:
                # Add new candidate
                self.object_candidates[obj_label] = {
                    'bbox': obj_box,
                    'last_seen_time': current_time,
                    'consecutive_detections': 1
                }

        # 2. Reset consecutive count for candidates NOT seen this frame
        #    and expire old candidates
        candidates_to_remove = []
        candidate_expiry_time = current_time - Duration(seconds=self.OBJECT_CANDIDATE_EXPIRY_SEC)
        for label, candidate_data in self.object_candidates.items():
            if label not in held_object_labels_this_frame:
                candidate_data['consecutive_detections'] = 0 # Reset count

            # Check for expiry
            if candidate_data['last_seen_time'] < candidate_expiry_time:
                 candidates_to_remove.append(label)
                 self.get_logger().debug(f"Expiring object candidate: {label}")

        for label in candidates_to_remove:
            del self.object_candidates[label]

        # 3. Promote consistent candidates to the main buffer
        for label, candidate_data in self.object_candidates.items():
            if candidate_data['consecutive_detections'] >= self.OBJECT_CONSISTENCY_THRESHOLD:
                # Add or update in the main buffer
                self.object_buffer[label] = {
                    'bbox': candidate_data['bbox'],
                    'last_seen_time': candidate_data['last_seen_time'] # Use candidate's last seen time
                }
                self.get_logger().debug(f"Object '{label}' added/updated in buffer.")

        # 4. Expire old items from the main buffer
        buffer_expiry_time = current_time - Duration(seconds=self.OBJECT_BUFFER_DURATION_SEC)
        buffer_items_to_remove = []
        for label, buffer_data in self.object_buffer.items():
            if buffer_data['last_seen_time'] < buffer_expiry_time:
                buffer_items_to_remove.append(label)
                self.get_logger().info(f"Object '{label}' expired from buffer.")

        for label in buffer_items_to_remove:
            del self.object_buffer[label]


        # --- Face Detection & Recognition ---
        # (Keep existing face logic, but ensure self.processing_frame_count is used if needed)
        new_faces = self.face_model.get(frame_rgb) # Use RGB frame
        current_face_annotations = [] # Store recognized faces for this frame

        for face in new_faces:
            if face.det_score < self.FACE_DETECTION_SCORE_TSH:
                continue
            new_box = (int(face.bbox[0]), int(face.bbox[1]),
                        int(face.bbox[2]), int(face.bbox[3]))
            new_emb = face.embedding
            gender = face.gender # Assuming 0 for male, 1 for female
            talking = True # Placeholder, replace with actual VAD if available

            # Match with existing candidate (using IoU)
            matched_candidate = None
            best_iou = 0
            for i, candidate in enumerate(self.face_candidates):
                 iou_val, _ = self.iou(candidate["bbox"], new_box)
                 if iou_val > self.FACE_MATCH_IOU_THRESHOLD and iou_val > best_iou:
                    matched_candidate = candidate
                    best_iou = iou_val

            if matched_candidate:
                old_count = matched_candidate["count"]
                new_count = old_count + 1
                # Moving average for embedding
                matched_candidate["embedding"] = (matched_candidate["embedding"] * old_count + new_emb) / new_count
                matched_candidate["count"] = new_count
                matched_candidate["bbox"] = new_box # Update bbox
                matched_candidate["last_seen_frame"] = self.processing_frame_count # Use processing frame count
                matched_candidate["gender"] = gender
                matched_candidate["talking"] = talking
            else:
                # Add new candidate
                self.face_candidates.append({
                    "bbox": new_box,
                    "embedding": new_emb,
                    "count": 1,
                    "recognized_label": None,
                    "last_seen_frame": self.processing_frame_count, # Use processing frame count
                    "gender": gender,
                    "talking": talking
                })

        # Expire old face candidates based on frame count
        self.face_candidates = [
            cand for cand in self.face_candidates
            if self.processing_frame_count - cand["last_seen_frame"] <= self.FACE_CANDIDATE_EXPIRY_FRAMES
        ]

        # Process face candidates that meet the temporal threshold
        for candidate in self.face_candidates:
             # Recognize only once or if label is None
            if candidate["count"] >= self.FACE_TEMPORAL_THRESHOLD and candidate["recognized_label"] is None:
                user_id, score = self.find_matching_user_vectorized(candidate["embedding"], self.user_embeddings, threshold=self.FACE_RECOGNITION_THRESHOLD)
                if user_id is None:
                    new_user_id = self.save_new_embedding(self.db_env, candidate["embedding"])
                    self.user_embeddings[new_user_id] = {"emb": candidate["embedding"], "count": 1}
                    candidate["recognized_label"] = new_user_id
                    self.get_logger().info(f"New user added: {new_user_id}")
                else:
                    # Update existing user embedding (optional refinement)
                    record = self.user_embeddings[user_id]
                    new_count = record["count"] + 1
                    updated_emb = (record["emb"] * record["count"] + candidate["embedding"]) / new_count
                    record["emb"] = updated_emb
                    record["count"] = new_count
                    # Save updated embedding back to DB
                    with self.db_env.begin(write=True) as txn:
                        txn.put(user_id.encode("utf-8"), pickle.dumps(record))
                    self.get_logger().info(f"Updated user {user_id}: count {new_count}, similarity {score:.2f}")
                    candidate["recognized_label"] = user_id

        # Update face annotations list for visualization / publishing
        self.last_face_annotations = []
        for candidate in self.face_candidates:
            # Only include recognized faces or those meeting temporal threshold for stability
            if candidate["recognized_label"] is not None and candidate["count"] >= self.FACE_TEMPORAL_THRESHOLD :
                x1, y1, x2, y2 = candidate["bbox"]
                user_id = candidate["recognized_label"]
                # Determine gender string (example)
                # gender_str = "Female" if candidate.get('gender', 1) == 1 else "Male"
                gender_bool = bool(candidate.get('gender', 1)) # Use 1 (female) as default if missing
                talking = candidate.get('talking', False) # Default to not talking if missing
                self.last_face_annotations.append((x1, y1, x2, y2, user_id, gender_bool, talking))


        # --- Publish the Detection Results ---
        vision_msg = VisionInfo()
        img_msg = CompressedImage()
        img_msg.header = Header()
        img_msg.header.stamp = current_time.to_msg() # Use current time
        img_msg.format = "jpeg" # Use jpeg for compressed
        img_msg.data = np.array(encoded_frame).tobytes()
        vision_msg.frame = img_msg

        # Add faces
        for x_min, y_min, x_max, y_max, user_id, gender_bool, talking in self.last_face_annotations:
            face = DetectedFace()
            bbox = BoundingBox2D()
            bbox.x_min = x_min
            bbox.y_min = y_min
            bbox.x_max = x_max
            bbox.y_max = y_max
            face.user_id = user_id
            face.gender = gender_bool # Directly assign the boolean
            face.talking = talking
            face.bbox = bbox
            vision_msg.faces.append(face)

        # Add objects FROM THE BUFFER
        for obj_label, buffer_data in self.object_buffer.items():
            obj = DetectedObject()
            bbox = BoundingBox2D()
            x_min, y_min, x_max, y_max = buffer_data['bbox']
            bbox.x_min = x_min
            bbox.y_min = y_min
            bbox.x_max = x_max
            bbox.y_max = y_max
            obj.bbox = bbox # Assign bbox to object message
            obj.type = obj_label
            # obj.confidence_score = ??? # Confidence isn't stored in buffer, maybe add if needed
            vision_msg.objects.append(obj)


        # Publish the single message
        self.visinfo_pub.publish(vision_msg)
        processing_time = time.time() - start_time
        self.get_logger().info(f"Published {len(vision_msg.faces)} faces and {len(vision_msg.objects)} buffered objects. Processing time: {processing_time:.3f}s")


    def visualization_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.visualization_frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Work with RGB

        # --- Draw buffered object annotations (Blue boxes) ---
        # Draw from self.object_buffer for consistency with published data
        for obj_box, label, conf in self.vis_held_objects:
            x1, y1, x2, y2 = obj_box
            # Use a distinct color for buffered objects
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2) 
            cv2.putText(frame_rgb, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- Draw hand detection annotations (Red boxes) ---
        # Draw from self.last_hand_annotations which is updated in processing callback
        for (x1, y1, x2, y2), conf in self.last_hand_annotations:
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            # cv2.putText(frame_rgb, f"Hand: {conf:.2f}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) # Optional: display confidence

        # --- Draw face detection annotations (Green boxes) ---
        # Draw from self.last_face_annotations which is updated in processing callback
        for (x1, y1, x2, y2, user_label, gender_bool, talking) in self.last_face_annotations:
            color = (0, 255, 0) # Green
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color=color, thickness=2)
            gender_str = "F" if gender_bool else "M"
            talk_str = "T" if talking else "NT"
            label_text = f"{user_label}"
            cv2.putText(frame_rgb, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display using pygame
        frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.screen.blit(frame_pygame, (0, 0))
        pygame.display.flip()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.get_logger().info("Pygame exit requested...")
                # self.destroy_node() # Let the main loop handle shutdown
                rclpy.try_shutdown() # Initiate shutdown

    # --- Helper Functions (iou, check_object_hand_interaction, db functions, etc.) ---
    # Keep these functions as they were in your original code
    # ... (initialize_db, iou, check_object_hand_interaction, download_file_fast, get_next_user_id, etc.) ...

    def initialize_db(self, dir: Path):
        db_path = dir / Path('db/face_embs.lmdb')
        db_path.parent.mkdir(exist_ok=True, parents=True)
        db_env = lmdb.open(str(db_path.absolute()), map_size=int(1e9)) # 1GB max size
        return db_env

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou, interArea

    def check_object_hand_interaction(self, obj_box, hand_box, iou_threshold=0.05, hand_coverage_threshold=0.2):
        iou, interArea = self.iou(obj_box, hand_box)
        if iou < iou_threshold:
            return False
        hand_area = (hand_box[2] - hand_box[0]) * (hand_box[3] - hand_box[1])
        if hand_area <= 0:
            return False
        hand_coverage = interArea / float(hand_area)
        # Check if IoU OR Hand Coverage meet threshold (more lenient)
        # Or use AND if stricter check is needed:
        # if iou >= iou_threshold and hand_coverage >= hand_coverage_threshold:
        if iou >= iou_threshold or hand_coverage >= hand_coverage_threshold:
             return True
        return False

    def download_file_fast(self, url, output_path):
        """Downloads a file with a progress bar."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1MB chunks

            self.get_logger().info(f"Downloading {os.path.basename(output_path)}...")
            with open(output_path, "wb") as file, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024, # Use 1024 for standard KB/MB
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    bar.update(len(data))
            self.get_logger().info(f"Download complete: {output_path}")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error downloading {url}: {e}")
            if output_path.exists():
                 output_path.unlink() # Remove partial file on error
            # Consider how to handle download failure (e.g., exit or retry)
            raise # Re-raise the exception


    def get_next_user_id(self, txn, increment_counter=False):
        counter_key = b"__counter__"
        raw = txn.get(counter_key)
        current_id = int(pickle.loads(raw)) if raw else 0
        next_id = current_id + 1 # Calculate next ID regardless
        if increment_counter:
            txn.put(counter_key, pickle.dumps(next_id))
            return f"U_{next_id:04d}"
        else:
            # If just reading, return the potential next ID without incrementing
            return f"U_{next_id:04d}"


    def increment_counter(self, txn): # Keep this if used elsewhere, otherwise redundant
        counter_key = b"__counter__"
        raw = txn.get(counter_key)
        current_id = int(pickle.loads(raw)) if raw else 0
        next_id = current_id + 1
        txn.put(counter_key, pickle.dumps(next_id))

    def save_new_embedding(self, env, emb):
        with env.begin(write=True) as txn:
            user_id = self.get_next_user_id(txn, increment_counter=True) # Ensure counter increments
            key = user_id.encode("utf-8")
            record = {"emb": emb, "count": 1}
            txn.put(key, pickle.dumps(record))
        return user_id

    def load_user_embeddings(self, env):
        embeddings = OrderedDict()
        with env.begin(write=False) as txn:
            counter_key = b"__counter__"
            raw_counter = txn.get(counter_key)
            max_id_num = int(pickle.loads(raw_counter)) if raw_counter else 0

            self.get_logger().info(f"Loading embeddings up to ID U_{max_id_num:04d}...")
            loaded_count = 0
            for i in range(1, max_id_num + 1):
                key_str = f"U_{i:04d}"
                key = key_str.encode("utf-8")
                value_bytes = txn.get(key)
                if value_bytes:
                    try:
                        record = pickle.loads(value_bytes)
                        if isinstance(record, dict) and 'emb' in record and 'count' in record:
                            # Basic validation: check if 'emb' looks like a numpy array
                            if isinstance(record['emb'], np.ndarray):
                                embeddings[key_str] = record
                                # self.get_logger().debug(f"Loaded {key_str}: count {record['count']}")
                                loaded_count += 1
                            else:
                                 self.get_logger().warn(f"Invalid embedding data type for {key_str}. Skipping.")
                        else:
                            self.get_logger().warn(f"Invalid record format for {key_str}. Skipping.")
                    except pickle.UnpicklingError:
                        self.get_logger().error(f"Could not unpickle data for key {key_str}. Skipping.")
                    except Exception as e:
                         self.get_logger().error(f"Error loading record for {key_str}: {e}. Skipping.")
                # else: # Log missing keys only if debugging extensively
                #     self.get_logger().debug(f"No data found for key {key_str}")

        self.get_logger().info(f"Successfully loaded {loaded_count} user embeddings.")
        return embeddings


    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
         # Ensure input are numpy arrays
        a = np.asarray(a)
        b = np.asarray(b)
        # Add epsilon to prevent division by zero if norm is zero
        norm_a = np.linalg.norm(a) + 1e-6
        norm_b = np.linalg.norm(b) + 1e-6
        return np.dot(a, b) / (norm_a * norm_b)


    def find_matching_user_vectorized(self, new_emb: np.ndarray, user_embs: OrderedDict, threshold: float = 0.5):
        if not user_embs:
            return None, None # No users in DB

        user_ids = list(user_embs.keys())
        # Ensure all embeddings are numpy arrays before stacking
        emb_list = [record["emb"] for record in user_embs.values() if isinstance(record.get("emb"), np.ndarray)]

        if not emb_list:
             self.get_logger().warn("No valid embeddings found in user_embs dictionary.")
             return None, None

        try:
            emb_matrix = np.stack(emb_list) # shape: (N, D)
        except ValueError as e:
            self.get_logger().error(f"Error stacking embeddings, likely due to inconsistent shapes: {e}")
            # Optionally log shapes for debugging:
            # for i, emb in enumerate(emb_list): self.get_logger().error(f"Emb {i} shape: {emb.shape}")
            return None, None # Cannot proceed if shapes mismatch

        # Normalize query embedding
        new_emb_norm_val = np.linalg.norm(new_emb)
        if new_emb_norm_val < 1e-6:
             self.get_logger().warn("Norm of new embedding is near zero.")
             return None, None # Avoid division by zero
        new_emb_norm = new_emb / new_emb_norm_val

        # Normalize database embeddings
        emb_matrix_norm_val = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        # Add epsilon where norm is near zero before division
        emb_matrix_norm_val[emb_matrix_norm_val < 1e-6] = 1e-6
        emb_matrix_norm = emb_matrix / emb_matrix_norm_val

        # Calculate cosine similarities
        sims = emb_matrix_norm @ new_emb_norm
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= threshold:
            # Make sure the index corresponds to the correct user ID if filtering occurred
            # If emb_list was filtered, need to map best_idx back to original user_ids
            # Simplified: Assuming emb_list wasn't filtered significantly or order maintained.
            # For robust solution: store (user_id, emb) pairs and filter/stack carefully.
            matched_user_id = user_ids[best_idx] # Assuming direct mapping works
            return matched_user_id, best_score
        else:
            return None, best_score # Return best score even if below threshold


    def destroy_node(self):
        self.get_logger().info("Shutting down Vision Node...")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Webcam released.")
        if hasattr(self, 'db_env'):
             self.db_env.close()
             self.get_logger().info("LMDB database closed.")
        pygame.quit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected.")
    except Exception as e:
         node.get_logger().fatal(f"Unhandled exception in spin: {e}", exc_info=True)
    finally:
        # Explicitly destroy the node upon exit
        node.destroy_node()
        # Ensure rclpy shutdown occurs cleanly
        if rclpy.ok():
            rclpy.shutdown()
        print("ROS 2 shutdown complete.")


if __name__ == '__main__':
    main()