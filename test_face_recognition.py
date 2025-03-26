

# import cv2
# import os
# import shutil
# import torch
# import sys
# from pathlib import Path
# import torchvision.transforms.v2 as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import lmdb
# import pickle
# from ultralytics import YOLO
# from tqdm import tqdm
# import requests
# from dotenv import load_dotenv
# import insightface
# import pygame
# from collections import OrderedDict
# import numpy as np
# import io

# load_dotenv()

# def download_file_fast(url, output_path):
#     """Downloads a file with a progress bar and handles retries."""
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(url, stream=True, headers=headers)
#     total_size = int(response.headers.get("content-length", 0))
#     chunk_size = 1024 * 1024  # 1MB chunks

#     with open(output_path, "wb") as file, tqdm(
#         desc=f"Downloading {os.path.basename(output_path)}",
#         total=total_size,
#         unit="B",
#         unit_scale=True,
#         unit_divisor=16384,
#     ) as bar:
#         for data in response.iter_content(chunk_size=chunk_size):
#             file.write(data)
#             bar.update(len(data))

# def get_next_user_id(txn, increment_counter=False):
#     counter_key = b"__counter__"
#     raw = txn.get(counter_key)
#     current_id = int(pickle.loads(raw)) if raw else 0
#     next_id = current_id + 1
#     if increment_counter:
#         txn.put(counter_key, pickle.dumps(next_id))
#     return f"U_{next_id:04d}"  # e.g., U_0001

# def increment_counter(txn):
#     counter_key = b"__counter__"
#     raw = txn.get(counter_key)
#     current_id = int(pickle.loads(raw)) if raw else 0
#     next_id = current_id + 1
#     txn.put(counter_key, pickle.dumps(next_id))
    
# # Modified: Save embedding record as a dictionary with embedding and update count
# def save_new_embedding(env, emb):
#     with env.begin(write=True) as txn:
#         user_id = get_next_user_id(txn, increment_counter=True)
#         key = user_id.encode("utf-8")
#         # New record: embedding and update count (starting at 1)
#         record = {"emb": emb, "count": 1}
#         txn.put(key, pickle.dumps(record))
#     return user_id

# def load_user_embeddings(env):
#     with env.begin(write=False) as txn:
#         counter_key = b"__counter__"
#         raw = txn.get(counter_key)
#         current_id = int(pickle.loads(raw)) if raw else 0
#         embeddings = OrderedDict()
#         for i in range(1, current_id + 1):  # IDs start from 1
#             key_str = f"U_{i:04d}"  # match format U_0001, U_0002, ...
#             key = key_str.encode("utf-8")
#             value_bytes = txn.get(key)
#             if value_bytes is None:
#                 print(f"Warning: Missing data for key {key_str}")
#                 continue
#             record = pickle.loads(value_bytes)
#             print(f"{key_str}: embedding shape {record['emb'].shape}, count {record['count']}")
#             embeddings[key_str] = record
#     return embeddings

# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # Modified: Use record['emb'] from each user record for matching
# def find_matching_user_vectorized(new_emb: np.ndarray, user_embs: OrderedDict, threshold: float = 0.5):
#     if not user_embs:
#         return None, None

#     user_ids = list(user_embs.keys())
#     emb_matrix = np.stack([record["emb"] for record in user_embs.values()])  # shape: (N, D)

#     # Normalize both the new embedding and the user embedding matrix
#     new_emb_norm = new_emb / np.linalg.norm(new_emb)
#     emb_matrix_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

#     # Cosine similarity via matrix multiplication
#     sims = emb_matrix_norm @ new_emb_norm  # shape: (N,)

#     best_idx = np.argmax(sims)
#     best_score = sims[best_idx]

#     if best_score >= threshold:
#         return user_ids[best_idx], best_score
#     else:
#         return None, best_score
    
# def initialize_db():
#     db_path = Path('temp/db/face_embs.lmdb')
#     db_path.parent.mkdir(exist_ok=True, parents=True)
#     db_env = lmdb.open(str(db_path.absolute()), map_size=int(1e8)) # 0.1GB max size
#     return db_env

# if __name__ == '__main__':
    
#     db_env = initialize_db()
#     user_embeddings = load_user_embeddings(db_env)

#     yolov12_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'
#     yolo_model_path = Path('temp/models/yolov12.pt')
#     yolo_model_path.parent.mkdir(exist_ok=True, parents=True)
    
#     if not yolo_model_path.exists():
#         download_file_fast(yolov12_url, yolo_model_path)
    
#     # Initialize object detection model
#     obj_model = YOLO(yolo_model_path)
#     obj_model = obj_model.eval()
#     obj_model = obj_model.to(torch.device('cuda'))
    
#     # Initialize face detection model
#     face_model = insightface.app.FaceAnalysis(name='buffalo_l')  # or use 'buffalo_m' for lighter model
#     face_model.prepare(ctx_id=0)  # 0 means GPU; use -1 for CPU

#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     if not ret:
#         raise RuntimeError("Failed to read from webcam.")
    
#     height, width, _ = frame.shape
#     pygame.init()
#     screen = pygame.display.set_mode((width, height))  # Use actual frame size

#     # Variables to store the latest detection annotations
#     last_obj_annotations = []  # Each element: (x1, y1, x2, y2, label, confidence)
#     last_face_annotations = []  # Each element: (x1, y1, x2, y2, user_id)
    
#     frame_count = 0
#     detection_interval = 5  # Run detections every 10 frames
#     # Maximum number of updates for a user embedding; after this, we stop updating to avoid excessive drift.
#     MAX_UPDATES = np.inf

#     running = True
#     while running:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB (for both detection and display)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Run detections every detection_interval frames
#         if frame_count % detection_interval == 0:
#             # Update object detections
#             last_obj_annotations = []
#             objects = obj_model(frame, verbose=False)
#             for result in objects:
#                 for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int()):
#                     label = result.names[cls.item()]
#                     if label == 'person':
#                         continue
#                     x1 = int(box[0].item())
#                     y1 = int(box[1].item())
#                     x2 = int(box[2].item())
#                     y2 = int(box[3].item())
#                     last_obj_annotations.append((x1, y1, x2, y2, label, conf.item()))
            
#             # Update face detections and recognition
#             last_face_annotations = []
#             faces = face_model.get(frame)
#             for face in faces:
#                 if face.det_score < 0.75:
#                     continue
#                 face_embedding = face.embedding  # numpy array from insightface
#                 # Recognize face using cosine similarity
#                 user_id, score = find_matching_user_vectorized(face_embedding, user_embeddings, threshold=0.5)
#                 if user_id is None:
#                     # New user detected, add to DB
#                     new_user_id = save_new_embedding(db_env, face_embedding)
#                     # Also update in-memory dictionary
#                     user_embeddings[new_user_id] = {"emb": face_embedding, "count": 1}
#                     recognized_label = new_user_id
#                     print(f"New user added: {new_user_id}")
#                 else:
#                     # Existing user recognized, update embedding using moving average if below MAX_UPDATES
#                     record = user_embeddings[user_id]
#                     if record["count"] < MAX_UPDATES:
#                         new_count = record["count"] + 1
#                         # Moving average update: new_avg = (old_avg * count + new_emb) / new_count
#                         updated_emb = (record["emb"] * record["count"] + face_embedding) / new_count
#                         record["emb"] = updated_emb
#                         record["count"] = new_count
#                         # Update record in DB
#                         with db_env.begin(write=True) as txn:
#                             txn.put(user_id.encode("utf-8"), pickle.dumps(record))
#                         print(f"Updated user {user_id}: count {new_count}, similarity score {score:.2f}")
#                     else:
#                         print(f"User {user_id} recognized with similarity {score:.2f} (max updates reached)")
#                     recognized_label = user_id
                
#                 # Get face bounding box and annotate with recognized label
#                 x1, y1, x2, y2 = map(int, face.bbox)
#                 last_face_annotations.append((x1, y1, x2, y2, recognized_label))
        
#         # Draw object detection annotations (red boxes and labels)
#         for (x1, y1, x2, y2, label, conf) in last_obj_annotations:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
#             cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
#         # Draw face detection annotations (green boxes with user id)
#         for (x1, y1, x2, y2, user_label) in last_face_annotations:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
#             cv2.putText(frame, user_label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Convert the frame for display in pygame
#         frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
#         screen.blit(frame, (0, 0))
#         pygame.display.flip()

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         frame_count += 1

#     cap.release()
#     pygame.quit()


import cv2
import os
import shutil
import torch
import sys
from pathlib import Path
import torchvision.transforms.v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt
import lmdb
import pickle
from ultralytics import YOLO
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import insightface
import pygame
from collections import OrderedDict
import numpy as np
import io

load_dotenv()

# Helper function: Intersection over Union (IoU) for two bounding boxes.
def iou(boxA, boxB):
    # boxA and boxB are tuples: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def download_file_fast(url, output_path):
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

def get_next_user_id(txn, increment_counter=False):
    counter_key = b"__counter__"
    raw = txn.get(counter_key)
    current_id = int(pickle.loads(raw)) if raw else 0
    next_id = current_id + 1
    if increment_counter:
        txn.put(counter_key, pickle.dumps(next_id))
    return f"U_{next_id:04d}"  # e.g., U_0001

def increment_counter(txn):
    counter_key = b"__counter__"
    raw = txn.get(counter_key)
    current_id = int(pickle.loads(raw)) if raw else 0
    next_id = current_id + 1
    txn.put(counter_key, pickle.dumps(next_id))
    
# Save a new embedding record (with a moving average counter)
def save_new_embedding(env, emb):
    with env.begin(write=True) as txn:
        user_id = get_next_user_id(txn, increment_counter=True)
        key = user_id.encode("utf-8")
        # Record contains the embedding and an update count (starting at 1)
        record = {"emb": emb, "count": 1}
        txn.put(key, pickle.dumps(record))
    return user_id

def load_user_embeddings(env):
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Match a new embedding against all stored user embeddings.
def find_matching_user_vectorized(new_emb: np.ndarray, user_embs: OrderedDict, threshold: float = 0.5):
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

def initialize_db():
    db_path = Path('temp/db/face_embs.lmdb')
    db_path.parent.mkdir(exist_ok=True, parents=True)
    db_env = lmdb.open(str(db_path.absolute()), map_size=int(1e8)) # 0.1GB max size
    return db_env

if __name__ == '__main__':
    
    db_env = initialize_db()
    user_embeddings = load_user_embeddings(db_env)

    yolov12_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'
    yolo_model_path = Path('temp/models/yolov12.pt')
    yolo_model_path.parent.mkdir(exist_ok=True, parents=True)
    
    if not yolo_model_path.exists():
        download_file_fast(yolov12_url, yolo_model_path)
    
    # Initialize object detection model
    obj_model = YOLO(yolo_model_path)
    obj_model = obj_model.eval()
    obj_model = obj_model.to(torch.device('cuda'))
    
    # Initialize face detection model
    face_model = insightface.app.FaceAnalysis(name='buffalo_l')  # or 'buffalo_m' for a lighter model
    face_model.prepare(ctx_id=0)  # 0 for GPU; -1 for CPU

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from webcam.")
    
    height, width, _ = frame.shape
    pygame.init()
    screen = pygame.display.set_mode((width, height))  # Use actual frame size

    # For object detection annotations
    last_obj_annotations = []  # Each element: (x1, y1, x2, y2, label, confidence)
    # For face annotations, we will use the temporal consistency mechanism
    last_face_annotations = []  # Each element: (x1, y1, x2, y2, recognized_label)

    # Temporal consistency candidates: each candidate is a dict with keys:
    # "bbox": bounding box (x1, y1, x2, y2)
    # "embedding": averaged embedding
    # "count": how many consecutive detection cycles it has been seen
    # "recognized_label": assigned user id (once processed)
    # "last_seen": last frame_count when candidate was updated
    face_candidates = []
    TEMPORAL_THRESHOLD = 3  # Require detection in 3 consecutive cycles before recognition.
    CANDIDATE_EXPIRY = 20   # Expire candidate if not seen within this many frame counts.
    detection_interval = 5  # Run detections every 5 frames
    MAX_UPDATES = np.inf
    DETECTION_SCORE_TSH = 0.75 # Threshold for considering a face as being detected.

    frame_count = 0
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for processing and display.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process detections every detection_interval frames.
        if frame_count % detection_interval == 0:
            # Update object detections
            last_obj_annotations = []
            objects = obj_model(frame, verbose=False)
            for result in objects:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls.int()):
                    label = result.names[cls.item()]
                    if label == 'person':
                        continue
                    x1 = int(box[0].item())
                    y1 = int(box[1].item())
                    x2 = int(box[2].item())
                    y2 = int(box[3].item())
                    last_obj_annotations.append((x1, y1, x2, y2, label, conf.item()))
            
            # Get face detections
            new_faces = face_model.get(frame)
            for face in new_faces:
                # Filter out low-confidence detections.
                if face.det_score < DETECTION_SCORE_TSH:
                    continue
                new_box = (int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3]))
                new_emb = face.embedding

                # Try to match the new face with an existing candidate (using IoU).
                matched_candidate = None
                for candidate in face_candidates:
                    if iou(candidate["bbox"], new_box) >= 0.3:
                        matched_candidate = candidate
                        break
                if matched_candidate:
                    old_count = matched_candidate["count"]
                    new_count = old_count + 1
                    # Update moving average of the embedding.
                    matched_candidate["embedding"] = (matched_candidate["embedding"] * old_count + new_emb) / new_count
                    matched_candidate["count"] = new_count
                    matched_candidate["bbox"] = new_box  # Update bbox to the latest.
                    matched_candidate["last_seen"] = frame_count
                else:
                    # Add a new candidate.
                    face_candidates.append({
                        "bbox": new_box,
                        "embedding": new_emb,
                        "count": 1,
                        "recognized_label": None,
                        "last_seen": frame_count
                    })

            # Remove candidates that haven’t been updated recently.
            face_candidates = [cand for cand in face_candidates if frame_count - cand["last_seen"] <= CANDIDATE_EXPIRY]

            # For each candidate that has been seen consistently, run recognition.
            for candidate in face_candidates:
                if candidate["count"] >= TEMPORAL_THRESHOLD and candidate["recognized_label"] is None:
                    user_id, score = find_matching_user_vectorized(candidate["embedding"], user_embeddings, threshold=0.5)
                    if user_id is None:
                        # New user detected.
                        new_user_id = save_new_embedding(db_env, candidate["embedding"])
                        user_embeddings[new_user_id] = {"emb": candidate["embedding"], "count": 1}
                        candidate["recognized_label"] = new_user_id
                        print(f"New user added: {new_user_id}")
                    else:
                        # Existing user: update the stored embedding.
                        record = user_embeddings[user_id]
                        if record["count"] < MAX_UPDATES:
                            new_count = record["count"] + 1
                            updated_emb = (record["emb"] * record["count"] + candidate["embedding"]) / new_count
                            record["emb"] = updated_emb
                            record["count"] = new_count
                            with db_env.begin(write=True) as txn:
                                txn.put(user_id.encode("utf-8"), pickle.dumps(record))
                            print(f"Updated user {user_id}: count {new_count}, similarity {score:.2f}")
                        else:
                            print(f"User {user_id} recognized with similarity {score:.2f} (max updates reached)")
                        candidate["recognized_label"] = user_id

            # Prepare face annotations from candidates that have been processed.
            last_face_annotations = []
            for candidate in face_candidates:
                if candidate["recognized_label"] is not None:
                    x1, y1, x2, y2 = candidate["bbox"]
                    last_face_annotations.append((x1, y1, x2, y2, candidate["recognized_label"]))
        
        # Draw object detection annotations (red boxes).
        for (x1, y1, x2, y2, label, conf) in last_obj_annotations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw face detection annotations (green boxes with recognized labels).
        for (x1, y1, x2, y2, user_label) in last_face_annotations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, user_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert frame for display in pygame.
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(frame, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_count += 1

    cap.release()
    pygame.quit()
