import cv2
import os

import shutil

import torch
import sys
from pathlib import Path
import torchvision.transforms.v2 as transforms
from PIL import Image
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

import insightface


if __name__ == '__main__':
    





    # model = torch.hub.load('otroshi/edgeface', 'edgeface_s_gamma_05', source='github', pretrained=True, trust_repo=True)
    # model.eval()
    # model.to(torch.device('cuda'))
    
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])
    
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(16, 9))
    # ax.axis('off') 
    
    # model = insightface.app.FaceAnalysis(name='buffalo_l')  # or use 'buffalo_m' for lighter model
    # model.prepare(ctx_id=0)  # 0 means GPU; use -1 for CPU
    # img = cv2.imread('temp/imgs/test_img.png')  # Read image using OpenCV
    # faces = model.get(img)  # Detect faces
    # embedding1 = faces[0].embedding
    # for face in faces:
    #     x1, y1, x2, y2 = map(int, face.bbox)  # Bounding box
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    #     # Optional: Draw confidence or ID
    #     cv2.putText(img, f"Face", (x1, y1 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # ax.imshow(img)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove white borders
    # plt.show(block=True)
    
    model = insightface.app.FaceAnalysis(name='buffalo_l')  # or use 'buffalo_m' for lighter model
    model.prepare(ctx_id=0)  # 0 means GPU; use -1 for CPU
    
    cap = cv2.VideoCapture(0)
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off') 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = model.get(frame_rgb)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)  # Bounding box
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Optional: Draw confidence or ID
            cv2.putText(frame_rgb, f"Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(frame_rgb)
        plt.pause(0.001)
        ax.clear()
    
    
    # Typically the virtual device appears as device 0, but you may need to adjust this index.
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imshow("DroidCam Video", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # cap = cv2.VideoCapture(0)
    # plt.ion()  # Interactive mode on
    # fig, ax = plt.subplots()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     ax.imshow(frame_rgb)
    #     plt.pause(0.001)
    #     ax.clear()
    
    # KPRPE also takes keypoints locations as input
    



