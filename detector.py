'''from ultralytics import YOLO
import random
model = YOLO("weights/yolov8n.pt")

def detect_vehicles(video_path):
    results = model(video_path, stream=True)
    detections = []
    for frame in results:
        frame_boxes = []
        for r in frame.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            frame_boxes.append([x1, y1, x2, y2, conf, cls])
        detections.append(frame_boxes)
    return detections


def get_vehicle_count_from_yolo(lane_id):
    """
    This function should run YOLO model on a video/image
    and return the number of detected vehicles for a given lane.
    """
    # Example dummy return
    return random.randint(0, 20)  # Later replace this with real YOLO detection
'''

import cv2
from ultralytics import YOLO

model = YOLO("weights/yolov8n.pt")


def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    vehicle_count = 0
    frame_skip = 5
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            count = sum(1 for c in results.boxes.cls if int(c) in [2, 3, 5, 7])  # car, motorcycle, bus, truck
            vehicle_count += count

        frame_idx += 1

    cap.release()
    return vehicle_count // 10  # Average scaling
