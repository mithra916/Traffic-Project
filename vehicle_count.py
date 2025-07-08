'''from ultralytics import YOLO
from tracker import Tracker
import cv2

model = YOLO("yolov8n.pt")
tracker = Tracker()

def detect_vehicles(frame):
    results = model(frame)[0]
    detections = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = box.tolist()
            detections.append([x1, y1, x2, y2, conf.item(), int(cls)])

    tracked = tracker.update(detections, frame)
    return len(tracked)
'''

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    total_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        count = sum(1 for c in results.boxes.cls if int(c) in [2, 3, 5, 7])  # car, motorcycle, bus, truck
        total_count += count
        break  # Only first frame for speed
    cap.release()
    return total_count
