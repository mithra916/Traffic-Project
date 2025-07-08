from deep_sort.deep_sort import DeepSort

class Tracker:
    def __init__(self):
        self.deepsort = DeepSort()

    def update(self, detections, frame):
        # Each detection: [x1, y1, x2, y2, confidence, class_id]
        tracked_objects = self.deepsort.update_tracks(detections, frame=frame)
        return tracked_objects

