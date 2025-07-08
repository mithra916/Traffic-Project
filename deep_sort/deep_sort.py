from deep_sort_realtime.deepsort_tracker import DeepSort as DS

class DeepSort:
    def __init__(self, max_age=30, n_init=3, nn_budget=100):
        """
        Initialize the DeepSort tracker with the given parameters.
        """
        self.deepsort = DS(max_age=max_age, n_init=n_init, nn_budget=nn_budget)

    def update(self, detections, frame):
        """
        Update the tracking state with new detections.
        :param detections: A list of detections for the current frame
        :param frame: The current video frame
        :return: Updated tracks
        """
        # Format: [[x1, y1, x2, y2, confidence, class_id], ...]
        tracks = self.deepsort.update_tracks(detections, frame)
        return tracks

    def get_tracks(self, frame):
        """
        Get all the active tracks.
        """
        return self.deepsort.get_active_tracks(frame)
