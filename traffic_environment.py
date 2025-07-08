
import random
from detector import detect_vehicles

class TrafficEnvironment:
    def __init__(self):
        self.num_lanes = 4
        self.vehicle_count = [0] * self.num_lanes
        self.lane_timings = [0] * self.num_lanes
        self.video_paths = [f"videos/lane{i+1}.mp4" for i in range(self.num_lanes)]

    def reset(self):
        self.vehicle_count = []
        for i in range(self.num_lanes):
            count = detect_vehicles(self.video_paths[i])
            self.vehicle_count.append(count)
        return self.vehicle_count

    def step(self, action):
        # Vehicles cleared during green signal
        cleared = min(self.vehicle_count[action], random.randint(5, 15))
        self.vehicle_count[action] -= cleared

        # Simulate new vehicles entering other lanes
        for i in range(self.num_lanes):
            if i != action:
                self.vehicle_count[i] += random.randint(0, 3)

        # Green time allocation logic
        if cleared > 50:
            green_time = 60
        elif cleared > 20:
            green_time = 40
        else:
            green_time = 20

        self.lane_timings[action] = green_time
        reward = cleared  # reward based on how many vehicles were cleared

        done = all(v <= 0 for v in self.vehicle_count)
        return self.vehicle_count, reward, done

    def get_lane_timings(self):
        return self.lane_timings
