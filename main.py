from camera_handler import process_video
from signal_controller import allocate_green_time
import random
from traffic_environment import TrafficEnvironment
import matplotlib.pyplot as plt
#from sumo_connector import start_sumo, close_sumo, set_traffic_light_phase, get_vehicle_count
'''
def handle_lane(main_cam, second_cam, lane_number):
    print(f"\n[INFO] Processing LANE {lane_number}...")

    main_count = process_video(main_cam)
    print(f"[RESULT] Main camera (RED) vehicle count (Lane {lane_number}): {main_count}")

    base_green_time = allocate_green_time(main_count)
    print(f"[INFO] Base green time: {base_green_time} seconds")

    print(f"[INFO] GREEN signal ON for Lane {lane_number}...")

    incoming_count = process_video(second_cam)
    print(f"[RESULT] Incoming vehicle count (100m cam): {incoming_count}")

    adjusted_time = optimize_with_rl(main_count, incoming_count, base_green_time)
    print(f"[INFO] Adjusted green time from RL: {adjusted_time} seconds")
    
    print(f"[INFO] RED signal ON for Lane {lane_number}\n{'-'*50}")

def main():
    pass

if __name__ == "__main__":
    main()
'''


'''
from traffic_environment import TrafficEnvironment
from reinforcement_agent import ActorCritic
from detector import detect_vehicles
import cv2


def main():
    env = TrafficEnvironment()
    agent = ActorCritic(state_dim=4, action_dim=4)

    for episode in range(100):
        state = env.reset()
        total_reward = 0

        for step in range(10):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

    print("\n--- FINAL SIGNAL TIMING & SWITCHING OUTPUT ---")
    lane_timings = env.get_lane_timings()
    for i, timing in enumerate(lane_timings):
        print(f"Lane {i+1}: Green Signal Time = {timing:.2f} seconds")


if __name__ == "__main__":
    main()

'''
 

'''
# main.py
from detector import detect_vehicles
import cv2

# Dummy video files for each lane (main and second cameras)
main_camera_videos = [
    "videos/lane1_main.mp4",
    "videos/lane2_main.mp4"
    #"videos/lane3_main.mp4",
    #"videos/lane4_main.mp4"
]

second_camera_videos = [
    "videos/lane1_second.mp4",
    "videos/lane2_second.mp4"
    #"videos/lane3_second.mp4",
    #"videos/lane4_second.mp4"
]


def allocate_green_time(vehicle_count):
    if vehicle_count > 50:
        return 60
    elif vehicle_count > 20:
        return 40
    else:
        return 20


def main():
    for i in range(4):
        print(f"[INFO] Processing LANE {i+1}...")

        print(f"[INFO] LANE {i+1}: Main camera analyzing red signal traffic...")
        main_count = detect_vehicles(main_camera_videos[i])
        print(f"[RESULT] Vehicles detected by main camera (Lane {i+1}): {main_count}")

        green_time = allocate_green_time(main_count)
        print(f"[INFO] LANE {i+1}: Base green time allocated: {green_time} seconds")

        print(f"[INFO] LANE {i+1}: GREEN signal ON...")

        print(f"[INFO] LANE {i+1}: Second camera analyzing incoming traffic before red...")
        second_count = detect_vehicles(second_camera_videos[i])
        print(f"[RESULT] Incoming vehicles detected (Lane {i+1}): {second_count}")

        print(f"[INFO] LANE {i+1}: Data sent to main camera for future optimization.")

        print(f"\n[INFO] LANE {i+1}: Signal switched to RED.")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
'''


from vehicle_count import detect_vehicles
import time
import os

def main():
    num_lanes = 4

    # Define video paths for each lane (main and second camera)
    main_camera_videos = [f"videos/lane{i+1}_main.mp4" for i in range(num_lanes)]
    second_camera_videos = [f"videos/lane{i+1}_second.mp4" for i in range(num_lanes)]

    for i in range(num_lanes):
        print(f"[INFO] Processing LANE {i+1}...")

        print(f"[INFO] LANE {i+1}: Main camera analyzing red signal traffic...")
        main_video_path = main_camera_videos[i]
        if not os.path.exists(main_video_path):
            print(f"[ERROR] Main camera video not found for LANE {i+1}: {main_video_path}")
            continue

        main_count = detect_vehicles(main_video_path)
        print(f"[RESULT] Vehicles detected by main camera (Lane {i+1}): {main_count}")

        # Allocate green time based on main camera detection
        if main_count > 50:
            green_time = 60
        elif main_count > 20:
            green_time = 40
        else:
            green_time = 20

        print(f"[INFO] LANE {i+1}: Base green time allocated: {green_time} seconds")

        print(f"[INFO] LANE {i+1}: GREEN signal ON...")
        time.sleep(1)  # simulate green signal period (can use green_time if needed)

        # Second camera analysis
        print(f"[INFO] LANE {i+1}: Second camera analyzing incoming traffic before red...")
        second_video_path = second_camera_videos[i]
        if not os.path.exists(second_video_path):
            print(f"[ERROR] Second camera video not found for LANE {i+1}: {second_video_path}")
            continue

        incoming_count = detect_vehicles(second_video_path)
        print(f"[RESULT] Incoming vehicles detected (Lane {i+1}): {incoming_count}")

        print(f"[INFO] LANE {i+1}: Data sent to main camera for future optimization.\n")

        print(f"[INFO] LANE {i+1}: Signal switched to RED.")
        print("-" * 50)
        time.sleep(1)

if __name__ == "__main__":
    main()
