# Smart Traffic Signal System 


## Overview
```
This project implements a full-fledged AI-based Smart Traffic Signal System that intelligently manages signal timings at a four-way junction using real-time traffic data.
The system leverages cutting-edge deep learning techniques and reinforcement learning to ensure efficient traffic flow and reduced congestion. It is designed to be adaptable, scalable, and suitable for deployment in urban smart city environments.
```

## Objectives

```
Dynamically allocate green signal duration based on real-time vehicle density.

Ensure optimal traffic movement using reinforcement learning.

Simulate real-world traffic conditions using SUMO.

Automate signal switching without manual intervention.
```


## System Architecture

### Dual Camera Setup per Lane:
```
Main Camera (at signal): Detects and counts vehicles waiting at the red signal.

Second Camera (100 meters away): Detects incoming vehicles just before green ends and sends data to the main camera for future optimization.
```


## Core Components
```
Module	Description
YOLOv8	Real-time object detection to identify vehicles from camera feeds
DeepSORT	Multi-object tracking to assign IDs and count unique vehicles
Actor-Critic RL Agent	Allocates optimal green signal timing based on traffic flow
SUMO	Traffic simulation environment for reinforcement learning integration
OpenCV + PyTorch	Video processing and model deployment backend
```

## Folder Structure
```

smart-traffic-signal/
├── venv/                    # Python virtual environment
├── deep_sort/               # DeepSORT tracking logic
├── videos/                  # Input videos for 4 lanes
├── weights/
│   └── yolov8n.pt           # Pre-trained YOLOv8 weights
├── actor_critic.py          # Actor-Critic reinforcement learning agent
├── camera_handler.py        # Captures and processes video input
├── detector.py              # Vehicle detection with YOLOv8
├── tracker.py               # Vehicle tracking with DeepSORT
├── reinforcement_agent.py   # RL logic and agent-environment interaction
├── signal_controller.py     # Logic to switch signals based on decisions
├── sumo_connector.py        # Integrates SUMO simulation
├── main.py                  # Entry point for system execution
└── requirements.txt         # Dependencies

```

## Features
✅ Real-time vehicle detection from multiple video streams

✅ Dual camera logic per lane

✅ Dynamic green signal duration between 20 to 80 seconds

✅ Vehicle count–based decision-making

✅ Reinforcement learning–powered intelligent switching

✅ Observation window for incoming traffic detection

✅ SUMO simulation support for training and evaluation



## Sample Output
```
📌 Signal switching logs per lane

📦 Vehicle count logs (from both cameras)

📈 Dynamic green light timing displayed per lane

🧠 Reinforcement agent decision outputs

🎥 Bounding boxes on vehicles shown in real time
```

## Technologies Used
```
YOLOv8 – Object Detection (Ultralytics)

DeepSORT – Object Tracking

PyTorch – Deep Learning Framework

OpenCV – Video Processing

SUMO – Traffic Simulation

Python – Core Development Language
```



