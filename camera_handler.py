import cv2
from vehicle_count import detect_vehicles

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count = detect_vehicles(frame)
        total_count += count

        cv2.putText(frame, f"Vehicles: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Traffic View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return total_count