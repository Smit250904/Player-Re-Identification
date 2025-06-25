import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model and video
model = YOLO("best.pt")
cap = cv2.VideoCapture("15sec_input_720p.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# DeepSORT tracker
tracker = DeepSort(max_age=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] != "player":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "player"))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, w_box, h_box = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(l+w_box), int(t+h_box)), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {tid}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("✅ output_tracked.mp4 saved!")
