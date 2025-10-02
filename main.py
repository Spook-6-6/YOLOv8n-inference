import cv2
from ultralytics import YOLO
import supervision as sv
import argparse
import time

class Tracker:
    current_id = None
    ids = []
    locked_id = None
    lock_time = None

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                     int(cap.get(5)) or 30, (int(cap.get(3)), int(cap.get(4))))

normal_box = sv.BoxAnnotator(color=sv.Color.BLUE)
selected_box = sv.BoxAnnotator(color=sv.Color.RED)
label_annotator = sv.LabelAnnotator()

tracker = sv.ByteTrack()
state = Tracker()

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    detections = tracker.update_with_detections(sv.Detections.from_ultralytics(model(frame)[0]))
    state.ids = sorted(set(detections.tracker_id)) if detections else []
    
    if not state.current_id or state.current_id not in state.ids:
        state.current_id = state.ids[0] if state.ids else None
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and state.ids:
        idx = state.ids.index(state.current_id)
        state.current_id = state.ids[(idx - 1) % len(state.ids)]
    elif key == ord('d') and state.ids:
        idx = state.ids.index(state.current_id)
        state.current_id = state.ids[(idx + 1) % len(state.ids)]
    elif key == 32:
        state.locked_id = None if state.locked_id else state.current_id
        state.lock_time = None
    elif key == 27: break
    
    if state.locked_id and state.locked_id not in state.ids:
        state.lock_time = state.lock_time or time.time()
        if time.time() - state.lock_time > 3:
            state.locked_id = None
    
    if state.locked_id:
        if state.locked_id in state.ids:
            mask = detections.tracker_id == state.locked_id
            selected, normal = detections[mask], sv.Detections.empty()
            status = f"Target {state.locked_id} locked"
        else:
            selected = normal = sv.Detections.empty()
            status = f"Target {state.locked_id} lost"
    else:
        if state.current_id and detections:
            mask = detections.tracker_id == state.current_id
            selected, normal = detections[mask], detections[~mask]
        else:
            selected, normal = sv.Detections.empty(), detections
        status = ""
    
    for dets, box in [(normal, normal_box), (selected, selected_box)]:
        if dets:
            labels = [f"{model.model.names[cls]} {conf:.2f} {tid}" 
                     for cls, conf, tid in zip(dets.class_id, dets.confidence, dets.tracker_id)]
            frame = box.annotate(frame, dets)
            frame = label_annotator.annotate(frame, dets, labels=labels)
    
    if state.current_id:
        cv2.putText(frame, f'Selected ID: {state.current_id}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if status:
        cv2.putText(frame, status, (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame, f'Time: {time.time()-start:.3f}s', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8n. A/D - switch, Space - lock/unlock, ESC - exit', frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()