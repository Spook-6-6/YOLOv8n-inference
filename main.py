import cv2
from ultralytics import YOLO
import supervision as sv
import argparse
import time

class Switch:
    current_id = None
    ids = []

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                     int(cap.get(5)) or 30, 
                     (int(cap.get(3)), int(cap.get(4))))

normal_annotator = sv.BoxAnnotator(color=sv.Color.BLUE)
selected_annotator = sv.BoxAnnotator(color=sv.Color.RED)
label_annotator = sv.LabelAnnotator()

switch = Switch()
tracker = sv.ByteTrack()

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    detections = sv.Detections.from_ultralytics(model(frame)[0])
    detections = tracker.update_with_detections(detections)
    
    switch.ids = sorted(set(detections.tracker_id)) if detections else []
    if not switch.current_id or switch.current_id not in switch.ids:
        switch.current_id = switch.ids[0] if switch.ids else None
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and switch.ids:
        idx = switch.ids.index(switch.current_id)
        switch.current_id = switch.ids[(idx - 1) % len(switch.ids)]
    elif key == ord('d') and switch.ids:
        idx = switch.ids.index(switch.current_id)
        switch.current_id = switch.ids[(idx + 1) % len(switch.ids)]
    elif key == 27: break
    
    if switch.current_id and detections:
        mask = detections.tracker_id == switch.current_id
        selected_detections, normal_detections = detections[mask], detections[~mask]
    else:
        selected_detections, normal_detections = sv.Detections.empty(), detections
    
    for dets, annotator in [(normal_detections, normal_annotator), 
                           (selected_detections, selected_annotator)]:
        if dets:
            labels = [f"{model.model.names[cls]} {conf:.2f} {tid}" 
                     for cls, conf, tid in zip(dets.class_id, dets.confidence, dets.tracker_id)]
            frame = annotator.annotate(frame, dets)
            frame = label_annotator.annotate(frame, dets, labels=labels)
    
    if switch.current_id:
        cv2.putText(frame, f'Selected ID: {switch.current_id}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.putText(frame, f'Time: {time.time()-start:.3f}s', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8n. Use A/D to switch IDs. ESC to exit.', frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()