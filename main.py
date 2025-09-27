import cv2
from ultralytics import YOLO
import supervision as sv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
tracker = sv.ByteTrack()
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    labels = [
        f"{model.model.names[class_id]} {conf:.2f} {tracker_id}"
        for class_id, conf, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)
    ]
    
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels=labels)
    
    cv2.putText(frame, f'Time: {time.time()-start:.3f}s', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8n. Press ESC to exit.', frame)
    out.write(frame)
    if cv2.waitKey(1) == 27: break

cap.release()
out.release()
cv2.destroyAllWindows()