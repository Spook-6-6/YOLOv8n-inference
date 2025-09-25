import cv2
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model(frame)
    cv2.imshow('YOLOv8n. Press ESC to exit.', results[0].plot())
    if cv2.waitKey(1) == 27:    #  27 = ESC
        break

cap.release()
cv2.destroyAllWindows()