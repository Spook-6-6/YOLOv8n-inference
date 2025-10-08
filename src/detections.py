import cv2
import supervision as sv
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='../models/yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.detections = None
        self.ids = []
        
    def process_frame(self, frame):
        self.detections = self.tracker.update_with_detections(
            sv.Detections.from_ultralytics(self.model(frame)[0]))
        
        self.ids = sorted(set(self.detections.tracker_id)) if self.detections else []
        return self.detections