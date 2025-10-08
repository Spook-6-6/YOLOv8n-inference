import cv2
import time
import os
import supervision as sv
from overlay import draw_boxes, is_inner_bbox_crossing_border, draw_csrt_tracker, draw_nano_tracker, draw_vit_tracker
from detections import Detector

class Tracker:
    def __init__(self, model_path='../models/yolov8n.pt', tracker_type='csrt'):
        self.detector = Detector(model_path)
        self.tracker_type = tracker_type
        self.single_tracker = self._create_tracker(tracker_type)
        
        self.current_id = None
        self.use_single_tracker = False
        self.target_lost_timer = None
        
    def _create_tracker(self, tracker_type):
        if tracker_type == 'csrt':
            return CSRTracker()
        elif tracker_type == 'nano':
            return NanoTracker()
        elif tracker_type == 'vit':
            return ViTTracker()
        return None
        
    def process_frame(self, frame):
        start = time.time()
        
        # Детекция для всех режимов
        self.detector.process_frame(frame)
        self.detections = self.detector.detections
        self.ids = self.detector.ids
        
        # Обработка кадра
        if not self.use_single_tracker:
            frame = self._process_multi_tracker(frame)
        else:
            frame = self._process_single_tracker(frame)
            
        processing_time = time.time() - start
        return frame, processing_time
    
    def _process_multi_tracker(self, frame):
        if not self.current_id or self.current_id not in self.ids:
            self.current_id = self.ids[0] if self.ids else None
            
        if self.current_id and self.detections:
            mask = self.detections.tracker_id == self.current_id
            selected, normal = self.detections[mask], self.detections[~mask]
        else:
            selected, normal = sv.Detections.empty(), self.detections
            
        return draw_boxes(frame, normal, selected, self.detector.model)
    
    def _process_single_tracker(self, frame):
        if self.target_lost_timer is not None:
            if time.time() - self.target_lost_timer > 3:
                self.use_single_tracker = False
                self.single_tracker.reset()
                self.target_lost_timer = None
        elif self.single_tracker and self._tracker_failed(frame):
            self.target_lost_timer = time.time()
        return frame

    def _tracker_failed(self, frame):
        success = self.single_tracker.update(frame)
        bbox = self.single_tracker._bbox
        
        if success and bbox and not is_inner_bbox_crossing_border(bbox, frame.shape):
            self._draw_single_tracker(frame, bbox)
            return False
        return True

    def _draw_single_tracker(self, frame, bbox):
        if self.tracker_type == 'csrt':
            draw_csrt_tracker(frame, bbox, self.current_id)
        elif self.tracker_type == 'nano':
            draw_nano_tracker(frame, bbox, self.current_id)
        elif self.tracker_type == 'vit':
            draw_vit_tracker(frame, bbox, self.current_id)
    
    def switch_mode(self, frame):
        if self.use_single_tracker:
            # Возврат в мультитрекер
            self.use_single_tracker = False
            if self.single_tracker:
                self.single_tracker.reset()
            self.target_lost_timer = None
        else:
            # Переход в одиночный трекинг
            if self.current_id and self.detections and self.single_tracker:
                if self._init_single_tracker(frame):
                    self.use_single_tracker = True
                    self.target_lost_timer = None
    
    def _init_single_tracker(self, frame):
        mask = self.detections.tracker_id == self.current_id
        bbox_detection = self.detections[mask]
        if len(bbox_detection) > 0:
            x1, y1, x2, y2 = bbox_detection.xyxy[0]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            return self.single_tracker.init(frame, bbox)
        return False
    
    def switch_target(self, direction):
        if not self.use_single_tracker and self.ids and self.current_id in self.ids:
            idx = self.ids.index(self.current_id)
            if direction == 'prev':
                self.current_id = self.ids[(idx - 1) % len(self.ids)]
            elif direction == 'next':
                self.current_id = self.ids[(idx + 1) % len(self.ids)]
    
    def get_status(self):
        mode = "Multi" if not self.use_single_tracker else self.tracker_type.upper()
        mode_color = (0, 255, 255) if not self.use_single_tracker else (0, 255, 0)
        status_text = ""
        
        if self.target_lost_timer is not None:
            status_text = f"Target {self.current_id} lost"
            
        return {
            'mode': mode,
            'mode_color': mode_color,
            'current_id': self.current_id,
            'status_text': status_text
        }

class CSRTracker:
    def __init__(self):
        self.tracker = None
        self._bbox = None
        
    @property
    def bbox(self):
        return self._bbox
        
    def init(self, frame, bbox):
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
            success = self.tracker.init(frame, bbox)
            if success:
                self._bbox = bbox
            return success
        except Exception as e:
            print(f"CSRT init error: {e}")
            return False
        
    def update(self, frame):
        if self.tracker:
            success, self._bbox = self.tracker.update(frame)
            return success
        return False
        
    def reset(self):
        self.tracker = self._bbox = None

class NanoTracker:
    def __init__(self):
        self.tracker = None
        self._bbox = None
        
    @property
    def bbox(self):
        return self._bbox
        
    def init(self, frame, bbox):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(current_dir, '..', 'weights')
            backbone_path = os.path.join(weights_dir, "nanotrack_backbone_sim.onnx")
            neckhead_path = os.path.join(weights_dir, "nanotrack_head_sim.onnx")
            
            params = cv2.TrackerNano_Params()
            params.backbone = backbone_path
            params.neckhead = neckhead_path
            
            self.tracker = cv2.TrackerNano_create(params)
            
            # Тихая инициализация
            result = self.tracker.init(frame, bbox)
            self._bbox = bbox
            return True  # Обход бага с None
            
        except Exception as e:
            return False
        
    def update(self, frame):
        if self.tracker:
            success, self._bbox = self.tracker.update(frame)
            return success
        return False
        
    def reset(self):
        self.tracker = self._bbox = None

class ViTTracker:
    def __init__(self):
        self.tracker = None
        self._bbox = None
        
    @property
    def bbox(self):
        return self._bbox
        
    def init(self, frame, bbox):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(current_dir, '..', 'weights')
            vit_path = os.path.join(weights_dir, "object_tracking_vittrack_2023sep.onnx")
            
            params = cv2.TrackerVit_Params()
            params.net = vit_path
            
            self.tracker = cv2.TrackerVit_create(params)
            result = self.tracker.init(frame, bbox)
            self._bbox = bbox
            return True
        except Exception as e:
            return False
        
    def update(self, frame):
        if self.tracker:
            success, self._bbox = self.tracker.update(frame)
            return success
        return False
        
    def reset(self):
        self.tracker = self._bbox = None