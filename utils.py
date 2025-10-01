import cv2
import supervision as sv

# Аннотаторы для мультитрекера (все объекты)
normal_box = sv.BoxAnnotator(color=sv.Color.BLUE)
selected_box = sv.BoxAnnotator(color=sv.Color.RED)
label_annotator = sv.LabelAnnotator()

def draw_boxes(frame, normal_dets, selected_dets, model):
    """Отрисовка объектов мультитрекера: синие - обычные, красные - выбранные"""
    for dets, box in [(normal_dets, normal_box), (selected_dets, selected_box)]:
        if dets:
            labels = [
                f"{model.model.names[cls]} {conf:.2f} {tid}" 
                for cls, conf, tid in zip(dets.class_id, dets.confidence, dets.tracker_id)
            ]
            frame = box.annotate(frame, dets)
            frame = label_annotator.annotate(frame, dets, labels=labels)
    return frame

def draw_text(frame, text, position, color=(255, 255, 255)):
    """Добавление текстовой информации на кадр"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

class CSRTracker:
    """CSRT трекер для точного отслеживания одного объекта"""
    
    def __init__(self):
        self.tracker = None
        self.bbox = None
        
    def init(self, frame, bbox):
        """Инициализация трекера на выбранном объекте"""
        self.tracker = cv2.TrackerCSRT_create()
        success = self.tracker.init(frame, bbox)
        if success:
            self.bbox = bbox
        return success
        
    def update(self, frame):
        """Обновление позиции трекера"""
        if self.tracker:
            success, self.bbox = self.tracker.update(frame)
            return success
        return False
        
    def draw(self, frame):
        """Отрисовка bounding box трекера"""
        if self.bbox:
            x, y, w, h = [int(v) for v in self.bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_text(frame, "CSRT Tracking", (x, y-10), (0, 255, 0))
        
    def reset(self):
        """Сброс трекера для возврата в мультитрекинг"""
        self.tracker = None
        self.bbox = None