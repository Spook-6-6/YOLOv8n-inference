import cv2
import supervision as sv

# Аннотаторы
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
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
            success = self.tracker.init(frame, bbox)
            if success:
                self.bbox = bbox
                print("CSRT трекер успешно инициализирован через legacy!")
            return success
        except Exception as e:
            print(f"Ошибка инициализации CSRT трекера: {e}")
            return False
        
    def update(self, frame):
        """Обновление позиции трекера"""
        if self.tracker:
            success, self.bbox = self.tracker.update(frame)
            return success
        return False

    def get_bbox(self):
        """Возвращает текущий bbox"""
        return self.bbox
        
    def draw(self, frame, tracker_id=None):
        """Отрисовка bounding box трекера с ID"""
        if self.bbox:
            x, y, w, h = [int(v) for v in self.bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Отображаем ID_CSRT
            label = f"ID{tracker_id}_CSRT" if tracker_id else "CSRT"
            draw_text(frame, label, (x, y-10), (0, 255, 0))
        
    def reset(self):
        """Сброс трекера для возврата в мультитрекинг"""
        self.tracker = None
        self.bbox = None