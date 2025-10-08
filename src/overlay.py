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

def is_inner_bbox_crossing_border(bbox, frame_shape):
    """Проверяет пересекает ли внутренняя зона BB границы кадра"""
    if not bbox:
        return True
        
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]
    
    inner_x = x + w * 0.25
    inner_y = y + h * 0.25
    inner_w = w * 0.5
    inner_h = h * 0.5
    
    crossing_left = inner_x < 0
    crossing_right = inner_x + inner_w > frame_width
    crossing_top = inner_y < 0
    crossing_bottom = inner_y + inner_h > frame_height
    
    return crossing_left or crossing_right or crossing_top or crossing_bottom

def draw_csrt_tracker(frame, bbox, tracker_id=None):
    """Отрисовка CSRT трекера"""
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID{tracker_id}_CSRT" if tracker_id else "CSRT"
        draw_text(frame, label, (x, y-10), (0, 255, 0))

def draw_nano_tracker(frame, bbox, tracker_id=None):
    """Отрисовка NanoTrack трекера"""
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Синий
        label = f"ID{tracker_id}_NANO" if tracker_id else "NANO"
        draw_text(frame, label, (x, y-10), (255, 0, 0))

def draw_vit_tracker(frame, bbox, tracker_id=None):
    """Отрисовка ViT трекера"""
    if bbox:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красный
        label = f"ID{tracker_id}_ViT" if tracker_id else "ViT"
        draw_text(frame, label, (x, y-10), (0, 0, 255))