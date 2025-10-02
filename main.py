import cv2
from ultralytics import YOLO
import supervision as sv
import argparse
import time
from utils import draw_boxes, draw_text, CSRTracker

class TrackerState:
    current_id = None
    ids = []
    use_csrt = False
    detections = None
    target_lost_timer = None

def setup_camera(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                         int(cap.get(5)) or 30, (int(cap.get(3)), int(cap.get(4))))
    return cap, out

def is_inner_bbox_crossing_border(bbox, frame_shape):
    """Проверяет пересекает ли внутренняя зона BB границы кадра"""
    if bbox is None:
        return True
        
    x, y, w, h = bbox
    frame_height, frame_width = frame_shape[:2]
    
    # Вычисляем внутреннюю зону (50% от BB)
    inner_x = x + w * 0.25
    inner_y = y + h * 0.25
    inner_w = w * 0.5
    inner_h = h * 0.5
    
    # Проверяем пересечение с границами кадра
    crossing_left = inner_x < 0
    crossing_right = inner_x + inner_w > frame_width
    crossing_top = inner_y < 0
    crossing_bottom = inner_y + inner_h > frame_height
    
    return crossing_left or crossing_right or crossing_top or crossing_bottom

# Инициализация
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
cap, out = setup_camera(args.source)
tracker = sv.ByteTrack()
state = TrackerState()
csrt_tracker = CSRTracker()

# СОЗДАЕМ ОДНО ОКНО ЗАРАНЕЕ
window_name = 'YOLOv8n Tracking System'
cv2.namedWindow(window_name)

# Главный цикл обработки
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    # ОБРАБОТКА КЛАВИШ
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if not state.use_csrt:  # Переход в CSRT
            if state.current_id and state.current_id in state.ids and state.detections is not None:
                mask = state.detections.tracker_id == state.current_id
                bbox_detection = state.detections[mask]
                if len(bbox_detection) > 0:
                    x1, y1, x2, y2 = bbox_detection.xyxy[0]
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    print("Переход в CSRT режим с bbox:", bbox)
                    
                    if csrt_tracker.init(frame, bbox):
                        state.use_csrt = True
                        state.target_lost_timer = None
                        print("Успешно переключились в CSRT режим!")
        else:  # Возврат в мультитрекер
            state.use_csrt = False
            csrt_tracker.reset()
            state.target_lost_timer = None
            print("Возврат в мультитрекинг")
    
    # A/D - переключение объектов только в мультитрекере
    elif not state.use_csrt and state.ids:
        if key == ord('a'):
            idx = state.ids.index(state.current_id)
            state.current_id = state.ids[(idx - 1) % len(state.ids)]
        elif key == ord('d'):
            idx = state.ids.index(state.current_id)
            state.current_id = state.ids[(idx + 1) % len(state.ids)]
    
    # ОБЩАЯ ДЕТЕКЦИЯ ДЛЯ ВСЕХ РЕЖИМОВ
    detections = tracker.update_with_detections(sv.Detections.from_ultralytics(model(frame)[0]))
    state.ids = sorted(set(detections.tracker_id)) if detections else []
    state.detections = detections
    
    # ОСНОВНАЯ ЛОГИКА ОТРИСОВКИ
    status = ""
    
    if not state.use_csrt:
        # РЕЖИМ МУЛЬТИТРЕКЕРА
        if not state.current_id or state.current_id not in state.ids:
            state.current_id = state.ids[0] if state.ids else None
        
        if state.current_id and detections:
            mask = detections.tracker_id == state.current_id
            selected, normal = detections[mask], detections[~mask]
        else:
            selected, normal = sv.Detections.empty(), detections
        
        frame = draw_boxes(frame, normal, selected, model)
        
    else:
        # РЕЖИМ CSRT ТРЕКЕРА
        # Если таймер уже запущен (цель потеряна) - не обновляем CSRT
        if state.target_lost_timer is not None:
            status = f"Target {state.current_id} lost"
            
            # Через 3 секунды возвращаемся в мультитрекер
            if time.time() - state.target_lost_timer > 3:
                state.use_csrt = False
                csrt_tracker.reset()
                state.target_lost_timer = None
                print("Авто-возврат в мультитрекинг")
        
        else:
            # Цель еще не потеряна - обновляем CSRT
            success = csrt_tracker.update(frame)
            bbox = csrt_tracker.get_bbox()
            
            if success and bbox and not is_inner_bbox_crossing_border(bbox, frame.shape):
                # Цель в пределах - отрисовываем
                csrt_tracker.draw(frame, state.current_id)
            else:
                # Цель потеряна - запускаем таймер и СТОПИМ трекинг
                state.target_lost_timer = time.time()
                print("Цель потеряна - CSRT остановлен!")
                status = f"Target {state.current_id} lost"
    
    # Отрисовка интерфейса
    draw_text(frame, f'Time: {time.time()-start:.3f}s', (10, 30), color=(0, 255, 0))
    
    # Отображение режима справа
    mode = "Multi" if not state.use_csrt else "CSRT"
    mode_color = (0, 255, 255) if not state.use_csrt else (0, 255, 0)
    draw_text(frame, f'Mode: {mode}', (300, 30), color=mode_color)
    
    if not state.use_csrt and state.current_id:
        draw_text(frame, f'Selected ID: {state.current_id}', (10, 70), color=(0, 255, 255))
    
    if status:
        draw_text(frame, status, (10, 110), color=(0, 0, 255))
    
    # ОБНОВЛЯЕМ ОДНО ОКНО
    cv2.setWindowTitle(window_name, f'YOLOv8n. Mode: {mode} (SPACE-switch, ESC-exit)')
    cv2.imshow(window_name, frame)
    out.write(frame)

# Завершение работы
cap.release()
out.release()
cv2.destroyAllWindows()