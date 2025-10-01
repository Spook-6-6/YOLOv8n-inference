import cv2
from ultralytics import YOLO
import supervision as sv
import argparse
import time
from utils import draw_boxes, draw_text, CSRTracker

class TrackerState:
    current_id = None
    ids = []
    use_csrt = False  # Флаг режима: False=мультитрекер, True=CSRT

def setup_camera(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                         int(cap.get(5)) or 30, (int(cap.get(3)), int(cap.get(4))))
    return cap, out

def handle_multitracker(frame, model, tracker, state, csrt_tracker):
    """Режим мультитрекера: отслеживание всех объектов с переключением между ними"""
    # Детекция всех объектов в кадре
    detections = tracker.update_with_detections(sv.Detections.from_ultralytics(model(frame)[0]))
    state.ids = sorted(set(detections.tracker_id)) if detections else []
    
    # Автовыбор первого объекта если ничего не выбрано
    if not state.current_id or state.current_id not in state.ids:
        state.current_id = state.ids[0] if state.ids else None
    
    # Обработка клавиш в режиме мультитрекера
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and state.ids:  # Переключение на предыдущий объект
        idx = state.ids.index(state.current_id)
        state.current_id = state.ids[(idx - 1) % len(state.ids)]
    elif key == ord('d') and state.ids:  # Переключение на следующий объект
        idx = state.ids.index(state.current_id)
        state.current_id = state.ids[(idx + 1) % len(state.ids)]
    elif key == 32 and state.current_id in state.ids:  # SPACE - ПЕРЕКЛЮЧЕНИЕ В CSRT РЕЖИМ
        # Захват выбранного объекта для CSRT трекера
        mask = detections.tracker_id == state.current_id
        bbox_detection = detections[mask]
        if len(bbox_detection) > 0:
            x1, y1, x2, y2 = bbox_detection.xyxy[0]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            csrt_tracker.init(frame, bbox)
            state.use_csrt = True  # Активация CSRT режима
    
    # Разделение объектов на выбранные и обычные для отрисовки
    if state.current_id and detections:
        mask = detections.tracker_id == state.current_id
        selected, normal = detections[mask], detections[~mask]
    else:
        selected, normal = sv.Detections.empty(), detections
    
    frame = draw_boxes(frame, normal, selected, model)
    return frame, ""

def handle_csrt_tracker(frame, state, csrt_tracker):
    """Режим CSRT трекера: точное отслеживание одного объекта"""
    # Обновление позиции CSRT трекера
    csrt_tracker.update(frame)
    csrt_tracker.draw(frame)
    
    # Обработка клавиш в режиме CSRT
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE - ВОЗВРАТ В МУЛЬТИТРЕКЕР
        state.use_csrt = False  # Деактивация CSRT режима
        csrt_tracker.reset()
    
    return frame, "CSRT Tracker Active"

def draw_ui(frame, state, status, processing_time):
    """Отрисовка интерфейса с информацией о текущем режиме"""
    draw_text(frame, f'Time: {processing_time:.3f}s', (10, 30), color=(0, 255, 0))
    
    # Отображение информации в зависимости от режима
    if not state.use_csrt and state.current_id:
        draw_text(frame, f'Selected ID: {state.current_id}', (10, 70), color=(0, 255, 255))
    elif state.use_csrt:
        draw_text(frame, "CSRT Tracker Mode", (10, 70), color=(0, 255, 0))
        draw_text(frame, "Press SPACE to return", (10, 110), color=(255, 255, 0))
    
    if status:
        draw_text(frame, status, (10, 110), color=(0, 0, 255))

# Инициализация
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='0')
args = parser.parse_args()

model = YOLO('yolov8n.pt')
cap, out = setup_camera(args.source)
tracker = sv.ByteTrack()
state = TrackerState()
csrt_tracker = CSRTracker()

# Главный цикл обработки
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    # ОСНОВНАЯ ЛОГИКА ВЫБОРА РЕЖИМА:
    if not state.use_csrt:
        # АКТИВЕН МУЛЬТИТРЕКЕР - отслеживание всех объектов
        frame, status = handle_multitracker(frame, model, tracker, state, csrt_tracker)
    else:
        # АКТИВЕН CSRT ТРЕКЕР - точное отслеживание одного объекта
        frame, status = handle_csrt_tracker(frame, state, csrt_tracker)
    
    # Отрисовка интерфейса
    draw_ui(frame, state, status, time.time() - start)
    
    # Обновление заголовка окна с информацией о режиме
    mode = "Multi" if not state.use_csrt else "CSRT"
    cv2.imshow(f'YOLOv8n. Mode: {mode} (SPACE-switch, ESC-exit)', frame)
    out.write(frame)

# Завершение работы
cap.release()
out.release()
cv2.destroyAllWindows()