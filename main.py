import cv2
import argparse
from tracking import Tracker
from controls import Controls

def setup_camera(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                         int(cap.get(5)) or 30, (int(cap.get(3)), int(cap.get(4))))
    return cap, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0')
    parser.add_argument('--csrt', action='store_true', help='Use CSRT tracker')
    parser.add_argument('--nano', action='store_true', help='Use NanoTrack tracker') 
    parser.add_argument('--vit', action='store_true', help='Use ViT tracker')
    args = parser.parse_args()
    
    # Определяем тип трекера
    if args.csrt: 
        tracker_type = 'csrt'
    elif args.nano: 
        tracker_type = 'nano'  
    elif args.vit: 
        tracker_type = 'vit'
    else: 
        tracker_type = 'csrt'  # по умолчанию
    
    tracker = Tracker(tracker_type=tracker_type)
    controls = Controls(tracker)
    cap, out = setup_camera(args.source)
    window_name = 'YOLOv8n Tracking System'
    cv2.namedWindow(window_name)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        processed_frame, processing_time = tracker.process_frame(frame)
        status = tracker.get_status()
        
        # Обновление интерфейса
        controls.update_display_info(processed_frame, processing_time, status)
        
        # Обновление окна
        cv2.setWindowTitle(window_name, f'YOLOv8n. Mode: {status["mode"]} (SPACE-switch, ESC-exit)')
        cv2.imshow(window_name, processed_frame)
        out.write(processed_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if not controls.handle_keys(key, frame):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()