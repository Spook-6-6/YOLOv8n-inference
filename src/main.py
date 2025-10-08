import cv2
import argparse
import os
from tracking import Tracker
from controls import Controls

def setup_camera(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    output_dir = '../media_local'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                         int(cap.get(5)) or 30, (int(cap.get(3)), int(cap.get(4))))
    return cap, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0')
    parser.add_argument('--tracker', type=str, choices=['csrt', 'nano', 'vit'], 
                       default='csrt', help='Choose tracker type')
    args = parser.parse_args()
    
    tracker_type = args.tracker
    
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