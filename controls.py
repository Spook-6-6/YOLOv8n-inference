import cv2
from overlay import draw_text

class Controls:
    def __init__(self, tracker):
        self.tracker = tracker
        
    def handle_keys(self, key, frame):
        """Обработка нажатий клавиш"""
        if key == 27:  # ESC
            return False
        elif key == 32:  # SPACE
            self.tracker.switch_mode(frame)
        elif key == ord('a'):  # A
            self.tracker.switch_target('prev')
        elif key == ord('d'):  # D
            self.tracker.switch_target('next')
        return True
    
    def update_display_info(self, frame, processing_time, status):
        """Обновление информации на экране"""
        mode_color = (0, 255, 255) if status['mode'] == 'Multi' else (0, 255, 0)
        draw_text(frame, f'Time: {processing_time:.3f}s', (10, 30), (0, 255, 0))
        draw_text(frame, f'Mode: {status["mode"]}', (300, 30), mode_color)
        
        if not self.tracker.use_single_tracker and status['current_id']:
            draw_text(frame, f'Selected ID: {status["current_id"]}', (10, 70), (0, 255, 255))
        
        if status['status_text']:
            draw_text(frame, status['status_text'], (10, 110), (0, 0, 255))