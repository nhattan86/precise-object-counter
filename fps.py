import cv2
import numpy as np
import time
from datetime import datetime
import collections
import os
import logging
from pathlib import Path
import argparse

class LightweightObjectCounter:
    def __init__(self, config):
        self.setup_logging()
        self.logger.info(f"Initializing Object Counter...")
        self.logger.info(f"Started by: {self.get_user_info()}")
        self.logger.info(f"Start time: {self.get_current_time()}")

        # Load configuration
        self.config = config
        self.setup_camera()
        self.setup_model()
        self.setup_tracking()
        self.setup_roi()
        
        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = collections.deque(maxlen=50)  # Limit memory usage
        self.up_track_ids = collections.deque(maxlen=50)
        
        # Performance monitoring
        self.fps_start_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.frame_times = collections.deque(maxlen=50)  # Last 50 frame times

    def setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger('ObjectCounter')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(self.config['camera_id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def setup_model(self):
        """Initialize TFLite model for lightweight inference"""
        import tflite_runtime.interpreter as tflite
        
        self.interpreter = tflite.Interpreter(
            model_path=self.config['model_path']
        )
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def setup_tracking(self):
        """Initialize simple tracking system"""
        self.tracks = {}
        self.next_track_id = 0
        self.track_threshold = 0.3
        self.max_track_age = 10

    def setup_roi(self):
        """Set up counting region"""
        height = self.config['height']
        width = self.config['width']
        
        self.count_roi = [
            0,                          # x start
            int(height - height / 1.8), # y start
            width,                      # width
            int(height / 8.6)           # height
        ]

    def get_user_info(self):
        """Get current user information"""
        return os.getenv('USER', 'unknown')

    def get_current_time(self):
        """Get current time in UTC"""
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        input_shape = self.input_shape
        processed = cv2.resize(frame, (input_shape[1], input_shape[2]))
        processed = processed.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        return processed

    def detect_objects(self, frame):
        """Perform lightweight object detection"""
        processed = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            processed
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index']
        )[0]
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index']
        )[0]
        
        return self.process_detections(boxes, classes, scores)

    def process_detections(self, boxes, classes, scores):
        """Process detections with confidence threshold"""
        detections = []
        for box, class_id, score in zip(boxes, classes, scores):
            if score >= self.config['conf_threshold']:
                detections.append({
                    'bbox': box,
                    'class_id': int(class_id),
                    'confidence': score
                })
        return detections

    def update_tracking(self, detections):
        """Simple tracking update"""
        current_time = time.time()
        
        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track['age'] += 1
            if track['age'] > self.max_track_age:
                del self.tracks[track_id]
        
        # Match detections to tracks
        for detection in detections:
            matched = False
            for track in self.tracks.values():
                if self.calculate_iou(detection['bbox'], track['bbox']) > self.track_threshold:
                    track.update(detection)
                    matched = True
                    break
            
            if not matched:
                self.tracks[self.next_track_id] = {
                    'id': self.next_track_id,
                    'bbox': detection['bbox'],
                    'age': 0,
                    'history': collections.deque(maxlen=5)
                }
                self.next_track_id += 1

    def process_frame(self):
        """Process a single frame"""
        try:
            frame_start = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # Detect and track objects
            detections = self.detect_objects(frame)
            self.update_tracking(detections)
            
            # Count objects
            self.count_objects()
            
            # Update performance metrics
            self.update_performance_metrics(frame_start)
            
            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'fps': self.fps,
                'frame': frame
            }
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None

def main():
    # Configuration
    config = {
        'camera_id': 0,
        'width': 320,
        'height': 240,
        'fps': 15,
        'model_path': 'model.tflite',
        'conf_threshold': 0.3
    }
    
    counter = LightweightObjectCounter(config)
    
    try:
        while True:
            result = counter.process_frame()
            if result:
                # Display minimal info to conserve resources
                print(f"\rFPS: {result['fps']:.1f} | "
                      f"Up: {result['up_down_count']} | "
                      f"Down: {result['down_up_count']}", end='')
            
            time.sleep(0.001)  # Minimal delay
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        counter.cap.release()

if __name__ == "__main__":
    main()
