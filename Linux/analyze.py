% python analyze.py --camera 0 --model ./models/yolov8s.onnx

import cv2
import numpy as np
from bytetrack_cpp import BYTETracker
import time
from datetime import datetime
import argparse
import sys
import os
import collections

class ObjectDetectionCounter:
    def __init__(self, camera_id=0, model_path="./models/yolov8s.onnx"):
        """Initialize with performance monitoring"""
        print(f"Current Date and Time (UTC): {self.get_current_time()}")
        print(f"Current User's Login: {self.get_user_info()}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize YOLO model
        self.detector = cv2.dnn.readNetFromONNX(model_path)
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(
            track_thresh=0.4,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )

        # Detection parameters
        self.conf_threshold = 0.3
        self.iou_threshold = 0.45
        self.input_size = (640, 640)

        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = []
        self.up_track_ids = []

        # Performance monitoring
        self.frame_times = collections.deque(maxlen=30)  # Store last 30 frames
        self.fps = 0
        self.fps_update_time = time.time()
        self.frame_count = 0
        
        # Setup ROI
        self.setup_roi()

    def get_current_time(self):
        """Get current time in UTC"""
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    def get_user_info(self):
        """Get current user information"""
        return os.getenv('USER', 'unknown')

    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.fps_update_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.fps_update_time)
            self.frame_count = 0
            self.fps_update_time = current_time

        # Store frame processing time
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return avg_frame_time
        return 0

    def process_frame(self):
        """Process a single frame with performance monitoring"""
        try:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                return None

            # Detect objects
            detections = self.detect_objects(frame)
            
            # Update tracker
            if detections:
                tracks = self.tracker.update(
                    np.array([d['bbox'] for d in detections]),
                    np.array([d['confidence'] for d in detections]),
                    np.array([d['class_id'] for d in detections])
                )
            else:
                tracks = []

            # Count objects
            self.count_tracks(tracks)
            
            # Update performance metrics
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            self.update_fps()
            
            # Draw information
            self.draw_info(frame)

            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'current_tracks': len(tracks),
                'fps': self.fps,
                'frame_time': frame_time,
                'frame': frame
            }

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def draw_info(self, frame):
        """Draw ROI and performance information"""
        # Draw ROI
        cv2.rectangle(frame, 
                     (self.count_roi[0], self.count_roi[1]),
                     (self.count_roi[0] + self.count_roi[2], 
                      self.count_roi[1] + self.count_roi[3]),
                     (0, 255, 0), 2)

        # Draw counts and performance metrics
        info_text = [
            f"FPS: {self.fps:.1f}",
            f"Up->Down: {self.up_down_count}",
            f"Down->Up: {self.down_up_count}",
            f"Time: {self.get_current_time()}"
        ]

        y_position = 30
        for text in info_text:
            cv2.putText(frame, text, 
                       (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            y_position += 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="./models/yolov5n.onnx", 
                        help="Path to YOLO ONNX model")
    args = parser.parse_args()

    try:
        detector_counter = ObjectDetectionCounter(
            camera_id=args.camera,
            model_path=args.model
        )
        
        print("Starting detection and counting...")
        last_stats_time = time.time()
        stats_interval = 5.0  # Print detailed stats every 5 seconds
        
        while True:
            result = detector_counter.process_frame()
            if result:
                # Display the frame
                cv2.imshow('Counter', result['frame'])
                
                # Print real-time stats
                print(f"\rFPS: {result['fps']:.1f} | "
                      f"Up->Down: {result['up_down_count']} | "
                      f"Down->Up: {result['down_up_count']} | "
                      f"Tracks: {result['current_tracks']} | "
                      f"Frame Time: {result['frame_time']*1000:.1f}ms", 
                      end='')
                
                # Print detailed stats periodically
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    print(f"\n\n--- Performance Statistics ---")
                    print(f"DateTime (UTC): {detector_counter.get_current_time()}")
                    print(f"User: {detector_counter.get_user_info()}")
                    print(f"Average FPS: {result['fps']:.1f}")
                    if len(detector_counter.frame_times) > 0:
                        avg_time = sum(detector_counter.frame_times) / len(detector_counter.frame_times)
                        print(f"Average frame time: {avg_time*1000:.1f}ms")
                    print("---------------------------\n")
                    last_stats_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.001)  # Minimal delay

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        cv2.destroyAllWindows()
        if hasattr(detector_counter, 'cap'):
            detector_counter.cap.release()

if __name__ == "__main__":
    main()
