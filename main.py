% python object_counter.py --camera 0 --model ./models/yolov8s.onnx

import cv2
import numpy as np
from bytetrack_cpp import BYTETracker  # Assuming ByteTrack is installed
import time
from datetime import datetime
import argparse
import sys

class ObjectDetectionCounter:
    def __init__(self, camera_id=0, model_path="./models/yolov8s.onnx"):
        """
        Initialize with standard components compatible with Linux systems
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize YOLO model using OpenCV's DNN
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
        self.input_size = (640, 640)  # YOLO default size

        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = []
        self.up_track_ids = []

        # Setup ROI after camera initialization
        self.setup_roi()

    def setup_roi(self):
        """Set up the region of interest (ROI) for movement counting"""
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.count_roi = [
            0,  # x start
            int(frame_height - frame_height / 1.8),  # y start
            frame_width,  # width
            int(frame_height / 8.6)  # height
        ]

    def preprocess_image(self, frame):
        """Preprocess image for YOLO model"""
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            self.input_size, 
            swapRB=True, 
            crop=False
        )
        return blob

    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        blob = self.preprocess_image(frame)
        self.detector.setInput(blob)
        outputs = self.detector.forward()
        return self.process_detections(outputs[0], frame)

    def process_detections(self, outputs, frame):
        """Process YOLO outputs into detection objects"""
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs.shape[0]
        image_width, image_height = frame.shape[1], frame.shape[0]

        x_factor = image_width / self.input_size[0]
        y_factor = image_height / self.input_size[1]

        for r in range(rows):
            row = outputs[r]
            confidence = row[4]
            
            if confidence >= self.conf_threshold:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if classes_scores[class_id] > self.conf_threshold:
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    x, y, w, h = row[0], row[1], row[2], row[3]
                    
                    left = int((x - w/2) * x_factor)
                    top = int((y - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        
        detections = []
        for i in indexes:
            box = boxes[i]
            detection = {
                'bbox': box,
                'confidence': confidences[i],
                'class_id': class_ids[i]
            }
            detections.append(detection)
            
        return detections

    def obj_in_roi(self, obj, roi):
        """Check if object is in ROI and determine its position"""
        bbox = obj['bbox']
        x = bbox[0] + bbox[2] // 2
        y = bbox[1] + bbox[3] // 2
        
        if x >= roi[0] and x <= roi[0] + roi[2]:
            if y < roi[1]:
                return -1  # Above ROI
            if y > roi[1] + roi[3]:
                return 1   # Below ROI
            return 0       # Inside ROI
        return None       # Outside ROI

    def process_frame(self):
        """Process a single frame for detection and counting"""
        try:
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
            
            # Draw ROI and counts on frame
            self.draw_info(frame)

            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'current_tracks': len(tracks),
                'frame': frame
            }

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def draw_info(self, frame):
        """Draw ROI and counting information on frame"""
        # Draw ROI
        cv2.rectangle(frame, 
                     (self.count_roi[0], self.count_roi[1]),
                     (self.count_roi[0] + self.count_roi[2], 
                      self.count_roi[1] + self.count_roi[3]),
                     (0, 255, 0), 2)

        # Draw counts
        cv2.putText(frame, f"Up->Down: {self.up_down_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Down->Up: {self.down_up_count}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        while True:
            result = detector_counter.process_frame()
            if result:
                # Display the frame
                cv2.imshow('Counter', result['frame'])
                
                # Print counts
                print(f"Up->Down Count: {result['up_down_count']}")
                print(f"Down->Up Count: {result['down_up_count']}")
                print(f"Current Tracks: {result['current_tracks']}")
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        cv2.destroyAllWindows()
        if hasattr(detector_counter, 'cap'):
            detector_counter.cap.release()

if __name__ == "__main__":
    main()
