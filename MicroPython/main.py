import machine
import time
from collections import deque
import gc
import camera  # MicroPython camera module
import tflite  # TFLite for MicroPython

class MicroObjectCounter:
    def __init__(self):
        # Print initialization info
        print("=" * 40)
        print("Object Counter Initialization")
        print(f"Date: 2025-01-29 08:30:56")
        print(f"User: nhattan86")
        print("=" * 40)

        # Setup hardware
        self.setup_camera()
        self.setup_model()
        
        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = deque((), 20)  # Limited size deque
        self.up_track_ids = deque((), 20)
        
        # Detection parameters
        self.conf_threshold = 0.3
        self.input_size = (320, 320)  # Reduced size for MCU
        
        # Setup ROI
        self.setup_roi()
        
        # Enable garbage collection
        gc.enable()

    def setup_camera(self):
        """Initialize camera with optimized settings"""
        try:
            camera.init(0)
            camera.framesize(camera.FRAME_VGA)
            camera.contrast(2)
            
            self.cam_width = 640
            self.cam_height = 480
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            raise

    def setup_model(self):
        """Initialize TFLite model"""
        try:
            # Load quantized model for efficiency
            self.interpreter = tflite.Interpreter(
                model_path='/models/yolov8s_q.tflite'
            )
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise

    def setup_roi(self):
        """Setup Region of Interest"""
        self.count_roi = [
            0,                                    # x start
            int(self.cam_height - self.cam_height / 1.8),  # y start
            self.cam_width,                       # width
            int(self.cam_height / 8.6)           # height
        ]

    def preprocess_image(self, frame):
        """Simplified image preprocessing"""
        # Resize image
        if hasattr(camera, 'resize'):
            frame = camera.resize(frame, self.input_size[0], self.input_size[1])
        
        # Convert to float and normalize
        frame = frame.astype('float32') / 255.0
        return frame

    def detect_objects(self, frame):
        """Lightweight object detection"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(frame)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                input_data
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            outputs = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            return self.process_detections(outputs)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def process_detections(self, outputs):
        """Process model outputs"""
        detections = []
        
        for output in outputs:
            confidence = output[4]
            
            if confidence >= self.conf_threshold:
                detection = {
                    'bbox': [
                        int(output[0]),  # x
                        int(output[1]),  # y
                        int(output[2]),  # width
                        int(output[3])   # height
                    ],
                    'confidence': confidence,
                    'class_id': int(output[5])
                }
                detections.append(detection)
        
        return detections

    def obj_in_roi(self, obj, roi):
        """Check object position relative to ROI"""
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
        """Process a single frame"""
        try:
            # Capture frame
            frame = camera.capture()
            if not frame:
                return None
            
            # Memory check
            if gc.mem_free() < 10000:  # 10KB threshold
                gc.collect()
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Update counts
            self.update_counts(detections)
            
            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'detections': len(detections)
            }
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None

    def update_counts(self, detections):
        """Update crossing counts"""
        for detection in detections:
            position = self.obj_in_roi(detection, self.count_roi)
            
            if position is None:
                continue
            
            # Generate simple track ID based on position
            track_id = f"{detection['bbox'][0]}_{detection['bbox'][1]}"
            
            if position > 0 and track_id not in self.down_track_ids:
                self.up_down_count += 1
                self.down_track_ids.append(track_id)
                print(f"Up->Down: {self.up_down_count}")
                
            if position < 0 and track_id not in self.up_track_ids:
                self.down_up_count += 1
                self.up_track_ids.append(track_id)
                print(f"Down->Up: {self.down_up_count}")

def main():
    try:
        counter = MicroObjectCounter()
        print("Starting object counting...")
        
        while True:
            result = counter.process_frame()
            if result:
                print(
                    f"Up->Down: {result['up_down_count']} | "
                    f"Down->Up: {result['down_up_count']} | "
                    f"Objects: {result['detections']}"
                )
            
            # Small delay to prevent watchdog timer issues
            time.sleep_ms(50)
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        camera.deinit()
        gc.collect()

if __name__ == "__main__":
    main()
