import machine
import time
import camera  # MicroPython camera module
import tflite  # TFLite for MicroPython
from collections import deque
import gc

class MicroObjectCounter:
    def __init__(self):
        print("=" * 40)
        print(f"Current Date and Time (UTC): 2025-01-29 08:33:07")
        print(f"Current User's Login: nhattan86")
        print("Object Counter Initialization")
        print("=" * 40)

        # Initialize hardware
        self.setup_camera()
        self.setup_model()
        
        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = deque((), 20)  # Limited size for memory
        self.up_track_ids = deque((), 20)
        
        # Detection parameters
        self.conf_threshold = 0.3
        self.iou_threshold = 0.45
        self.input_size = (320, 320)  # Reduced for MCU
        
        # Performance monitoring
        self.frame_times = deque((), 10)  # Store last 10 frames
        self.fps = 0
        self.fps_timer = machine.Timer(-1)
        self.frame_count = 0
        self.last_fps_update = time.ticks_ms()
        
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
            print("Camera initialized")
        except Exception as e:
            print(f"Camera init failed: {e}")
            raise

    def setup_model(self):
        """Initialize TFLite model"""
        try:
            # Load quantized model
            self.interpreter = tflite.Interpreter(
                model_path='/models/yolov8s_q.tflite'
            )
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded")
        except Exception as e:
            print(f"Model init failed: {e}")
            raise

    def setup_roi(self):
        """Setup counting region"""
        self.count_roi = [
            0,                                    # x start
            int(self.cam_height - self.cam_height / 1.8),  # y start
            self.cam_width,                       # width
            int(self.cam_height / 8.6)           # height
        ]

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.ticks_ms()
        time_diff = time.ticks_diff(current_time, self.last_fps_update)
        
        if time_diff >= 1000:  # Update every second
            self.fps = (self.frame_count * 1000) / time_diff
            self.frame_count = 0
            self.last_fps_update = current_time
            
            # Memory management
            gc.collect()
            print(f"Free memory: {gc.mem_free()} bytes")

    def process_frame(self):
        """Process a single frame with performance monitoring"""
        try:
            frame_start = time.ticks_ms()
            
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
            self.count_objects(detections)
            
            # Update performance metrics
            frame_time = time.ticks_diff(time.ticks_ms(), frame_start)
            self.frame_times.append(frame_time)
            self.update_fps()
            
            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'detections': len(detections),
                'fps': self.fps,
                'frame_time': frame_time
            }
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None

    def detect_objects(self, frame):
        """Lightweight object detection"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(frame)
            
            # Run inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data
            )
            self.interpreter.invoke()
            
            # Get outputs
            outputs = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Process detections
            return self.process_detections(outputs)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def count_objects(self, detections):
        """Update object counts"""
        for detection in detections:
            position = self.obj_in_roi(detection, self.count_roi)
            
            if position is None:
                continue
            
            # Simple tracking ID based on position
            track_id = f"{detection['bbox'][0]}_{detection['bbox'][1]}"
            
            if position > 0 and track_id not in self.down_track_ids:
                self.up_down_count += 1
                self.down_track_ids.append(track_id)
                
            if position < 0 and track_id not in self.up_track_ids:
                self.down_up_count += 1
                self.up_track_ids.append(track_id)

def main():
    try:
        counter = MicroObjectCounter()
        print("Starting counting...")
        
        while True:
            result = counter.process_frame()
            if result:
                print(
                    f"\rFPS: {result['fps']:.1f} | "
                    f"Up->Down: {result['up_down_count']} | "
                    f"Down->Up: {result['down_up_count']} | "
                    f"Objects: {result['detections']} | "
                    f"Frame Time: {result['frame_time']}ms",
                    end=''
                )
            
            # Small delay
            time.sleep_ms(10)
            
    except KeyboardInterrupt:
        print("\nProgram terminated")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        camera.deinit()
        gc.collect()

if __name__ == "__main__":
    main()
