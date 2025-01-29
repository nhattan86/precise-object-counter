from maix import nn, camera, tracker, time, app
import time
from datetime import datetime
import collections

class ObjectDetectionCounter:
    def __init__(self):
        print(f"Initializing Object Detection Counter...")
        print(f"Started by: {self.get_user_info()}")
        print(f"Start time: {self.get_current_time()}")
        
        # Initialize detector and camera
        self.detector = nn.YOLO11(model="/root/models/yolo11n.mud", dual_buff=True)
        self.cam = camera.Camera(
            self.detector.input_width(), 
            self.detector.input_height(), 
            self.detector.input_format()
        )
        
        # Initialize tracker
        self.tracker = tracker.ByteTracker(1, 0.4, 0.6, 0.8, 18)
        
        # Detection thresholds
        self.conf_threshold = 0.3
        self.iou_threshold = 0.45
        
        # Initialize counters
        self.up_down_count = 0
        self.down_up_count = 0
        self.down_track_ids = []
        self.up_track_ids = []
        
        # FPS calculation variables
        self.fps_start_time = time.ticks_ms()
        self.fps = 0
        self.frame_count = 0
        self.fps_update_interval = 1000  # Update FPS every 1 second
        
        # Frame time tracking for detailed performance analysis
        self.frame_times = collections.deque(maxlen=100)  # Keep last 100 frame times
        
        # Set up ROI for counting
        self.setup_roi()

    def get_user_info(self):
        """Get current user information"""
        return "nhattan86"  # You can modify this to get actual system user

    def get_current_time(self):
        """Get current time in UTC"""
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    def setup_roi(self):
        """Set up the region of interest (ROI) for movement counting"""
        self.count_roi = [
            0,  # x start
            int(self.cam.height() - self.cam.height() / 1.8),  # y start
            self.cam.width(),  # width
            int(self.cam.height() / 8.6)  # height
        ]

    def obj_in_roi(self, obj, roi):
        """Check if object is in ROI and determine its position"""
        x = obj.x + obj.w // 2
        y = obj.y + obj.h // 2
        if x >= roi[0] and x <= roi[0] + roi[2]:
            if y < roi[1]:
                return -1  # Above ROI
            if y > roi[1] + roi[3]:
                return 1   # Below ROI
            return 0       # Inside ROI
        return None       # Outside ROI

    def yolo_objs_to_tracker_objs(self, objs, valid_class_id=[0]):
        """Convert YOLO detected objects to tracker objects"""
        return [
            tracker.Object(obj.x, obj.y, obj.w, obj.h, obj.class_id, obj.score) 
            for obj in objs if len(valid_class_id) == 0 or obj.class_id in valid_class_id
        ]

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.ticks_ms()
        elapsed_time = time.ticks_diff(current_time, self.fps_start_time)
        
        # Store frame processing time
        self.frame_times.append(elapsed_time)
        
        # Update FPS every second
        if elapsed_time >= self.fps_update_interval:
            self.fps = self.frame_count * 1000 / elapsed_time
            self.frame_count = 0
            self.fps_start_time = current_time
            
            # Calculate average frame time
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            print(f"Average frame processing time: {avg_frame_time:.2f}ms")

    def count_tracks(self, tracks):
        """Count objects moving up and down through ROI"""
        for track in tracks:
            if track.lost:
                continue
                
            obj = track.history[-1]
            ret = self.obj_in_roi(obj, self.count_roi)
            
            if ret is None:
                continue

            # Object moving down
            if ret > 0 and track.id not in self.down_track_ids:
                for o in track.history[::-1][1:]:
                    if self.obj_in_roi(o, self.count_roi) == 0:
                        self.up_down_count += 1
                        self.down_track_ids.append(track.id)
                        break

            # Object moving up
            if ret < 0 and track.id not in self.up_track_ids:
                for o in track.history[::-1][1:]:
                    if self.obj_in_roi(o, self.count_roi) == 0:
                        self.down_up_count += 1
                        self.up_track_ids.append(track.id)
                        break

        # Maintain track ID lists
        if len(self.down_track_ids) > 1:
            self.down_track_ids = self.down_track_ids[0:]
        if len(self.up_track_ids) > 1:
            self.up_track_ids = self.up_track_ids[0:]

    def process_frame(self):
        """Process a single frame for detection and counting"""
        try:
            frame_start_time = time.ticks_ms()
            
            # Capture and detect objects
            img = self.cam.read()
            objs = self.detector.detect(
                img, 
                conf_th=self.conf_threshold, 
                iou_th=self.iou_threshold
            )
            
            # Convert to tracker objects (focusing on class_id 0)
            objs = self.yolo_objs_to_tracker_objs(objs, [0])
            
            # Update tracker and get tracks
            tracks = self.tracker.update(objs)
            
            # Count objects
            self.count_tracks(tracks)
            
            # Update FPS
            self.update_fps()
            
            # Calculate frame processing time
            frame_time = time.ticks_diff(time.ticks_ms(), frame_start_time)
            
            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'current_tracks': len(tracks),
                'fps': self.fps,
                'frame_time': frame_time,
                'image': img
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def get_performance_stats(self):
        """Get detailed performance statistics"""
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            max_frame_time = max(self.frame_times)
            min_frame_time = min(self.frame_times)
            return {
                'fps': self.fps,
                'avg_frame_time': avg_frame_time,
                'max_frame_time': max_frame_time,
                'min_frame_time': min_frame_time
            }
        return None

def main():
    detector_counter = ObjectDetectionCounter()
    
    print("Starting detection and counting...")
    last_stats_time = time.ticks_ms()
    stats_interval = 5000  # Print detailed stats every 5 seconds
    
    try:
        while not app.need_exit():
            result = detector_counter.process_frame()
            if result:
                # Print regular updates
                print(f"\rFPS: {result['fps']:.1f} | "
                      f"Up->Down: {result['up_down_count']} | "
                      f"Down->Up: {result['down_up_count']} | "
                      f"Current Tracks: {result['current_tracks']}", 
                      end='')
                
                # Print detailed stats periodically
                current_time = time.ticks_ms()
                if time.ticks_diff(current_time, last_stats_time) >= stats_interval:
                    stats = detector_counter.get_performance_stats()
                    if stats:
                        print(f"\n--- Performance Statistics ---")
                        print(f"Average FPS: {stats['fps']:.1f}")
                        print(f"Average frame time: {stats['avg_frame_time']:.2f}ms")
                        print(f"Min frame time: {stats['min_frame_time']:.2f}ms")
                        print(f"Max frame time: {stats['max_frame_time']:.2f}ms")
                        print(f"Current UTC time: {detector_counter.get_current_time()}")
                        print("---------------------------")
                    last_stats_time = current_time
                    
            time.sleep_ms(10)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        # Print final statistics
        stats = detector_counter.get_performance_stats()
        if stats:
            print("\nFinal Performance Statistics:")
            print(f"Average FPS: {stats['fps']:.1f}")
            print(f"Average frame time: {stats['avg_frame_time']:.2f}ms")
            print(f"Min frame time: {stats['min_frame_time']:.2f}ms")
            print(f"Max frame time: {stats['max_frame_time']:.2f}ms")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
