from maix import nn, camera, tracker, time, app
import time

class ObjectDetectionCounter:
    def __init__(self):
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
        
        # Set up ROI for counting
        self.setup_roi()

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
            
            return {
                'up_down_count': self.up_down_count,
                'down_up_count': self.down_up_count,
                'current_tracks': len(tracks),
                'image': img
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

def main():
    detector_counter = ObjectDetectionCounter()
    
    print("Starting detection and counting...")
    while not app.need_exit():
        result = detector_counter.process_frame()
        if result:
            print(f"Up->Down Count: {result['up_down_count']}")
            print(f"Down->Up Count: {result['down_up_count']}")
            print(f"Current Tracks: {result['current_tracks']}")
        time.sleep_ms(10)  # Small delay to prevent CPU overuse

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
