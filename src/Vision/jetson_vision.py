import numpy as np
import cv2
import os
import time

class JetsonVision():
    cwd = os.getcwd()

    # OBJECT DETECTION MODEL
    PATH_TO_MODEL = cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt"
    PATH_TO_LABELS = cwd+"/models/ssl_labels.txt"

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")

    def __init__(
        self,
        vertical_lines_offset = 320,
        vertical_lines_nr = 1,
        model_path=PATH_TO_MODEL, 
        labels_path=PATH_TO_LABELS, 
        score_threshold = 0.5,
        draw = False,
        camera_matrix = camera_matrix,
        camera_initial_position=calibration_position,
        points2d_path = PATH_TO_2D_POINTS,
        points3d_path = PATH_TO_3D_POINTS,
        debug = False,
        enable_field_detection = True,
        enable_randomized_observations = False,
        min_wall_length = 10   
        ):

        try:
            self.object_detector = DetectNet(
                model_path=model_path,
                labels_path=labels_path,
                score_threshold=score_threshold,
                draw=draw
                )
            self.object_detector.loadModel()
            self.has_object_detection = True
        except:
            print("TensorRT not available, not running object detection!")
            self.has_object_detection = False
        
        self.jetson_cam = Camera(
            camera_matrix=camera_matrix,
            camera_initial_position=camera_initial_position
            )
        points2d = np.loadtxt(points2d_path, dtype="float64")
        points3d = np.loadtxt(points3d_path, dtype="float64")
        self.jetson_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)
        
        self.field_detector = FieldDetection(
            vertical_lines_offset = vertical_lines_offset,
            vertical_lines_nr = vertical_lines_nr,
            min_wall_length = min_wall_length
            )
        self.enable_randomized_observations = enable_randomized_observations
        
        self.field = Field()
        self.field.redefineFieldLimits(x_max=4.2, y_max=3, x_min=-0.3, y_min=-3)
        self.current_frame = Frame()
        self.tracked_ball = Ball()
        self.tracked_goal = Goal()
        self.tracked_robot = Robot()

        self.debug_mode = debug
        self.has_field_detection = enable_field_detection

    def trackBall(self, score, xmin, xmax, ymin, ymax):
        # UPDATES BALL BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.ballAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        ball = self.current_frame.updateBall(x, y, score)
        return ball
    
    def trackGoal(self, score, xmin, xmax, ymin, ymax):
        # UPDATES GOAL BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.goalAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        goal = self.current_frame.updateGoalCenter(x, y, score)
        return goal

    def trackRobot(self, score, xmin, xmax, ymin, ymax):
        # UPDATES ROBOT BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.robotAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        robot = self.current_frame.updateRobot(x, y, score)
        return robot

    def updateObjectTracking(self, detection):
        """
        Detection ID's:
        0: background
        1: ball
        2: goal
        3: robot

        Labels are available at: ssl-detector/models/ssl_labels.txt
        """
        if self.field_detector.isOutOfField(detection):
            return

        class_id, score, xmin, xmax, ymin, ymax = detection
        if class_id == 1:
            ball = self.trackBall(score, xmin, xmax, ymin, ymax)
            self.tracked_ball = ball
        elif class_id == 2:
            goal = self.trackGoal(score, xmin, xmax, ymin, ymax)
            self.tracked_goal = goal
        elif class_id == 3:
            robot = self.trackRobot(score, xmin, xmax, ymin, ymax)
            self.tracked_robot = robot
        
    def detectAndTrackObjects(self, src):
        detections = self.object_detector.inference(src).detections
        for detection in detections:
            self.updateObjectTracking(detection)

    def detectObjects(self, src):
        return self.object_detector.inference(src).detections

    def trackObjects(self, detections):
        for detection in detections:
            self.updateObjectTracking(detection)

    def detectAndTrackFieldPoints(self, src, detected_objects):
        if self.enable_randomized_observations: 
            self.field_detector.arrangeVerticalLinesRandom(img_width=src.shape[1], detections=detected_objects)

        if self.has_object_detection: # if running on jetson, use optimized version
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundaryMerged(src)
        else:
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundary(src)
        
        boundary_ground_points = self.trackGroundPoints(src, boundary_points)
        
        # not using field lines detection
        #line_ground_points = self.trackGroundPoints(src, line_points)
        line_ground_points = []
        
        return boundary_ground_points, line_ground_points

    def detectFieldPoints(self, src, detected_objects):
        if self.enable_randomized_observations: 
            self.field_detector.arrangeVerticalLinesRandom(img_width=src.shape[1], detections=detected_objects)

        if self.has_object_detection: # if running on jetson, use optimized version
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundaryMerged(src)
        else:
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundary(src)
        
        self.field_detector.updateMask(boundary_points)
        return boundary_points, line_points
    
    def trackFieldPoints(self, src, boundary_points, line_points):
        boundary_ground_points = self.trackGroundPoints(src, boundary_points)
        #boundary_ground_points = []
        
        # not using field lines detection
        #line_ground_points = self.trackGroundPoints(src, line_points)
        line_ground_points = []
        
        return boundary_ground_points, line_ground_points
    def checkGroundPointValidity(self, dist, theta):
        if dist>8 or abs(theta)>40: 
            return False
        else: return True
        
    def trackGroundPoints(self, src, points):
        ground_points = []
        for point in points:
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            if self.debug_mode:
                src[pixel_y, pixel_x] = self.field_detector.RED
                cv2.drawMarker(src, (pixel_x, pixel_y), color=self.field_detector.RED)
            x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
            dist, theta = self.jetson_cam.xyToPolarCoordinates(x, y)
            if self.checkGroundPointValidity(dist, theta):
                ground_points.append([dist, theta])

        return ground_points 

    def process(self, src, timestamp):
        """
        Detects and tracks objects, field lines and boundary points

        Params:
        src: image source (camera frame)
        timestamp: current timestamp
        -----------------
        Returns:
        current_frame: current frame containing flags to check for objects' detection
        tracked_ball: ball position from tracking
        tracked_goal: goal center position from tracking
        tracked_robot: robot position from tracking
        particle_filter_observations: observations used for self-localization algorithm
        """
        # init
        self.current_frame = Frame(timestamp=timestamp, input_source=src)
        detections, boundary_points, line_points = [], [], []

        # CNN-based (SSD MobileNetv2) object detection ~30ms
        if self.has_object_detection:
            detections = self.detectObjects(self.current_frame.input)

        # 42ms with field lines detection, 8~9ms without it
        if self.has_field_detection:
            boundary_points, line_points = self.detectFieldPoints(self.current_frame.input, detections)            

        # remove out-of-field objects and compute relative positions 
        self.trackObjects(detections)        

        # compute field points relative positions
        particle_filter_observations = self.trackFieldPoints(src, boundary_points, line_points)

        processed_vision = self.current_frame, self.tracked_ball, self.tracked_goal, self.tracked_robot, particle_filter_observations

        return processed_vision

    def process_from_log(self, src, timestamp, has_goal, goal_bounding_box):
        """
        Detects and tracks objects, field lines and boundary points

        Params:
        src: image source (camera frame)
        timestamp: current timestamp
        -----------------
        Returns:
        current_frame: current frame containing flags to check for objects' detection
        tracked_ball: ball position from tracking
        tracked_goal: goal center position from tracking
        tracked_robot: robot position from tracking
        particle_filter_observations: observations used for self-localization algorithm
        """
        # init
        self.current_frame = Frame(timestamp=timestamp, input_source=src)
        detections, boundary_points, line_points = [], [], []

        # CNN-based (SSD MobileNetv2) object detection ~30ms
        if self.has_object_detection:
            detections = self.detectObjects(self.current_frame.input)
        else:
            class_id, score, xmin, xmax, ymin, ymax = 2, has_goal, goal_bounding_box[0], goal_bounding_box[1], goal_bounding_box[2], goal_bounding_box[3]
            detection = [class_id, score, xmin, xmax, ymin, ymax]
            if score>0.5:
                detections.append(detection)

        # 42ms with field lines detection, 8~9ms without it
        if self.has_field_detection:
            boundary_points, line_points = self.detectFieldPoints(self.current_frame.input, detections)            

        # remove out-of-field objects and compute relative positions 
        self.trackObjects(detections)        

        # compute field points relative positions
        boundary_ground_points, line_ground_points = self.trackFieldPoints(src, boundary_points, line_points)
        
        # parse field detections for particle filter observations
        particle_filter_observations = has_goal, boundary_ground_points, line_ground_points

        processed_vision = self.current_frame, self.tracked_ball, self.tracked_goal, self.tracked_robot, particle_filter_observations

        return processed_vision

if __name__ == "__main__":
    from entities import Ball, Robot, Goal, Field, Frame
    from field_detection import FieldDetection
    from camera_transformation import Camera
    try:
        from object_detection import DetectNet
    except Exception as e:
        print(f"Import failed due to: {e} => RUNNING WITHOUT OBJECTS DETECTION!")
    import time

    def get_image_from_frame_nr(path_to_images_folder, frame_nr):
        dir = path_to_images_folder+f'/cam/{frame_nr}.png'
        img = cv2.imread(dir)
        return img

    cwd = os.getcwd()

    frame_nr = 50
    scenario = 'igs'
    lap = 1

    vision = JetsonVision(vertical_lines_offset=100,
                          vertical_lines_nr=1,
                          enable_randomized_observations=True,
                          debug=True,
                          min_wall_length=5)
    vision.jetson_cam.setPoseFrom3DModel(170, 106.7)

    while True:
        dir = f'/home/rc-blackout/ssl-navigation-dataset/data/{scenario}_0{lap}'

        WINDOW_NAME = "BOUNDARY DETECTION"
        img = get_image_from_frame_nr(dir, frame_nr)
        height, width = img.shape[0], img.shape[1]
        _, _, _, _, particle_filter_observations = vision.process(img, timestamp=time.time())
        boundary_ground_points, line_ground_points = particle_filter_observations
        for point in boundary_ground_points:
            point = vision.jetson_cam.xyToPolarCoordinates(point[0], point[1])
            print(point)
        cv2.imshow(WINDOW_NAME, img[:, :])
        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(WINDOW_NAME + '.png', img[:, :])
        else:
            frame_nr=frame_nr+1

else:
    from Vision.entities import Ball, Robot, Goal, Field, Frame
    from Vision.field_detection import FieldDetection
    from Vision.camera_transformation import Camera
    try:
        from object_detection import DetectNet
    except Exception as e:
        print(f"Import failed due to: {e} => RUNNING WITHOUT OBJECTS DETECTION!")