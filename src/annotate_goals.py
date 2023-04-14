import cv2
import os
import csv
from jetson_vision import JetsonVision

def replace_csv_column(file_dir, new_data):
    # Open the CSV file and create a csv.reader object
    with open(file_dir, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Create a list of rows
        rows = []
        for row in csv_reader:
            rows.append(row)

    # Replace the column you want to update with new values
    for (row, new_row) in zip(rows, new_data):
        row[11:15] = new_row

    # Open the CSV file in write mode and create a csv.writer object
    with open(file_dir, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the modified rows to the CSV file
        for row in rows:
            csv_writer.writerow(row)

def serialize_goal_to_log(has_goal, jetson_vision_goal):
    # DETECTED GOAL BOUNDING BOX
    xmin, xmax, ymin, ymax = jetson_vision_goal

    goal = [has_goal, xmin, xmax, ymin, ymax]
    return goal

def get_bbox(vision, img):
    detections = vision.object_detector.inference(img).detections
    jetson_vision_goal = None
    for detection in detections:
        class_id, score, xmin, xmax, ymin, ymax = detection
        if class_id==2:
            jetson_vision_goal = [xmin, xmax, ymin, ymax]

    return jetson_vision_goal

def get_dataset_dir(cwd, path_type, path_nr):
    dir = cwd+'/data'
    if path_type == 'SQUARE':
        dir+=f'/sqr_0{path_nr}'
    elif path_type == 'RANDOM':
        dir+=f'/rnd_0{path_nr}'
    elif path_type == 'GAME':
        dir+=f'/igs_0{path_nr}'
    else:
        dir+='/'+path_type+f'_0{path_nr}'
    return dir

if __name__ ==  "__main__":
    cwd = os.getcwd()

    # INIT EMBEDDED VISION 
    vision = JetsonVision(draw=False, enable_field_detection=False, score_threshold=0.3)

    # FRAME NR COUNTER
    frame_nr = 1

    # DATA FOR LOGS
    fields = ['HAS GOAL', 'X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX']
    dataset_dir = get_dataset_dir(cwd, 'igs', 3)

    # MAKE GOALS LIST
    goals_log = []
    goals_log.append(fields)

    while True:
        # GET FRAME
        try:
            img = cv2.imread(dataset_dir + f'/{frame_nr}.png')
            
            # DETECT OBJECT'S BOUNDING BOXES
            jetson_vision_goal = get_bbox(vision, img)
        except:
            break

        # CHECK IF GOAL WAS DETECTED
        has_goal = (jetson_vision_goal is not None)

        # UPDATE DATA AND PRINT FOR DEBUG
        if not has_goal:
            jetson_vision_goal = [0, 0, 0, 0]
        
        # SERIALIZE DATA
        goal = serialize_goal_to_log(has_goal, jetson_vision_goal)
        print(f'FRAME: {frame_nr} | GOAL DETECTION: {goal}')
        goals_log.append(goal)

        # ADD FRAME NR
        frame_nr += 1

    # REPLACE GOALS DATA
    log_dir = dataset_dir + '/log.csv'
    replace_csv_column(log_dir, goals_log)
    print('FINISH!')
