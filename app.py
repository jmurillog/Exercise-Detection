import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import csv

BODY_PARTS = [
    (5, 7),    # Left shoulder to left elbow
    (7, 9),    # Left elbow to left wrist
    (6, 8),    # Right shoulder to right elbow
    (8, 10),   # Right elbow to right wrist
    (5, 6),    # Left shoulder to right shoulder
    (5, 11),   # Left shoulder to left hip
    (6, 12),   # Right shoulder to right hip
    (11, 12),  # Left hip to right hip
    (11, 13),  # Left hip to left knee
    (13, 15),  # Left knee to left ankle
    (12, 14),  # Right hip to right knee
    (14, 16)   # Right knee to right ankle
]

face_recognizer = cv2.face.LBPHFaceRecognizer_create()



# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to store labels and their corresponding names
labels_dict = {}
current_label = 0

# Prepare training data from your specific folder structure
def prepare_training_data(data_folder_path):
    global current_label
    faces = []
    labels = []
    
    # Listing directories in the folder containing faces of known persons
    person_names = os.listdir(data_folder_path)
    person_names.sort()  # Optional, to keep labels consistent between runs
    
    for person_name in person_names:
        if person_name.startswith("."):
            continue  # Skip system files like .DS_Store
        
        # Assign a unique label to each person based on the folder name
        if person_name not in labels_dict:
            labels_dict[person_name] = current_label
            current_label += 1
        
        label = labels_dict[person_name]
        subject_dir_path = os.path.join(data_folder_path, person_name)
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue  # Skip system files
            
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            equalized = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            face_rects = face_cascade.detectMultiScale(blurred, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
            
            for (x, y, w, h) in face_rects:
                roi_gray = gray[y:y+w, x:x+h]
                resized_roi = cv2.resize(roi_gray, (128, 128))
                faces.append(resized_roi)
                labels.append(label)
    
    return faces, np.array(labels)

# Training data path
data_folder_path = 'known_faces'
faces, labels = prepare_training_data(data_folder_path)

# Train the recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, labels)



def load_model():
    """Load the MoveNet model from TensorFlow Hub."""
    model_url = 'https://tfhub.dev/google/movenet/singlepose/thunder/4'
    model = hub.load(model_url)
    return model.signatures['serving_default']

model = load_model()

def preprocess_image(frame):
    """Preprocess the frame before feeding it to the model."""
    img = tf.convert_to_tensor(frame, dtype=tf.uint8)
    img = tf.image.resize_with_pad(img, 256, 256)
    img = tf.cast(img, dtype=tf.int32)
    return img

def detect_keypoints(model, frame):
    """Detect keypoints on the frame."""
    img = preprocess_image(frame)
    inputs = tf.expand_dims(img, axis=0)
    outputs = model(inputs)
    keypoints = outputs['output_0'].numpy()
    return keypoints

def draw_keypoints_and_lines(frame, keypoints):
    """Draw keypoints and lines between them on the frame."""
    height, width, _ = frame.shape
    keypoints = np.squeeze(keypoints)[:, :2]

    scaled_keypoints = keypoints.copy()
    scaled_keypoints[:, 0] *= height
    scaled_keypoints[:, 1] *= width

    for point in scaled_keypoints:
        cv2.circle(frame, (int(point[1]), int(point[0])), 5, (0, 256, 0), -1)

    for part in BODY_PARTS:
        start_point = (int(scaled_keypoints[part[0], 1]), int(scaled_keypoints[part[0], 0]))
        end_point = (int(scaled_keypoints[part[1], 1]), int(scaled_keypoints[part[1], 0]))
        cv2.line(frame, start_point, end_point, (256, 0, 0), 2)

def calculate_angle(point1, point2, point3):
    """Calculate the angle at point2 formed by line segments point1-point2 and point2-point3."""
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def calculate_vertical_movement(point1, point2):
    """Calculate the vertical movement between two points."""
    return abs(point1[0] - point2[0])

def detect_military_press(keypoints, state):
    """Detect military press repetitions based on the position of the wrists and shoulders."""
    head = keypoints[0][0][0][:2]
    left_shoulder = keypoints[0][0][5][:2]
    right_shoulder = keypoints[0][0][6][:2]
    left_wrist = keypoints[0][0][9][:2]
    right_wrist = keypoints[0][0][10][:2]

    wrists_above_head = (left_wrist[0] < head[0]) and (right_wrist[0] < head[0])
    wrists_below_shoulders = (left_wrist[0] > left_shoulder[0]) and (right_wrist[0] > right_shoulder[0])

    if wrists_above_head and not state['up']:
        state['up'] = True

    if state['up'] and wrists_below_shoulders:
        state['count'] += 1
        state['up'] = False

    return state['count']

def detect_squat(keypoints, state):
    """Detect squat repetitions based on the angle between hip, knee, and ankle keypoints."""
    left_hip = keypoints[0][0][11][:2]
    right_hip = keypoints[0][0][12][:2]
    left_knee = keypoints[0][0][13][:2]
    right_knee = keypoints[0][0][14][:2]
    left_ankle = keypoints[0][0][15][:2]
    right_ankle = keypoints[0][0][16][:2]

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

    squat_down_threshold = 90 
    squat_up_threshold = 160 

    if left_angle < squat_down_threshold and right_angle < squat_down_threshold:
        state['down'] = True

    if state['down'] and left_angle > squat_up_threshold and right_angle > squat_up_threshold:
        state['count'] += 1
        state['down'] = False

    return state['count']

def display_labels(frame, exercise_records):
    base_x, base_y = 10, 10
    label_height = 60 

    colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (50, 200, 200)]

    for index, (name, records) in enumerate(exercise_records.items()):
        current_y = base_y + index * (label_height + 10)
        label_text = f"{name}: Squats {records['Squats']}, Press {records['Press']}"

        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (base_x, current_y), (base_x + text_width + 10, current_y + label_height), colors[index % len(colors)], -1)
        cv2.putText(frame, label_text, (base_x + 5, current_y + text_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_faces_and_labels(frame, face_locations, face_names):
    """Draw rectangles and labels around recognized faces."""
    for (x, y, w, h), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def write_results_to_csv(exercise_records):
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Person', 'Exercise', 'Count'])
        for person, records in exercise_records.items():
            for exercise, count in records.items():
                writer.writerow([person, exercise, count])




cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

state_squats = {'count': 0, 'down': False}
state_press = {'count': 0, 'up': False}

exercise_records = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray) 
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        faces = face_cascade.detectMultiScale(blurred, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        face_names = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+w, x:x+h]
            label, confidence = face_recognizer.predict(roi_gray)
            print(f"Detected face with label {label} and confidence {confidence}")  # Log the confidence
            if confidence < 68:  # If very sure, display the label
                person_name = next((name for name, idx in labels_dict.items() if idx == label), "Unknown")
            else:
                person_name = "Unknown"
            face_names.append(person_name)
            label_text = f"{person_name} ({confidence:.2f})"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imshow('Live Face Recognition', frame)
        keypoints = detect_keypoints(model, frame)  
        draw_keypoints_and_lines(frame, keypoints) 
        
        for name in face_names:
            if name not in exercise_records:
                exercise_records[name] = {'Squats': 0, 'Press': 0}
                state_squats[name] = {'count': 0, 'down': False}  
                state_press[name] = {'count': 0, 'up': False}
            exercise_records[name]['Squats'] = detect_squat(keypoints, state_squats[name])  
            exercise_records[name]['Press'] = detect_military_press(keypoints, state_press[name]) 

        display_labels(frame, exercise_records)

        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Exercise session results:\n")
    for person, records in exercise_records.items():
        print(f"\n{person}:")
        for exercise, count in records.items():
            print(f" - {exercise}: {count}")
    write_results_to_csv(exercise_records)