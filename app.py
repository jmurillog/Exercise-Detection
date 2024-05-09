import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import csv

BODY_PARTS = [
    (5, 7),  # Left shoulder to left elbow
    (7, 9),  # Left elbow to left wrist
    (6, 8),  # Right shoulder to right elbow
    (8, 10), # Right elbow to right wrist
    (5, 6),  # Left shoulder to right shoulder
    (5, 11), # Left shoulder to left hip
    (6, 12), # Right shoulder to right hip
    (11, 12),# Left hip to right hip
    (11, 13),# Left hip to left knee
    (13, 15),# Left knee to left ankle
    (12, 14),# Right hip to right knee
    (14, 16) # Right knee to right ankle
]

# Load known faces
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


faces = []
labels = []
names = {}
for person_id, person_name in enumerate(os.listdir("known_faces")):
    if os.path.isdir(os.path.join("known_faces", person_name)):
        for image_name in os.listdir(f"known_faces/{person_name}"):
            image = cv2.imread(f"known_faces/{person_name}/{image_name}", cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read image: known_faces/{person_name}/{image_name}")
                continue
            image = cv2.resize(image, (200, 200))
            faces.append(image)
            labels.append(person_id)
        names[person_id] = person_name


face_recognizer.train(faces, np.array(labels))

def recognize_faces(frame, confidence_threshold=100):  # Adjusted threshold
    """Recognize faces in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=4, minSize=(30, 30))

    face_names = []
    for (x, y, w, h) in face_locations:
        face = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)
        if confidence < confidence_threshold:
            name = names[label]
        else:
            name = "Unknown"
        face_names.append(name)

    return face_locations, face_names


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

    squat_down_threshold = 90  # Adjust as needed for a full squat
    squat_up_threshold = 160   # Adjust as needed for standing up

    # Detect down phase
    if left_angle < squat_down_threshold and right_angle < squat_down_threshold:
        state['down'] = True

    # Detect up phase and count repetition
    if state['down'] and left_angle > squat_up_threshold and right_angle > squat_up_threshold:
        state['count'] += 1
        state['down'] = False

    return state['count']

def display_labels(frame, exercise_records):
    base_x, base_y = 10, 10  # Starting coordinates for the first label
    label_height = 60  # Height of each label box to fit text

    # Define a list of colors for different individuals
    colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (50, 200, 200)]

    for index, (name, records) in enumerate(exercise_records.items()):
        # Position for the current label, shift down based on how many are already displayed
        current_y = base_y + index * (label_height + 10)
        label_text = f"{name}: Squats {records['Squats']}, Press {records['Press']}"

        # Determine the width of the box based on text width
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (base_x, current_y), (base_x + text_width + 10, current_y + label_height), colors[index % len(colors)], -1)
        cv2.putText(frame, label_text, (base_x + 5, current_y + text_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_faces_and_labels(frame, face_locations, face_names):
    """Draw rectangles and labels around recognized faces."""
    for (x, y, w, h), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Put the name above the rectangle

def write_results_to_csv(exercise_records):
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Person', 'Exercise', 'Count'])  # Write the header row
        for person, records in exercise_records.items():
            for exercise, count in records.items():
                writer.writerow([person, exercise, count])  # Write the exercise data




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

        face_locations, face_names = recognize_faces(frame)  # Recognize faces
        keypoints = detect_keypoints(model, frame)  # Detect keypoints
        draw_keypoints_and_lines(frame, keypoints)  # Draw keypoints and lines

        draw_faces_and_labels(frame, face_locations, face_names)  # Draw faces and labels

        # Update and manage exercise records
        for name in face_names:
            if name not in exercise_records:
                exercise_records[name] = {'Squats': 0, 'Press': 0}
                state_squats[name] = {'count': 0, 'down': False}  # Ensure states are initialized for new faces
                state_press[name] = {'count': 0, 'up': False}
            exercise_records[name]['Squats'] = detect_squat(keypoints, state_squats[name])  # Update squat count
            exercise_records[name]['Press'] = detect_military_press(keypoints, state_press[name])  # Update press count

        # Display the updated labels in the top left corner
        display_labels(frame, exercise_records)

        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    # Print final exercise counts
    print("Exercise session results:\n")
    for person, records in exercise_records.items():
        print(f"\n{person}:")
        for exercise, count in records.items():
            print(f" - {exercise}: {count}")
    # Place this at the end of your main try block or after the session ends
    write_results_to_csv(exercise_records)