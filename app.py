import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

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

# Load model
def load_model():
    model_url = 'https://tfhub.dev/google/movenet/singlepose/lightning/4'
    model = hub.load(model_url)
    return model.signatures['serving_default']

model = load_model()

# Preprocess image
def preprocess_image(frame):
    img = tf.convert_to_tensor(frame, dtype=tf.uint8)
    img = tf.image.resize_with_pad(img, 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    return img

# Detect keypoints
def detect_keypoints(model, frame):
    img = preprocess_image(frame)
    inputs = tf.expand_dims(img, axis=0)
    outputs = model(inputs)
    keypoints = outputs['output_0'].numpy()
    return keypoints

# Draw keypoints and lines
def draw_keypoints_and_lines(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for point in shaped:
        cv2.circle(frame, (int(point[1]), int(point[0])), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    for part in BODY_PARTS:
        start_point = (int(shaped[part[0], 1]), int(shaped[part[0], 0]))
        end_point = (int(shaped[part[1], 1]), int(shaped[part[1], 0]))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 3)

# Detect squat and count repetitions
def detect_squat(keypoints, state):
    left_hip = keypoints[0][0][11][:2]
    right_hip = keypoints[0][0][12][:2]
    left_knee = keypoints[0][0][13][:2]
    right_knee = keypoints[0][0][14][:2]
    left_ankle = keypoints[0][0][15][:2]
    right_ankle = keypoints[0][0][16][:2]

    def calculate_angle(hip, knee, ankle):
        hip = np.array(hip)
        knee = np.array(knee)
        ankle = np.array(ankle)
        thigh = hip - knee
        shank = ankle - knee
        cosine_angle = np.dot(thigh, shank) / (np.linalg.norm(thigh) * np.linalg.norm(shank))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

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

# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

state = {'count': 0, 'down': False}  # State for counting repetitions

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        keypoints = detect_keypoints(model, frame)
        count = detect_squat(keypoints, state)  # Updated to include state

        keypoints = detect_keypoints(model, frame)
        draw_keypoints_and_lines(frame, keypoints)
        count = detect_squat(keypoints, state)

        # Display the count on a colored rectangle
        label = f"Squats: {count}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.rectangle(frame, (10, 30), (10 + label_width, 30 - label_height - 10), (255, 0, 0), -1)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()