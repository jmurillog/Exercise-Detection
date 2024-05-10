import cv2
import numpy as np
import os

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

            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            # Detect face
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

# Function to predict using live webcam feed
def predict_live():
    cap = cv2.VideoCapture()  # Start the webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+w, x:x+h]
            label, confidence = face_recognizer.predict(roi_gray)
            print(f"Detected face with label {label} and confidence {confidence}")  # Log the confidence
            
            if confidence < 68:  # If very sure, display the label
                person_name = next((name for name, idx in labels_dict.items() if idx == label), "Unknown")
            else:
                person_name = "Unknown"
            label_text = f"{person_name} ({confidence:.2f})"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        cv2.imshow('Live Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the live prediction
predict_live()