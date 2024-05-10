# Exercise Detection Application

This application utilizes computer vision and machine learning to detect and count exercises, specifically squats, in real-time using a webcam feed. It leverages TensorFlow, TensorFlow Hub, and OpenCV to process video frames, detect human keypoints, and calculate the number of squat repetitions.

## Features

- Real-time squat detection and counting
- Visualization of detected keypoints and skeletal structure on the video feed
- Utilizes MoveNet from TensorFlow Hub for efficient and accurate pose estimation
- Face recognition and exercise mapping

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- TensorFlow 2.x
- TensorFlow Hub
- OpenCV-Python
- NumPy

## Installation

To install the necessary libraries, run the following command:
```sh
pip3 install tensorflow tensorflow-hub opencv-python numpy os
```

## Usage
To start the exercise detection application, run:
```sh
python3 app.py
```

Ensure your webcam is connected and properly configured. The application window will display the live video feed with detected squats and a count of the repetitions.

Press 'q' to quit the application.
