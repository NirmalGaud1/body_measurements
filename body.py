import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")

# Function to calculate measurements
def calculate_measurements(image, pixel_to_cm_ratio):
    # Perform inference
    results = model(image)
    keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Extract keypoints

    # Keypoint indices (COCO format)
    head_top = keypoints[0]  # Top of the head
    left_ankle = keypoints[15]  # Left ankle
    right_ankle = keypoints[16]  # Right ankle
    left_shoulder = keypoints[5]  # Left shoulder
    right_shoulder = keypoints[6]  # Right shoulder
    left_hip = keypoints[11]  # Left hip
    right_hip = keypoints[12]  # Right hip

    # Calculate height (distance from head top to ankles)
    height_pixels = np.linalg.norm(head_top - (left_ankle + right_ankle) / 2)

    # Calculate shoulder width (distance between left and right shoulders)
    shoulder_width_pixels = np.linalg.norm(left_shoulder - right_shoulder)

    # Calculate hip width (distance between left and right hips)
    hip_width_pixels = np.linalg.norm(left_hip - right_hip)

    # Convert all measurements to centimeters
    height_cm = height_pixels * pixel_to_cm_ratio
    shoulder_width_cm = shoulder_width_pixels * pixel_to_cm_ratio
    hip_width_cm = hip_width_pixels * pixel_to_cm_ratio

    # For circumferences (waist), use contour detection
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Example: Calculate waist circumference (assuming contour at waist level)
    waist_contour = max(contours, key=cv2.contourArea)  # Use the largest contour
    waist_circumference_pixels = cv2.arcLength(waist_contour, True)
    waist_circumference_cm = waist_circumference_pixels * pixel_to_cm_ratio

    return {
        "Height (cm)": height_cm,
        "Shoulder Width (cm)": shoulder_width_cm,
        "Hip Width (cm)": hip_width_cm,
        "Waist Circumference (cm)": waist_circumference_cm,
    }

# Streamlit app
st.title("Body Measurement App")
st.write("Upload an image or take a picture using your webcam to get body measurements.")

# Option 1: Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Option 2: Take a picture from webcam
webcam_image = st.camera_input("Take a picture using your webcam")

# Reference object input
reference_object_width_cm = st.number_input("Enter the width of the reference object (in cm)", min_value=1.0, value=10.0)
reference_object_width_pixels = st.number_input("Enter the width of the reference object (in pixels)", min_value=1, value=100)

# Calculate pixel-to-cm ratio
pixel_to_cm_ratio = reference_object_width_cm / reference_object_width_pixels

# Process the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    measurements = calculate_measurements(image, pixel_to_cm_ratio)
elif webcam_image is not None:
    image = Image.open(webcam_image)
    st.image(image, caption="Webcam Image", use_column_width=True)
    measurements = calculate_measurements(image, pixel_to_cm_ratio)
else:
    measurements = None

# Display measurements
if measurements:
    st.write("### Body Measurements")
    for key, value in measurements.items():
        st.write(f"{key}: {value:.2f} cm")
