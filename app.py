import os
import numpy as np
import cv2
import mediapipe as mp
import math
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import tensorflow as tf
import json
import datetime

app = Flask(__name__)

# Ensure results directory exists
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the pre-trained Keras classifier model (onp top of extracted features)
model = load_model(r'model\model.h5')

# Initialize ResNet50 base + Flatten layer as feature extractor (same as training)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

# Initialize MediaPipe Face Mesh for landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3)

known_interocular_distance_cm = 6.3


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def extract_features(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    image_array = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        print("Error loading image")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            left_eye = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
            right_eye = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))
            left_brow = (int(face_landmarks.landmark[70].x * w), int(face_landmarks.landmark[70].y * h))
            right_brow = (int(face_landmarks.landmark[300].x * w), int(face_landmarks.landmark[300].y * h))
            nose_tip = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
            medial_canthus_left = (int(face_landmarks.landmark[133].x * w), int(face_landmarks.landmark[133].y * h))
            medial_canthus_right = (int(face_landmarks.landmark[362].x * w), int(face_landmarks.landmark[362].y * h))

            lateral_canthus_left = (int(face_landmarks.landmark[130].x * w), int(face_landmarks.landmark[130].y * h))
            lateral_canthus_right = (int(face_landmarks.landmark[359].x * w), int(face_landmarks.landmark[359].y * h))

            alar_left = (int(face_landmarks.landmark[2].x * w), int(face_landmarks.landmark[2].y * h))
            alar_right = (int(face_landmarks.landmark[98].x * w), int(face_landmarks.landmark[98].y * h))

            interocular_distance_px = calculate_distance(lateral_canthus_left, lateral_canthus_right)
            if interocular_distance_px == 0:
                print("Interocular distance is zero, invalid landmarks")
                return None, None

            px_to_cm_ratio = known_interocular_distance_cm / interocular_distance_px

            features = {
                "brow_lid_margin_distance_left": calculate_distance(left_brow, left_eye) * px_to_cm_ratio,
                "brow_pupil_distance_left": calculate_distance(left_brow, nose_tip) * px_to_cm_ratio,
                "lid_margin_pupil_distance_left": calculate_distance(left_eye, nose_tip) * px_to_cm_ratio,
                "brow_lateral_canthal_distance_left": calculate_distance(left_brow, lateral_canthus_left) * px_to_cm_ratio,
                "canthal_nasal_alar_distance_left": calculate_distance(lateral_canthus_left, alar_left) * px_to_cm_ratio,
                "brow_alar_distance_left": calculate_distance(left_brow, alar_left) * px_to_cm_ratio,
                "brow_medial_canthal_distance_left": calculate_distance(left_brow, medial_canthus_left) * px_to_cm_ratio,
                "brow_lid_margin_distance_right": calculate_distance(right_brow, right_eye) * px_to_cm_ratio,
                "brow_pupil_distance_right": calculate_distance(right_brow, nose_tip) * px_to_cm_ratio,
                "lid_margin_pupil_distance_right": calculate_distance(right_eye, nose_tip) * px_to_cm_ratio,
                "brow_lateral_canthal_distance_right": calculate_distance(right_brow, lateral_canthus_right) * px_to_cm_ratio,
                "canthal_nasal_alar_distance_right": calculate_distance(lateral_canthus_right, alar_right) * px_to_cm_ratio,
                "brow_alar_distance_right": calculate_distance(right_brow, alar_right) * px_to_cm_ratio,
                "brow_medial_canthal_distance_right": calculate_distance(right_brow, medial_canthus_right) * px_to_cm_ratio,
            }

            return features, image
    else:
        return None, None


def save_result_to_file(label, features, image_filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"result_{timestamp}.json"
    json_filepath = os.path.join(RESULTS_DIR, json_filename)

    data = {
        "label": label,
        "features": features,
        "image_filename": image_filename
    }
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)

    return json_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    features = None
    result_link = None
    download_link = None
    uploaded_image_filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No file uploaded"
            return render_template('index.html', error=error)

        image_file = request.files['image']
        if image_file.filename == '':
            error = "No selected file"
            return render_template('index.html', error=error)

        try:
            # Save uploaded image to RESULTS_DIR to keep it along with JSON
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(image_file.filename)[1]  # keep original extension
            uploaded_image_filename = f"upload_{timestamp}{ext}"
            image_path = os.path.join(RESULTS_DIR, uploaded_image_filename)
            image_file.save(image_path)

            # Load image for feature extraction
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            features, image_for_model = extract_features(image_np)

            if features is None or image_for_model is None:
                error = "Could not extract features from image"
                return render_template('index.html', error=error)

            resized = cv2.resize(image_for_model, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_array = np.expand_dims(rgb, axis=0).astype(np.float32)
            input_array = preprocess_input(input_array)

            with tf.device('/CPU:0'):
                features_vector = feature_extractor.predict(input_array)
                prediction = model.predict(features_vector)

            result = "Symmetric" if prediction[0][0] > 0.5 else "Asymmetric"

            # Save the results JSON file and get filename
            json_filename = save_result_to_file(result, features, uploaded_image_filename)
            download_link = url_for('download_result', filename=json_filename)

            # Prepare URL for the result page (passing label and features in query)
            result_link = url_for("result", label=result, **features)
            print(f"Generated result_link: {result_link}")

        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"Exception occurred: {error}")

    return render_template('index.html', result=result, error=error, result_link=result_link, download_link=download_link)


@app.route("/result", methods=["GET"])
def result():
    label = request.args.get("label")
    features = {key: float(value) for key, value in request.args.items() if key != "label"}

    formatted_features = {}
    adjustments = {}

    for key, value in features.items():
        formatted_features[key] = f"{value:.2f} cm"
        if label == "Asymmetric":
            if "left" in key:
                adjustments[key] = f"(Adjustment: {value + 0.1:.2f} cm)"
            elif "right" in key:
                adjustments[key] = f"(Adjustment: {value - 0.1:.2f} cm)"

    return render_template(
        "result.html",
        label=label,
        features=formatted_features,
        adjustments=adjustments if label == "Asymmetric" else None
    )


@app.route('/results/<filename>')
def download_result(filename):
    # Send the JSON file for download
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=80, threaded=False)
