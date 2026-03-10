import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
MODEL_PATH = r"D:\NUST\Task2.3\object_detector.keras"
IMAGE_PATH = r"D:\NUST\Task2.3\raw images\cd (32).jpg"  # input image cd 12 40 cp 12 41 p6
CLASS_NAMES = ['dog', 'car', 'person']  # same order as training

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img, original

# ===============================
# INFERENCE
# ===============================
def predict(image_path):
    img_input, original_img = preprocess_image(image_path)
    bbox_pred, class_pred = model.predict(img_input, verbose=0)

    bbox = bbox_pred[0]
    class_probs = class_pred[0]

    # -------- Temperature Scaling to increase confidence naturally --------
    T = 0.6
    class_probs = np.exp(np.log(class_probs + 1e-8) / T)
    class_probs = class_probs / np.sum(class_probs)

    # -------- Class Prediction --------
    class_id = np.argmax(class_probs)
    confidence = class_probs[class_id]
    label = CLASS_NAMES[class_id]

    # -------- Convert bbox to pixels --------
    h, w = original_img.shape[:2]
    x_center, y_center, bw, bh = bbox
    x_center *= w
    y_center *= h
    bw *= w
    bh *= h
    xmin = int(x_center - bw / 2)
    ymin = int(y_center - bh / 2)
    xmax = int(x_center + bw / 2)
    ymax = int(y_center + bh / 2)

    return original_img, (xmin, ymin, xmax, ymax), label, confidence

# ===============================
# VISUALIZATION
# ===============================
def visualize(image, bbox, label, confidence):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(image, text, (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# ===============================
# RUN
# ===============================
img, bbox, label, conf = predict(IMAGE_PATH)
print(f"\nPrediction → Class: {label}, Confidence: {conf:.3f}, BBox: {bbox}")
visualize(img, bbox, label, conf)