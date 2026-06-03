import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import EMOTION_LABELS, LABELS_PATH


def load_emotion_model(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        from config import IMAGE_SIZE
        from train_model import build_fer_cnn

        labels = load_labels()
        model = build_fer_cnn(num_classes=len(labels), image_size=IMAGE_SIZE)
        model.load_weights(model_path)
        return model


def load_labels(labels_path=LABELS_PATH):
    labels_path = Path(labels_path)
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return EMOTION_LABELS


def model_input_details(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1] or 48
    width = input_shape[2] or height
    channels = input_shape[3] or 1
    return int(height), int(width), int(channels)


def preprocess_face(face_bgr, model):
    height, width, channels = model_input_details(model)

    if channels == 1:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height))
        gray = cv2.equalizeHist(gray)
        arr = gray.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)
    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height))
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype("float32"))

    return np.expand_dims(arr, axis=0)


def padded_crop(frame, x, y, w, h, padding=0.20):
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, frame.shape[1])
    y2 = min(y + h + pad_y, frame.shape[0])
    return frame[y1:y2, x1:x2]
