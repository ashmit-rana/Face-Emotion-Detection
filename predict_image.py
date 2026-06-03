import argparse

import cv2
import numpy as np
import tensorflow as tf

from config import CASCADE_PATH, MODEL_PATH
from emotion_utils import load_emotion_model, load_labels, padded_crop, preprocess_face


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion for a single image.")
    parser.add_argument("image_path")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--cascade-path", default=str(CASCADE_PATH))
    parser.add_argument("--save-annotated")
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image_path}")

    model = load_emotion_model(args.model_path)
    labels = load_labels()
    face_cascade = cv2.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise SystemExit("Cannot load Haar cascade. Check --cascade-path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80))
    if len(faces) == 0:
        raise SystemExit("No face detected in image.")

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face = padded_crop(image, x, y, w, h)
    prediction = model.predict(preprocess_face(face, model), verbose=0)[0]

    order = np.argsort(prediction)[::-1]
    for index in order:
        print(f"{labels[index]}: {prediction[index]:.2%}")

    if args.save_annotated:
        best = int(order[0])
        scale = max(w, h) / 200
        font_scale = max(0.75, scale)
        thickness = max(2, int(scale * 2))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 180, 255), thickness)
        cv2.putText(
            image,
            f"{labels[best]} {prediction[best]:.0%}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 180, 255),
            thickness,
        )
        cv2.imwrite(args.save_annotated, image)
        print(f"Saved annotated image to {args.save_annotated}")


if __name__ == "__main__":
    main()
