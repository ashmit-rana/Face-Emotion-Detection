import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from config import CASCADE_PATH, MODEL_PATH
from emotion_utils import load_emotion_model, load_labels, padded_crop, preprocess_face


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time facial emotion detection.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--cascade-path", default=str(CASCADE_PATH))
    parser.add_argument("--smooth-frames", type=int, default=12)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_emotion_model(args.model_path)
    emotion_labels = load_labels()

    face_cascade = cv2.CascadeClassifier(args.cascade_path)
    if face_cascade.empty():
        raise SystemExit("Cannot load Haar cascade. Check --cascade-path.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Check camera permissions or --camera.")

    prediction_history = deque(maxlen=args.smooth_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))

        for (x, y, w, h) in faces:
            face = padded_crop(frame, x, y, w, h)
            roi = preprocess_face(face, model)

            prediction = model.predict(roi, verbose=0)[0]
            prediction_history.append(prediction)
            smoothed_prediction = np.mean(prediction_history, axis=0)

            confidence = float(np.max(smoothed_prediction))
            label_index = int(np.argmax(smoothed_prediction))
            label = emotion_labels[label_index] if confidence >= args.confidence_threshold else "Uncertain"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 180, 255),
                2,
            )

        cv2.imshow("Live Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
