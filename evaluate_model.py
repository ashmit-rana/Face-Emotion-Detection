import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import MODEL_PATH, REPORTS_DIR, TEST_DIR
from emotion_utils import load_emotion_model, load_labels, model_input_details


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the emotion model on the test set.")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--test-dir", default=str(TEST_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def make_generator(model, test_dir, batch_size):
    height, width, channels = model_input_details(model)
    if channels == 1:
        generator = ImageDataGenerator(rescale=1.0 / 255.0)
        color_mode = "grayscale"
    else:
        generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        color_mode = "rgb"

    return generator.flow_from_directory(
        test_dir,
        target_size=(height, width),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def save_confusion_matrix(matrix, labels, output_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Emotion Classification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model = load_emotion_model(args.model_path)
    test_data = make_generator(model, args.test_dir, args.batch_size)
    labels = load_labels()

    predictions = model.predict(test_data, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data.classes

    report = classification_report(true_classes, predicted_classes, target_names=labels, digits=4)
    matrix = confusion_matrix(true_classes, predicted_classes)

    report_path = reports_dir / "classification_report.txt"
    matrix_path = reports_dir / "confusion_matrix.png"
    report_path.write_text(report, encoding="utf-8")
    save_confusion_matrix(matrix, labels, matrix_path)

    print(report)
    print(f"Saved report to {report_path}")
    print(f"Saved confusion matrix to {matrix_path}")


if __name__ == "__main__":
    main()
