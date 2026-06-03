from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "emotion_model.h5"
LABELS_PATH = MODEL_DIR / "emotion_labels.json"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
REPORTS_DIR = BASE_DIR / "reports"

IMAGE_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 60
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
