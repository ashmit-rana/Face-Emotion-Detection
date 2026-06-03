from pathlib import Path

import tensorflow as tf

from config import EMOTION_LABELS, MODEL_PATH, TEST_DIR, TRAIN_DIR
from emotion_utils import load_emotion_model, load_labels


def test_dataset_structure_exists():
    assert TRAIN_DIR.exists()
    assert TEST_DIR.exists()
    for emotion in [label.lower() for label in EMOTION_LABELS]:
        assert (TRAIN_DIR / emotion).exists()
        assert (TEST_DIR / emotion).exists()


def test_model_loads_when_present():
    if not Path(MODEL_PATH).exists():
        return

    model = load_emotion_model(MODEL_PATH)
    labels = load_labels()
    assert model.output_shape[-1] == len(labels)
