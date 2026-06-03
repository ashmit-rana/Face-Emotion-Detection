# Face Emotion Detection

Real-time facial emotion detection using TensorFlow and OpenCV. The project trains a seven-class emotion classifier and runs webcam inference with face cropping, preprocessing, confidence display, and frame-level prediction smoothing.

## Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Project Structure

```text
face_emotion_detection/
  dataset/
    train/<emotion>/
    test/<emotion>/
  model/
    emotion_model.h5
    emotion_labels.json
  train_model.py
  evaluate_model.py
  predict_image.py
  realtime_detection.py
  emotion_utils.py
  config.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

The default model is a grayscale CNN tuned for FER2013-style 48x48 emotion images. If you want to try transfer learning instead, use:

```bash
python train_model.py --architecture mobilenet --image-size 96
```

The training script uses:

- a stronger FER2013-friendly CNN by default
- data augmentation
- square-root class weights for imbalanced emotions
- early stopping
- learning-rate reduction
- best-model checkpointing

## Evaluate

```bash
python evaluate_model.py
```

This writes:

- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

Use the confusion matrix to see which emotions are being mixed up. This is especially useful for visually similar classes such as fear, sad, neutral, and angry.

## Predict One Image

```bash
python predict_image.py path/to/image.jpg --save-annotated output.jpg
```

## Real-Time Webcam Detection

```bash
python realtime_detection.py
```

Press `q` to quit.

Useful options:

```bash
python realtime_detection.py --smooth-frames 15 --confidence-threshold 0.40
```

## Dataset

This project uses a FER2013-style dataset with seven emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

The dataset is not included in this repository because image datasets can be large and may have licensing or privacy restrictions. To train the model, place the data in this structure:

```text
dataset/
  train/angry/
  train/disgust/
  train/fear/
  train/happy/
  train/neutral/
  train/sad/
  train/surprise/
  test/angry/
  test/disgust/
  test/fear/
  test/happy/
  test/neutral/
  test/sad/
  test/surprise/
```

## Results and Limitations

The model performs strongest on clearer expressions such as Happy, Surprise, and Neutral. Some classes, especially Fear and Sad, are harder to separate because facial expressions can be subtle and visually similar in low-resolution FER2013-style images.

Class imbalance also affects performance. For example, Happy has many more training examples than Disgust, so the training pipeline uses augmentation and clipped class weights to reduce bias toward majority classes.

## Model File

The trained model is included for demo use. If future model files become larger than GitHub's standard file size limit, they should be shared through GitHub Releases or Git LFS.
