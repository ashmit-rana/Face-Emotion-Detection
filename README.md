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

- a FER2013-friendly CNN by default
- data augmentation
- clipped class weights for imbalanced emotions
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

## Why Happy Was Easier

The dataset has many more happy images than some other classes. For example, disgust has far fewer images, so the model gets less practice on that class. The updated training script adds class weights and augmentation to reduce that bias.

## GitHub Notes

The dataset is ignored by default because image datasets are usually large and may have licensing/privacy restrictions. Include dataset source instructions in this README before publishing.

If the model grows larger than GitHub's normal file limit, publish it through GitHub Releases or Git LFS.
