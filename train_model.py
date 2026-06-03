import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import BATCH_SIZE, CLASS_INDICES_PATH, EPOCHS, IMAGE_SIZE, LABELS_PATH, MODEL_DIR, MODEL_PATH, TEST_DIR, TRAIN_DIR


def build_fer_cnn_v1(num_classes, image_size):
    weight_decay = 1e-4
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(image_size, image_size, 1)),
            Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.30),

            Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.35),

            Flatten(),
            Dense(256, activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Dropout(0.50),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def build_fer_cnn(num_classes, image_size):
    weight_decay = 7e-5
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(image_size, image_size, 1)),

            Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.20),

            Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.30),

            Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.35),

            Flatten(),
            Dense(512, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Dropout(0.45),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=["accuracy"],
    )
    return model


def build_mobilenet(num_classes, image_size, weights):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def balanced_class_weights(classes, max_weight, mode):
    if mode == "none":
        return None

    counts = np.bincount(classes)
    total = np.sum(counts)
    weights = total / (len(counts) * np.maximum(counts, 1))

    if mode == "sqrt":
        weights = np.sqrt(weights)

    weights = np.minimum(weights, max_weight)
    return dict(enumerate(weights.astype("float32")))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the face emotion classifier.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--architecture", choices=["fer_cnn", "fer_cnn_v1", "mobilenet"], default="fer_cnn")
    parser.add_argument("--class-weight-mode", choices=["balanced", "sqrt", "none"], default="sqrt")
    parser.add_argument("--max-class-weight", type=float, default=2.5)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.architecture == "mobilenet":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        color_mode = "rgb"
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=12,
            width_shift_range=0.12,
            height_shift_range=0.12,
            zoom_range=0.15,
            shear_range=0.08,
            brightness_range=(0.75, 1.25),
            horizontal_flip=True,
            fill_mode="nearest",
        )
        test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        color_mode = "grayscale"
        train_gen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=10,
            width_shift_range=0.10,
            height_shift_range=0.10,
            zoom_range=0.10,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(args.image_size, args.image_size),
        color_mode=color_mode,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(args.image_size, args.image_size),
        color_mode=color_mode,
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    labels = [label.title() for label, _ in sorted(train_data.class_indices.items(), key=lambda item: item[1])]
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data.class_indices, f, indent=2)

    class_weight_map = balanced_class_weights(train_data.classes, args.max_class_weight, args.class_weight_mode)
    print(f"Using class weights: {class_weight_map}")

    if args.architecture == "mobilenet":
        weights = None if args.weights == "none" else args.weights
        model = build_mobilenet(num_classes=train_data.num_classes, image_size=args.image_size, weights=weights)
    elif args.architecture == "fer_cnn_v1":
        model = build_fer_cnn_v1(num_classes=train_data.num_classes, image_size=args.image_size)
    else:
        model = build_fer_cnn(num_classes=train_data.num_classes, image_size=args.image_size)
    model.summary()

    callbacks = [
        ModelCheckpoint(args.model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        CSVLogger(str(MODEL_DIR / "training_log.csv")),
    ]

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=args.epochs,
        class_weight=class_weight_map,
        callbacks=callbacks,
    )

    model.save(args.model_path)
    print(f"Model trained and saved to {args.model_path}")


if __name__ == "__main__":
    main()
