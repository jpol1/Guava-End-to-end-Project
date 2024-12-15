import ast
import os

import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
INPUT_SHAPE_STR = os.getenv("INPUT_SHAPE")
INPUT_SHAPE = ast.literal_eval(INPUT_SHAPE_STR)


DATA_AUGMENTATION_1 = {
    "random_zoom_height": 0.1,
    "random_zoom_width": 0.1,
    "random_transaltion_height": 0.1,
    "random_translation_width": 0.1,
    "random_rotation": 0.1,
    "random_filp_mode": "horizontal",
}
DATA_AUGMENTATION_2 = {
    "random_zoom_height": 0.2,
    "random_zoom_width": 0.4,
    "random_transaltion_height": 0.3,
    "random_translation_width": 0.3,
    "random_rotation": 0.4,
    "random_filp_mode": "vertical",
}


MODEL_1 = [
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),  # Zastępuje Flatten
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
]

MODEL_2 = [
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),  # Zastępuje Flatten
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
]

MODEL_3 = [
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),  # Zastępuje Flatten
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
]

MODEL_4 = [
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),  # Zastępuje Flatten
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
]


DATA_AUGMENTATION_GRID = {
    "data_augmentation_1": DATA_AUGMENTATION_1,
    "data_augmentation_2": DATA_AUGMENTATION_2,
}

MODELS_GRID = {
    "model_conv8_16_dense32": MODEL_1,
    "model_conv8_16_32_dense64": MODEL_2,
    "model_conv8_dense16_32": MODEL_3,
    "model_conv16_dense32_64_64": MODEL_4,
}
