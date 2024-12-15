import os
import sys

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.creating_model.model_skeleton import (
    create_data_augmentation,
    create_model,
    set_up_model,
    start_train,
)

NUM_CLASSES = 5
INPUT_SHAPE = (256, 256, 3)


@pytest.fixture
def sample_data_augmentation():
    return create_data_augmentation(
        random_zoom_height=0.1,
        random_zoom_width=0.1,
        random_transaltion_height=0.2,
        random_translation_width=0.2,
        random_rotation=0.3,
        random_filp_mode="horizontal",
    )


@pytest.fixture
def sample_layers():
    return [
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]


@pytest.fixture
def sample_model(sample_data_augmentation, sample_layers):
    model_data = create_model(
        input_shape=INPUT_SHAPE,
        data_augmentation=sample_data_augmentation,
        layers=sample_layers,
    )
    model = model_data["model"]
    set_up_model(
        model,
        optimizer="adam",
        metrics=["accuracy"],
        loss="sparse_categorical_crossentropy",
    )
    return model


@pytest.fixture
def sample_dataset():
    random_images = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
    random_labels = np.random.randint(0, NUM_CLASSES, 10)
    dataset = (
        tf.data.Dataset.from_tensor_slices((random_images, random_labels))
        .batch(2)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


# Testy


def test_create_data_augmentation(sample_data_augmentation):
    augmentation = sample_data_augmentation

    assert isinstance(augmentation, tf.keras.Sequential)
    assert len(augmentation.layers) == 4
    assert isinstance(augmentation.layers[0], layers.RandomZoom)
    assert isinstance(augmentation.layers[1], layers.RandomTranslation)
    assert isinstance(augmentation.layers[2], layers.RandomRotation)
    assert isinstance(augmentation.layers[3], layers.RandomFlip)


def test_create_model(sample_model):
    model = sample_model

    data_augmentation = model.layers[0]

    aug_layers = data_augmentation.layers

    assert model.input_shape == (None, *INPUT_SHAPE)
    assert isinstance(model, tf.keras.Sequential)
    assert isinstance(data_augmentation, tf.keras.Sequential)  # Data Augmentation layer
    assert isinstance(model.layers[-1], layers.Dense)
    assert model.layers[-1].units == NUM_CLASSES
    assert len(model.layers) + len(aug_layers) == 10


def test_set_up_model(sample_model):
    model = sample_model

    assert model.optimizer.__class__.__name__ == "Adam"
    assert model.loss == "sparse_categorical_crossentropy"


def test_start_train(sample_model, sample_dataset):
    model = sample_model
    dataset = sample_dataset

    history = start_train(
        model, dataset=dataset, epochs=1, callbacks=[], validation_data=None
    )

    assert "history" in history
    assert "epochs" in history
    assert history["epochs"] == 1
    assert "accuracy" in history["history"].history
    assert len(history["history"].history["accuracy"]) == 1


def test_set_up_model_metrics(sample_model, sample_dataset):
    model = sample_model
    dataset = sample_dataset

    history = model.fit(dataset, epochs=1, verbose=1)

    assert "accuracy" in history.history
