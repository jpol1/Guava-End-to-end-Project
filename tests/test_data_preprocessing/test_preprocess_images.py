import os
import sys

import numpy as np
import pytest
import tensorflow as tf

from modules.data_preprocessing.preprocess_images import preprocess_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_preprocess_image_valid_input():
    random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    encoded_image = tf.io.encode_jpeg(random_image).numpy()

    processed_image = preprocess_image(encoded_image)

    assert processed_image.shape == (512, 512, 3)


def test_preprocess_image_invalid_input():
    invalid_input = b"not_an_image"
    with pytest.raises(tf.errors.InvalidArgumentError):
        preprocess_image(invalid_input)


def test_preprocess_image_grayscale_input():
    gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    encoded_image = tf.io.encode_jpeg(gray_image[:, :, np.newaxis]).numpy()

    processed_image = preprocess_image(encoded_image)

    assert processed_image.shape == (512, 512, 3)
