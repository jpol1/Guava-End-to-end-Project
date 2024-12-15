import os
import sys
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from modules.data_preprocessing.load_images import (
    get_filenames_and_labels,
    load_image,
    load_image_from_stream,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TEST_FOLDER_BASE = r"D:\TestDataset"
TEST_LABELS_DICT = {"Anthracnose": 0, "healthy_guava": 1, "fruit_fly": 2}


@pytest.fixture
def mock_os_listdir():
    with patch("os.listdir") as mock_listdir:
        mock_listdir.side_effect = lambda path: {
            os.path.join(TEST_FOLDER_BASE, "Anthracnose"): ["image1.jpg", "image2.jpg"],
            os.path.join(TEST_FOLDER_BASE, "healthy_guava"): [
                "image3.jpg",
                "image4.jpg",
            ],
            os.path.join(TEST_FOLDER_BASE, "fruit_fly"): ["image5.jpg", "image6.jpg"],
        }.get(path, [])
        yield mock_listdir


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        yield mock_isfile


@pytest.fixture
def fake_image_file():
    image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    image_tensor = tf.image.encode_png(image_array)
    return BytesIO(image_tensor.numpy())


def test_get_filenames_and_labels(mock_os_listdir, mock_os_path_isfile):
    filenames, labels = get_filenames_and_labels(TEST_FOLDER_BASE, TEST_LABELS_DICT)

    expected_filenames = [
        os.path.join(TEST_FOLDER_BASE, "Anthracnose", "image1.jpg"),
        os.path.join(TEST_FOLDER_BASE, "Anthracnose", "image2.jpg"),
        os.path.join(TEST_FOLDER_BASE, "healthy_guava", "image3.jpg"),
        os.path.join(TEST_FOLDER_BASE, "healthy_guava", "image4.jpg"),
        os.path.join(TEST_FOLDER_BASE, "fruit_fly", "image5.jpg"),
        os.path.join(TEST_FOLDER_BASE, "fruit_fly", "image6.jpg"),
    ]
    expected_labels = [0, 0, 1, 1, 2, 2]

    assert filenames == expected_filenames
    assert labels == expected_labels


def test_load_image():
    image_path = "dummy_path/image.jpg"
    with patch("tensorflow.io.read_file") as mock_read_file:
        mock_read_file.return_value = b"image_data"
        result = load_image(image_path)

        assert result == b"image_data"


def test_get_filenames_and_labels_invalid_folder():
    invalid_folder = r"D:\InvalidFolder"

    with patch("os.listdir") as mock_listdir:
        mock_listdir.side_effect = FileNotFoundError

        with pytest.raises(FileNotFoundError):
            get_filenames_and_labels(invalid_folder, TEST_LABELS_DICT)


def test_load_image_from_stream(fake_image_file):
    image = load_image_from_stream(fake_image_file)

    assert isinstance(image, tf.Tensor)
    assert image.shape == (512, 512, 3)
    assert image.dtype == tf.float32
