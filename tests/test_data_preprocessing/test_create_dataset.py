import os
import sys
from unittest.mock import patch

import pytest
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_preprocessing.create_dataset import create_dataset, prefetch_dataset


@pytest.fixture
def sample_dataset():
    dummy_data = [tf.constant([[1.0]]), tf.constant([[2.0]])]
    dummy_labels = [0, 1]
    dataset = tf.data.Dataset.from_tensor_slices((dummy_data, dummy_labels))

    return dataset


@patch("modules.data_preprocessing.create_dataset.load_image")
@patch("modules.data_preprocessing.create_dataset.preprocess_image")
def test_create_dataset(mock_preprocess_image, mock_load_image):
    filenames = ["image1.jpg", "image2.jpg", "image3.jpg"]
    labels = [0, 1, 0]

    mock_load_image.side_effect = lambda x: tf.constant(
        "mocked_image_data", dtype=tf.string
    )

    mock_preprocess_image.side_effect = lambda x: tf.ones(
        (512, 512, 3), dtype=tf.float32
    )

    dataset = create_dataset(filenames, labels)

    dataset_iterator = iter(dataset)
    for expected_label in labels:
        image, label = next(dataset_iterator)
        assert image.shape == (512, 512, 3)
        assert tf.reduce_all(image == 1.0)
        assert label.numpy() == expected_label


def test_prefetch_dataset(sample_dataset):
    dataset = sample_dataset

    batch_size = 2
    prefetched_dataset = prefetch_dataset(dataset, batch_size=batch_size)

    for batch in prefetched_dataset:
        images, labels = batch
        assert images.shape[0] <= batch_size
        assert len(labels) == images.shape[0]
