import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_preprocessing.split_data import split_data


@pytest.fixture
def filenames():
    filenames = [f"{_}" for _ in range(0, 100)]
    return filenames


@pytest.fixture
def labels():
    labels = np.random.randint(0, 2, 100).tolist()
    return labels


@pytest.fixture
def wrong_filenames():
    filenames = [f"{_}" for _ in range(0, 105)]
    return filenames


@pytest.fixture
def wrong_labels():
    labels = np.random.randint(0, 2, 100).tolist()
    return labels


def test_split_data_valid_input(filenames, labels):

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(filenames, labels)

    X_train_size = 60
    X_val_size = 20
    X_test_size = 20
    y_train_size = 60
    y_val_size = 20
    y_test_size = 20

    assert len(filenames) == len(labels)
    assert len(X_train) == X_train_size
    assert len(X_val) == X_val_size
    assert len(X_test) == X_test_size
    assert len(y_train) == y_train_size
    assert len(y_val) == y_val_size
    assert len(y_test) == y_test_size


def test_split_data_invalid_input(wrong_filenames, wrong_labels):
    filenames = wrong_filenames
    labels = wrong_labels

    with pytest.raises(ValueError):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(filenames, labels)
