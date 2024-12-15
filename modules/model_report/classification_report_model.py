import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_preprocessing.load_images import load_image
from data_preprocessing.preprocess_images import preprocess_image
from sklearn.metrics import classification_report


def get_classification_report(model, test_filenames, y_true):
    """
    Generates a classification report for the model based on test image file paths.

    Parameters:
        model: The loaded Keras model.
        test_filenames: List of full file paths to the test images.
        y_true: List of true class labels (class indices).

    Returns:
        A classification report as a string.
    """

    images = []

    for file_path in test_filenames:
        img = load_image(file_path)
        img = preprocess_image(img)
        images.append(img)

    X_test = np.array(images)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = np.argmax(y_true, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=[f"Class {i}" for i in range(len(set(y_true)))]
    )

    return report
