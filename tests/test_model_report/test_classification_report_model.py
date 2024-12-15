from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf

from modules.model_report.classification_report_model import get_classification_report


@patch("modules.model_report.classification_report_model.load_image")
@patch("modules.model_report.classification_report_model.preprocess_image")
def test_get_classification_report(mock_load_image, mock_preprocess_image):
    mock_load_image.return_value = tf.io.encode_jpeg(
        tf.zeros((224, 224, 3), dtype=tf.uint8)
    )  # Przykładowy obraz JPEG
    mock_preprocess_image.return_value = np.zeros(
        (512, 512, 3)
    )  # Obraz po preprocessing'u

    test_filenames = ["path/to/image1.jpg", "path/to/image2.jpg"]
    y_true = np.array([[1, 0], [0, 1]])

    mock_model = Mock()
    mock_model.predict.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

    result = get_classification_report(mock_model, test_filenames, y_true)

    assert "Class 0" in result
    assert "Class 1" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1-score" in result

    assert mock_model.predict.called
    assert mock_load_image.call_count == len(test_filenames)
    assert mock_preprocess_image.call_count == len(test_filenames)
    mock_model.predict.assert_called_once()
