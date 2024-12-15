import os
import sys
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_preprocessing.data_visualization import (
    generate_training_plot,
    show_labels_chart,
    show_labels_count,
)


def test_show_labels_count():
    labels_to_count = [0, 1, 1, 1, 2, 0, 3, 0, 1, 2, 2, 0]
    expected_result = {0: 4, 1: 4, 2: 3, 3: 1}

    result = show_labels_count(labels_to_count)

    assert result == expected_result


@patch("matplotlib.pyplot.show")  # Mockowanie plt.show(), aby nie wyświetlać wykresu
def test_show_labels_chart(mock_show):
    class_counts = {"Class A": 10, "Class B": 20, "Class C": 15}

    with patch("matplotlib.pyplot.bar") as mock_bar, patch(
        "matplotlib.pyplot.xticks"
    ) as mock_xticks, patch("matplotlib.pyplot.title") as mock_title:

        show_labels_chart(class_counts)

        mock_bar.assert_called_once_with(
            list(class_counts.keys()),
            list(class_counts.values()),
            color="skyblue",
            edgecolor="black",
        )
        mock_xticks.assert_called_once()
        mock_title.assert_called_once_with("Size of classes", fontsize=14)

        mock_show.assert_called_once()


def test_generate_training_plot(tmp_path):
    history = {
        "accuracy": [0.8, 0.85, 0.9],
        "val_accuracy": [0.75, 0.8, 0.85],
        "loss": [0.4, 0.3, 0.2],
        "val_loss": [0.5, 0.4, 0.35],
    }
    model_name = "test_model"
    save_location = tmp_path

    generate_training_plot(history, model_name, save_location)

    output_file = tmp_path / f"{model_name}_training_chart.png"
    assert output_file.exists()
