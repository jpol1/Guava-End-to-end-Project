import os
import sys
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tracking_models.mlflow_track import (
    mlflow_start_experiment,
    mlflow_start_run,
)


@patch("modules.tracking_models.mlflow_track.mlflow.set_tracking_uri")
@patch("modules.tracking_models.mlflow_track.mlflow.set_experiment")
def test_mlflow_start_experiment(mock_set_experiment, mock_set_tracking_uri):
    title = "test_experiment"
    uri = "http://127.0.0.1:5000"

    mlflow_start_experiment(title, set_tracking_uri=uri)

    mock_set_tracking_uri.assert_called_once_with(uri)
    mock_set_experiment.assert_called_once_with(title)


@patch("modules.tracking_models.mlflow_track.mlflow.start_run")
@patch("modules.tracking_models.mlflow_track.mlflow.log_params")
@patch("modules.tracking_models.mlflow_track.mlflow.log_metrics")
@patch("modules.tracking_models.mlflow_track.mlflow.set_tag")
@patch("modules.tracking_models.mlflow_track.mlflow.log_artifact")
def test_mlflow_start_run(
    mock_log_artifact, mock_set_tag, mock_log_metrics, mock_log_params, mock_start_run
):
    params = {"learning_rate": 0.001, "batch_size": 32}
    metrics = {"accuracy": 0.95, "loss": 0.1}
    model_name = "test_model"
    artifacts = ["artifact1.txt", "artifact2.txt"]

    mlflow_start_run(params, metrics, model_name, artifacts)

    mock_start_run.assert_called_once()

    mock_log_params.assert_called_once_with(params)

    mock_log_metrics.assert_called_once_with(metrics)

    mock_set_tag.assert_called_once_with("model_name", model_name)

    assert mock_log_artifact.call_count == len(artifacts)
    for artifact in artifacts:
        mock_log_artifact.assert_any_call(artifact)
