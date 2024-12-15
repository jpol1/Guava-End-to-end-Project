import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

SET_TRACKING_URI = os.getenv("SET_TRACKING_URI")


def mlflow_start_experiment(title, set_tracking_uri=SET_TRACKING_URI):
    mlflow.set_tracking_uri(set_tracking_uri)
    mlflow.set_experiment(title)


def mlflow_start_run(params: dict, metrics: dict, model_name, artifacts=[]):
    """
    :param params: parameters used to train model like learning_rate, batch_size, epochs_number
    :param metrics: results which model returned like accuracy, precision, recall
    :param artifacts: A list of file paths to artifacts (e.g., model files, training logs, plots, or configuration files) to be logged to the MLflow tracking server.
     Each artifact is stored alongside the run's metadata for future reference.
    :return: MLFlow ready to read results
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", model_name)
        if artifacts:
            for arti in artifacts:
                mlflow.log_artifact(arti)
