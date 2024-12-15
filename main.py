import os

from dotenv import load_dotenv
from tensorflow.keras.utils import to_categorical

from modules.creating_model.model_callbacks import early_stopping_func, model_checkpoint
from modules.creating_model.model_skeleton import (
    create_data_augmentation,
    create_model,
    set_up_model,
    start_train,
)
from modules.creating_model.neural_network_grid import (
    DATA_AUGMENTATION_GRID,
    INPUT_SHAPE,
    MODELS_GRID,
)
from modules.data_preprocessing.create_dataset import create_dataset, prefetch_dataset
from modules.data_preprocessing.data_visualization import generate_training_plot
from modules.data_preprocessing.load_images import get_filenames_and_labels
from modules.data_preprocessing.split_data import split_data
from modules.tracking_models.mlflow_track import (
    mlflow_start_experiment,
    mlflow_start_run,
)

if __name__ == "__main__":
    load_dotenv()

    TRAINING_CHARTS = os.getenv("TRAINING_CHARTS")
    FOLDER_BASE = os.getenv("FOLDER_BASE")
    NUM_CLASSES = int(os.getenv("NUM_CLASSES"))

    filenames, labels = get_filenames_and_labels(FOLDER_BASE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(filenames, labels)

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    dataset_train = prefetch_dataset(create_dataset(X_train, y_train))
    dataset_val = prefetch_dataset(create_dataset(X_val, y_val))
    dataset_test = prefetch_dataset(create_dataset(X_test, y_test))

    models = {
        f"{model_name}-{augmentation_number}": create_model(
            INPUT_SHAPE, create_data_augmentation(**data_augmentation), layers
        )
        for model_name, layers in MODELS_GRID.items()
        for augmentation_number, data_augmentation in DATA_AUGMENTATION_GRID.items()
    }

    batch_size = 0
    for batch_data, batch_labels in dataset_train.take(1):
        batch_size = batch_data.shape[0]

    mlflow_start_experiment("Trained Models")

    for model_name, model in models.items():
        actual_model = model["model"]
        set_up_info = set_up_model(actual_model, loss="categorical_crossentropy")
        early_stopping = early_stopping_func()
        checkpoint = model_checkpoint(f"trained_models/{model_name}.keras")
        history = start_train(
            actual_model, dataset_train, 50, [early_stopping, checkpoint], dataset_val
        )

        generate_training_plot(history["history"].history, model_name, TRAINING_CHARTS)

        evaluating = actual_model.evaluate(dataset_test)

        layers_num = model["layers_num"]

        optimizer = set_up_info["optimizer"]
        loss = set_up_info["loss"]

        max_epochs = history["epochs"]
        epochs = len(history["history"].history["loss"])

        params = {
            "layers_num": layers_num,
            "optimizer": optimizer,
            "loss": loss,
            "max_epochs": max_epochs,
            "real_epochs": epochs,
        }

        metrics = {
            "loss": evaluating[0],
            "accuracy": evaluating[1],
        }

        training_chart_save_path = os.path.join(
            TRAINING_CHARTS, f"{model_name}_training_chart.png"
        )

        mlflow_start_run(
            params=params,
            metrics=metrics,
            model_name=model_name,
            artifacts=[rf"trained_models/{model_name}.keras", training_chart_save_path],
        )
