import ast
import os

import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
INPUT_SHAPE_STR = os.getenv("INPUT_SHAPE")
INPUT_SHAPE = ast.literal_eval(INPUT_SHAPE_STR)


def create_data_augmentation(
    random_zoom_height: float = 0.0,
    random_zoom_width: float = 0.0,
    random_transaltion_height: float = 0.0,
    random_translation_width: float = 0.0,
    random_rotation: float = 0.0,
    random_filp_mode: str = "horizontal_and_vertical",
):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomZoom(
                height_factor=random_zoom_height, width_factor=random_zoom_width
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=random_transaltion_height,
                width_factor=random_translation_width,
            ),
            tf.keras.layers.RandomRotation(random_rotation),
            tf.keras.layers.RandomFlip(mode=random_filp_mode),
        ]
    )
    return data_augmentation


def create_model(input_shape: tuple, data_augmentation=None, layers=[]):
    """
    SIGNIFICANT - Input Layer, Each layer in data augmentation, Rescaling layer and the rest of the layers
    are given as a parameter to the function are included in final layers number.

    :param input_shape:
    :param data_augmentation:
    :param layers:
    :return:
    """
    if data_augmentation is None:
        data_augmentation = create_data_augmentation()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=input_shape),
            data_augmentation,
            tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1),
        ]
    )

    for layer in layers:
        model.add(layer)

    return {
        "model": model,
        "layers_num": len(layers) + len(data_augmentation.layers) + 2,
    }


def set_up_model(
    model,
    optimizer="adam",
    metrics=["accuracy"],
    loss="sparse_categorical_crossentropy",
):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return {"optimizer": optimizer, "loss": loss}


def start_train(model, dataset, epochs, callbacks, validation_data):
    history = model.fit(
        dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_data
    )

    return {"history": history, "epochs": epochs}
