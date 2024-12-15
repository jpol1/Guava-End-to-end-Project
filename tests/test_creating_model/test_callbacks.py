import os
import sys

import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.creating_model.model_callbacks import early_stopping_func, model_checkpoint


def test_early_stopping_func():
    patience = 10
    restore_best_weights = True

    callback = early_stopping_func(
        patience=patience, restore_best_weights=restore_best_weights
    )

    assert isinstance(callback, tf.keras.callbacks.EarlyStopping)
    assert callback.patience == patience
    assert callback.restore_best_weights == restore_best_weights


def test_model_checkpoint(tmp_path):
    save_location = os.path.join(tmp_path, "model_checkpoint.keras")
    save_best_only = True

    callback = model_checkpoint(save_location, save_best_only=save_best_only)

    assert isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
    assert callback.filepath == save_location
    assert callback.save_best_only == save_best_only

    assert not os.path.exists(save_location)
