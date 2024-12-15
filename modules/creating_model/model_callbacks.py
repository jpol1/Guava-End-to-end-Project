import tensorflow as tf


def early_stopping_func(patience=5, restore_best_weights=True):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=patience, restore_best_weights=restore_best_weights
    )
    return early_stopping


def model_checkpoint(save_location, save_best_only=True):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_location,
        save_best_only=save_best_only,
    )
    return model_checkpoint
