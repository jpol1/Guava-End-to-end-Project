import tensorflow as tf

from modules.data_preprocessing.load_images import load_image
from modules.data_preprocessing.preprocess_images import preprocess_image


def create_dataset(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda x, y: (preprocess_image(load_image(x)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def prefetch_dataset(dataset, batch_size=64):
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
