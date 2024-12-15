import tensorflow as tf


def preprocess_image(image):
    try:
        image = tf.image.decode_jpeg(image, channels=3)  # Wymuś RGB
    except tf.errors.InvalidArgumentError:
        grayscale_image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.grayscale_to_rgb(grayscale_image)

    image = tf.image.resize(image, [512, 512])
    return image
