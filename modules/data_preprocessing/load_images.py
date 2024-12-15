import os

import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

FOLDER_BASE = os.getenv("FOLDER_BASE")
LABELS_DICT = {"Anthracnose": 0, "healthy_guava": 1, "fruit_fly": 2}


def get_filenames_and_labels(base_folder=FOLDER_BASE, labels_instruction=LABELS_DICT):
    filenames = []
    labels = []
    for subfolder, label in labels_instruction.items():
        full_folder_path = os.path.join(base_folder, subfolder)
        files_list = [
            os.path.join(full_folder_path, f)
            for f in os.listdir(full_folder_path)
            if os.path.isfile(os.path.join(full_folder_path, f))
        ]
        labels_list = len(files_list) * [label]
        filenames.extend(files_list)
        labels.extend(labels_list)
    return filenames, labels


def load_image(image_path):
    image = tf.io.read_file(image_path)
    return image


def load_image_from_stream(uploaded_file):
    image_content = uploaded_file.read()
    image = tf.image.decode_image(image_content, channels=3)
    image = tf.image.resize(image, [512, 512])
    return image
