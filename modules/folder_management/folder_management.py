import os
import shutil


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def delete_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
