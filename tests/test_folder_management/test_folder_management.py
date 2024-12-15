import os
import shutil
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.folder_management.folder_management import create_folder, delete_folder


@pytest.fixture
def test_folder():
    folder_name = "test_folder"
    yield folder_name
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


def test_create_folder(test_folder):
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    create_folder(test_folder)
    assert os.path.exists(test_folder) == True


def test_delete_folder(test_folder):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    delete_folder(test_folder)
    assert os.path.exists(test_folder) == False


def test_create_folder_already_exists(test_folder):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    create_folder(test_folder)
    assert os.path.exists(test_folder) == True


def test_delete_folder_not_exists(test_folder):
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    delete_folder(test_folder)
    assert os.path.exists(test_folder) == False
