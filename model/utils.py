import os


def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)