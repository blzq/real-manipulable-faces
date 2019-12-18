import os
import tensorflow as tf


def filenames_from_dataset_path(dataset_path):
    img_dirs = map(lambda d: os.path.join(dataset_path, d), os.listdir(dataset_path))

    img_files = []
    for img_dir in img_dirs:
        img_files_one_dir = list(map(lambda f: os.path.join(img_dir, f), os.listdir(img_dir)))
        img_files = img_files + img_files_one_dir

    return img_files


def load_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img
