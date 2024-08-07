import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--images_folder', help='Input file', required=True)
parser.add_argument('--labels_folder', help='Input file', required=True)

args = parser.parse_args()

import os
import shutil
import numpy as np

train_val_ratio = 0.95

images_path = args.images_folder
labels_path = args.labels_folder

for im_name in os.listdir(os.path.join(images_path, "train")):
    if np.random.rand() > train_val_ratio:
        shutil.move(os.path.join(images_path, "train", im_name), os.path.join(images_path, "val", im_name))
        label_name = im_name[:-4] + "_bin.png"
        shutil.move(os.path.join(labels_path, "train", label_name), os.path.join(labels_path, "val", label_name))