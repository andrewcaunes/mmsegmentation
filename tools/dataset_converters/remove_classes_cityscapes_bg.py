"""
Script made by Andrew Caunes.
Remove classes from ground truth labels in png format.
Example use:
Specify the classes to replace in classes_to_replace dict then run:
python remove_classes.py --folder_path /path/to/folder
"""
import os
import shutil
import argparse
import logging

from tqdm import tqdm
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from PIL import Image

def main(args):
    logging.info("args = %s", args)

    # classes_to_replace = [11, 12, 13, 14, 15, 16, 17, 18] (see mmseg/datasets/cityscapes_static.py)
    # static with background as all other classes
    replace_classes_dict = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10:11,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 0
    }
    
    # static
    # replace_classes_dict = {
    #     11: 255,
    #     12: 255,
    #     13: 255,
    #     14: 255,
    #     15: 255,
    #     16: 255,
    #     17: 255,
    #     18: 255
    # }
    if not os.path.exists(args.output_folder):
        logging.info("Creating output folder %s", args.output_folder)
        os.makedirs(args.output_folder)
    replace_classes(input_folder=args.input_folder, output_folder=args.output_folder, replace_classes_dict=replace_classes_dict)

def replace_classes(input_folder, output_folder, replace_classes_dict):
    """ Replace classes in png files in folder with classes in replace_classes_dict."""
    for root, dirs, files in os.walk(input_folder):
        if len(files) == 0 or (output_folder in root and input_folder != output_folder):
            continue
        logging.info("Processing folder %s", root)
        for file in tqdm(files, total=len(files)):
            if file.endswith(args.suffix):
                file_path = os.path.join(root, file)
                out_file_path = file_path.replace(input_folder, output_folder)
                if not os.path.exists(os.path.dirname(out_file_path)):
                    logging.info("Creating output folder %s", os.path.dirname(out_file_path))
                    os.makedirs(os.path.dirname(out_file_path))
                # logging.info("file_path = %s", file_path)
                # load greyscale image
                img_original = Image.open(file_path)
                # convert to numpy array
                img_original = np.array(img_original)
                new_img = img_original.copy()
            
                # replace classes
                for key, value in replace_classes_dict.items():
                    new_img[img_original == key] = value

                # save image as greyscale png
                new_img = Image.fromarray(new_img)
                new_img.save(out_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_folder', help='/data/cityscapes', required=True)
    parser.add_argument('--output_folder', help='/data/cityscapes_new', required=True)
    parser.add_argument('--suffix', help='_labelTrainIds.png', default=".png")
    # parser.add_argument('--classes_to_remove', help='', default=None)
    args = parser.parse_args()

    main(args)