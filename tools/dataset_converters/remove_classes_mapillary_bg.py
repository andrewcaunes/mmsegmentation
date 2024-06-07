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
import json
import cv2

from tqdm import tqdm
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from PIL import Image

def main(args):
    logging.info("args = %s", args)

#  'Bird',
#  'Ground Animal',
#  'Curb',
#  'Fence',
#  'Guard Rail',
#  'Barrier',
#  'Wall',
#  'Bike Lane',
#  'Crosswalk - Plain',
#  'Curb Cut',
#  'Parking',
#  'Pedestrian Area',
#  'Rail Track',
#  'Road',
#  'Service Lane',
#  'Sidewalk',
#  'Bridge',
#  'Building',
#  'Tunnel',
#  'Person',
#  'Bicyclist',
#  'Motorcyclist',
#  'Other Rider',
#  'Lane Marking - Crosswalk',
#  'Lane Marking - General',
#  'Mountain',
#  'Sand',
#  'Sky',
#  'Snow',
#  'Terrain',
#  'Vegetation',
#  'Water',
#  'Banner',
#  'Bench',
#  'Bike Rack',
#  'Billboard',
#  'Catch Basin',
#  'CCTV Camera',
#  'Fire Hydrant',
#  'Junction Box',
#  'Mailbox',
#  'Manhole',
#  'Phone Booth',
#  'Pothole',
#  'Street Light',
#  'Pole',
#  'Traffic Sign Frame',
#  'Utility Pole',
#  'Traffic Light',
#  'Traffic Sign (Back)',
#  'Traffic Sign (Front)',
#  'Trash Can',
#  'Bicycle',
#  'Boat',
#  'Bus',
#  'Car',
#  'Caravan',
#  'Motorcycle',
#  'On Rails',
#  'Other Vehicle',
#  'Trailer',
#  'Truck',
#  'Wheeled Slow',
#  'Car Mount',
#  'Ego Vehicle',
#  'Unlabeled'    
    
    # Keep :
    # 0: background
    # 1: road
    # 2: sidewalk
    # 3: building
    # 4: wall
    # 5: fence
    # 6: pole
    # 7: traffic light
    # 8: traffic sign
    # 9: vegetation
    # 10: terrain
    # 11: sky
    # static with background as all other classes
    replace_classes_dict = {
        0: 0,  # 'Bird',
        1: 0,  # 'Ground Animal',
        2: 2,  # 'Curb',
        3: 5,  # 'Fence',
        4: 5,  # 'Guard Rail',
        5: 0,  # 'Barrier',
        6: 4,  # 'Wall',
        7: 1,  # 'Bike Lane',
        8: 1,  # 'Crosswalk - Plain',
        9: 1,  # 'Curb Cut',
        10: 1, # 'Parking',
        11: 2, # 'Pedestrian Area',
        12: 1, # 'Rail Track',
        13: 1, # 'Road',
        14: 1, # 'Service Lane',
        15: 2, # 'Sidewalk',
        16: 3, # 'Bridge',
        17: 3, # 'Building',
        18: 3, # 'Tunnel',
        19: 0, # 'Person',
        20: 0, # 'Bicyclist',
        21: 0, # 'Motorcyclist',
        22: 0, # 'Other Rider',
        23: 1, # 'Lane Marking - Crosswalk',
        24: 1, # 'Lane Marking - General',
        25: 10,# 'Mountain',
        26: 10,# 'Sand',
        27: 11,# 'Sky',
        28: 10,# 'Snow',
        29: 10,# 'Terrain',
        30: 9, # 'Vegetation',
        31: 0, # 'Water',
        32: 0, # 'Banner'
        33: 0, # 'Bench'
        34: 0, # 'Bike Rack'
        35: 0, # 'Billboard'
        36: 0, # 'Catch Basin'
        37: 0, # 'CCTV Camera'
        38: 0, # 'Fire Hydrant'
        39: 0, # 'Junction Box'
        40: 0, # 'Mailbox'
        41: 1, # 'Manhole'
        42: 0, # 'Phone Booth'
        43: 1, # 'Pothole'
        44: 6, # 'Street Light' -> 'pole'
        45: 6, # 'Pole'
        46: 0, # 'Traffic Sign Frame'
        47: 6, # 'Utility Pole'
        48: 7, # 'Traffic Light'
        49: 8, # 'Traffic Sign (Back)'
        50: 8, # 'Traffic Sign (Front)'
        51: 0, # 'Trash Can'
        52: 0, # 'Bicycle'
        53: 0, # 'Boat'
        54: 0, # 'Bus'
        55: 0, # 'Car'
        56: 0, # 'Caravan'
        57: 0, # 'Motorcycle'
        58: 0, # 'On Rails'
        59: 0, # 'Other Vehicle'
        60: 0, # 'Trailer'
        61: 0, # 'Truck'
        62: 0, # 'Wheeled Slow'
        63: 0, # 'Car Mount'
        64: 0, # 'Ego Vehicle'
        65: 0, # 'Unlabeled'
        
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
    replace_classes(input_folder=args.input_folder, output_folder=args.output_folder, replace_classes_dict=replace_classes_dict, config_file=args.config_file)

def replace_classes(input_folder, output_folder, replace_classes_dict, config_file):
    """ Replace classes in png files in folder with classes in replace_classes_dict."""
    # with open(config_file) as f:
    #     classes_config = json.load(f)
    #     id_to_color = {}
    #     labels = classes_config["labels"]
    #     for i, label in enumerate(labels):
    #         id_to_color[i] = label["color"]


    for root, dirs, files in os.walk(input_folder):
        if len(files) == 0 or (output_folder in root and input_folder != output_folder):
            continue
        if "/labels" not in root:
            continue
        logging.info("Processing folder %s", root)
        already_processed = [file.replace(args.output_suffix, "") for file in files if file.endswith(args.output_suffix + ".png")]
        files_to_process = [file for file in files if file not in already_processed]

        for file in tqdm(files_to_process, total=len(files_to_process)):
            
            if file.endswith(args.suffix) and not file.endswith(args.output_suffix + ".png"):
                file_path = os.path.join(root, file)
                out_file_path = file_path.replace(input_folder, output_folder)
                if args.output_suffix != "":
                    out_file_path = out_file_path.split(".png")[0] + args.output_suffix + ".png"
                if not os.path.exists(os.path.dirname(out_file_path)):
                    logging.info("Creating output folder %s", os.path.dirname(out_file_path))
                    os.makedirs(os.path.dirname(out_file_path))


                img_original = Image.open(file_path)
                original_palette = img_original.getpalette()
                # convert to numpy array
                img_original = np.array(img_original, dtype=np.uint8)
                new_img = img_original.copy()
            
                # replace classes
                for key, value in replace_classes_dict.items():
                    new_img[img_original == key] = value

                # save image as greyscale png
                new_img = Image.fromarray(new_img)
                new_img.putpalette(original_palette)
                new_img.save(out_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_folder', help='/data/cityscapes', required=True)
    parser.add_argument('--output_folder', help='/data/cityscapes_new', required=True)
    parser.add_argument('--suffix', help='_labelTrainIds.png', default=".png")
    parser.add_argument('--config_file', help='/data/cityscapes_new', default="/data/cityscapes/config.json")
    parser.add_argument('--output_suffix', help='suffix to add between filename and extension like _new', default="")
    args = parser.parse_args()

    main(args)