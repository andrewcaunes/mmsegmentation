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
from classes_dicts import color_palette

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
    # 4: vegetation
    # 5: terrain
    # 6: sky
    # 7: road marking
    # 8: traffic sign
    # 9: traffic light
    # 10: pothole
    # 11: manhole
    # 12: street light
    # 13: pole
    # 14: vehicle
    # 15: wall (include barrier, fence)
    # static with background as all other classes

    # uda_classes_dict={
    # "background":0, # not reported
    # "car":1,
    # "bicycle":2,
    # "motorcycle":3,
    # "truck":4,
    # "other vehicle":5, # In T-UDA, but o-vehicle in LIDAR-UDA
    # "person":6,
    # "drivable":7,
    # "sidewalk":8,
    # "terrain":9,
    # "vegetation":10,
    # "manmade": 11, # in T-UDA only
    # # "fence"
    # }
    replace_classes_dict = {
        0: 0,  # 'Bird',
        1: 0,  # 'Ground Animal',
        2: 8,  # 'Curb',
        3: 11,  # 'Fence',
        4: 11,  # 'Guard Rail',
        5: 11,  # 'Barrier',
        6: 11,  # 'Wall',
        7: 7,  # 'Bike Lane',
        8: 7,  # 'Crosswalk - Plain',
        9: 7,  # 'Curb Cut',
        10: 7, # 'Parking',
        11: 7, # 'Pedestrian Area',
        12: 7, # 'Rail Track',
        13: 7, # 'Road',
        14: 7, # 'Service Lane',
        15: 8, # 'Sidewalk',
        16: 11, # 'Bridge',
        17: 11, # 'Building',
        18: 11, # 'Tunnel',
        19: 6, # 'Person',
        20: 2, # 'Bicyclist',
        21: 3, # 'Motorcyclist',
        22: 3, # 'Other Rider',
        23: 7, # 'Lane Marking - Crosswalk',
        24: 7, # 'Lane Marking - General',
        25: 9,# 'Mountain',
        26: 9,# 'Sand',
        27: 0,# 'Sky',
        28: 9,# 'Snow',
        29: 9,# 'Terrain',
        30: 10, # 'Vegetation',
        31: 0, # 'Water',
        32: 11, # 'Banner'
        33: 11, # 'Bench'
        34: 11, # 'Bike Rack'
        35: 11, # 'Billboard'
        36: 7, # 'Catch Basin'
        37: 11, # 'CCTV Camera'
        38: 11, # 'Fire Hydrant'
        39: 11, # 'Junction Box'
        40: 11, # 'Mailbox'
        41: 7, # 'Manhole'
        42: 11, # 'Phone Booth'
        43: 11, # 'Pothole'
        44: 11, # 'Street Light' -> 'pole'
        45: 11, # 'Pole'
        46: 11, # 'Traffic Sign Frame'
        47: 11, # 'Utility Pole'
        48: 11, # 'Traffic Light'
        49: 11, # 'Traffic Sign (Back)'
        50: 11, # 'Traffic Sign (Front)'
        51: 11, # 'Trash Can'
        52: 2, # 'Bicycle'
        53: 0, # 'Boat'
        54: 5, # 'Bus'
        55: 1, # 'Car'
        56: 1, # 'Caravan'
        57: 3, # 'Motorcycle'
        58: 5, # 'On Rails'
        59: 5, # 'Other Vehicle'
        60: 5, # 'Trailer'
        61: 4, # 'Truck'
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
    replace_classes(input_folder=args.input_folder, 
                    output_folder=args.output_folder, 
                    replace_classes_dict=replace_classes_dict, 
                    config_file=args.config_file,
                    input_suffix=args.input_suffix,
                    output_suffix=args.output_suffix)
    
def replace_classes(input_folder, output_folder, replace_classes_dict, config_file, input_suffix, output_suffix=""):
    """ Replace classes in png files in folder with classes in replace_classes_dict."""
    # with open(config_file) as f:
    #     classes_config = json.load(f)
    #     id_to_color = {}
    #     labels = classes_config["labels"]
    #     for i, label in enumerate(labels):
    #         id_to_color[i] = label["color"]
    ignore_suffixes = ["_static_bg_", "_static_bg_nbs"]

    for root, dirs, files in os.walk(input_folder):
        if len(files) == 0 or (output_folder in root and input_folder != output_folder):
            continue
        if "/labels" not in root:
            continue
        logging.info("Processing folder %s", root)
        already_processed = [file.replace(output_suffix, "") for file in files if (file.endswith(output_suffix + ".png"))]
        logging.info('len(already_processed)=\n%s',len(already_processed))
        files_to_process = []
        for file in files:
            if file not in already_processed:
                ignore = False
                for ignore_suffix in ignore_suffixes:
                    if file.endswith(ignore_suffix + ".png"):
                        ignore=True
                if not ignore:
                    files_to_process.append(file)
        # files_to_process = [file for file in files if (file not in already_processed and not file.endswith(ignore_suffix + ".png"))]
        logging.info('len(files_to_process)=\n%s',len(files_to_process))

        for file in tqdm(files_to_process, total=len(files_to_process)):
            
            if file.endswith(input_suffix) and not file.endswith(output_suffix + ".png"):
                file_path = os.path.join(root, file)
                out_file_path = file_path.replace(input_folder, output_folder)
                if output_suffix != "":
                    out_file_path = out_file_path.split(".png")[0] + output_suffix + ".png"
                if not os.path.exists(os.path.dirname(out_file_path)):
                    logging.info("Creating output folder %s", os.path.dirname(out_file_path))
                    os.makedirs(os.path.dirname(out_file_path))


                img_original = Image.open(file_path)
                # logging.info('np.array(img_original).shape=\n%s',np.array(img_original).shape)
                # original_palette = img_original.getpalette()
                pillow_palette = np.random.randint(0, 256, (256, 3)).astype(np.uint8)
                pillow_palette[:color_palette.shape[0]] = color_palette
                color_background = pillow_palette[0]
                pillow_palette[255] = color_background
                pillow_palette = pillow_palette.flatten()
                # convert to numpy array
                img_original = np.array(img_original, dtype=np.uint8)
                new_img = img_original.copy()
            
                # replace classes
                for key, value in replace_classes_dict.items():
                    new_img[img_original == key] = value

                # save image as greyscale png
                new_img = Image.fromarray(new_img)
                new_img.putpalette(pillow_palette)
                new_img.save(out_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_folder', help='/data/cityscapes', required=True)
    parser.add_argument('--output_folder', help='/data/cityscapes_new', required=True)
    parser.add_argument('--input_suffix', help='_labelTrainIds.png', default=".png")
    parser.add_argument('--config_file', help='/data/cityscapes_new', default="/data/cityscapes/config.json")
    parser.add_argument('--output_suffix', help='suffix to add between filename and extension like _new', default="")
    parser.add_argument('--ignore_suffix', help='suffix to ignore', default="")
    args = parser.parse_args()

    main(args)