# Convert png images to 8-bit grayscale images with classes according to order
# in dataset-api:

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torchvision
from torchvision.io import ImageReadMode
from tqdm import tqdm
from multiprocessing import Pool
import torch.multiprocessing as mp
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

from multiprocessing import set_start_method
from PIL import Image

# backend = 'cv2' 
backend = 'pillow'

def save_image(args):
    gs_img, gs_img_path = args
    gs_img = gs_img.numpy().astype(np.uint8)
    # logging.info('gs_img=\n%s',gs_img)
    # logging.info('gs_img_path=\n%s',gs_img_path)
    if backend=='cv2':
        cv2.imwrite(gs_img_path, gs_img)
    elif backend=='pillow':
        # create 'L' image from grayscale image
        Image.fromarray(gs_img, mode="L").save(gs_img_path)
    # logging.info('gs_img_path=\n%s',gs_img_path)

if __name__ == '__main__':
    # set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dataset', help=' \
                        The dataset should be like : \
                        dataset/ \
                        ├── images \
                        │   ├── train \
                        │   │   ├── X.jpg \
                        │   |── val \
                        │   │   ├── X.jpg \
                        |── labels \
                        │   ├── train \
                        │   │   ├── X.png \
                        │   |── val \
                        │   │   ├── X.png     \
                        ', required=True)

    args = parser.parse_args()
    classes=('void', 's_w_d', 's_y_d', 'ds_w_dn', 'ds_y_dn', 'sb_w_do', 'sb_y_do', 'b_w_g', 'b_y_g', 'db_w_g', 
                'db_y_g', 'db_w_s', 's_w_s', 'ds_w_s', 's_w_c', 's_y_c', 's_w_p', 's_n_p', 'c_wy_z', 'a_w_u', 'a_w_t', 
                'a_w_tl', 'a_w_tr', 'a_w_tlr', 'a_w_l', 'a_w_r', 'a_w_lr', 'a_n_lu', 'a_w_tu', 'a_w_m', 'a_y_t', 'b_n_sr', 
                'd_wy_za', 'r_wy_np', 'vom_wy_n', 'om_n_n', 'noise', 'ignored'),
    palette=[[0,   0,   0], [ 70, 130, 180], [220,  20,  60], [128,   0, 128], [255, 0,   0], [  0,   0,  60],
                [  0,  60, 100], [  0,   0, 142], [119,  11,  32], [244,  35, 232], [  0,   0, 160], [153, 153, 153],
                [220, 220,   0], [250, 170,  30], [102, 102, 156], [128,   0,   0], [128,  64, 128], [238, 232, 170],
                [190, 153, 153], [  0,   0, 230], [128, 128,   0], [128,  78, 160], [150, 100, 100], [255, 165,   0],
                [180, 165, 180], [107, 142,  35], [201, 255, 229], [0,   191, 255], [ 51, 255,  51], [250, 128, 114],
                [127, 255,   0], [255, 128,   0], [  0, 255, 255], [178, 132, 190], [128, 128,  64], [102,   0, 204],
                [  0, 153, 153], [255, 255, 255]]
    dims = (2710, 3384)
    gpu = 2
    os.makedirs(os.path.join(args.dataset, "labels_gs"), exist_ok=True)
    if gpu==0 :
        for root, dirs, files in os.walk(os.path.join(args.dataset, "labels")):
            if len(files) > 0:
                for file in files:
                    if file.endswith(".png") and "/labels/" in root:
                        print(os.path.join(root, file))
                        # replace all pixels according to palette
                        img = cv2.imread(os.path.join(root, file))
                        gs_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        for i, color in enumerate(np.array(palette)):
                            gs_img[(img == np.array(color)).all(axis=2)] = i
                        # save image
                        gs_img_path = os.path.join(args.dataset, "labels_gs", root.split("/")[-1], file)
                        os.makedirs(os.path.dirname(gs_img_path), exist_ok=True)
                        cv2.imwrite(gs_img_path, gs_img)

    elif gpu==1:
        palette_tensor = torch.tensor(palette, dtype=torch.uint8).cuda()
        for root, dirs, files in os.walk(os.path.join(args.dataset, "labels")):
            if len(files) > 0:
                for file in tqdm(files):
                    if file.endswith(".png") and "/labels/" in root:
                        # print(os.path.join(root, file))
                        # replace all pixels according to palette
                        img = torchvision.io.read_image(os.path.join(root, file), ImageReadMode.RGB).cuda()
                        gs_img = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.uint8).cuda()
                        for i, color in enumerate(palette_tensor):
                            gs_img[(img == color).all(axis=2)] = i
                        # save image
                        gs_img_path = os.path.join(args.dataset, "labels_gs", root.split("/")[-1], file)
                        os.makedirs(os.path.dirname(gs_img_path), exist_ok=True)
                        cv2.imwrite(gs_img_path, gs_img.cpu().numpy())

    elif gpu==2:
        import time
        K = 64
        with torch.no_grad():
            # Start timing palette tensor creation
            start_time_palette = time.time()
            palette_tensor = torch.tensor(palette, dtype=torch.uint8).cuda()
            print(f"Time taken for palette tensor creation: {time.time() - start_time_palette} seconds")

            for root, dirs, files in os.walk(os.path.join(args.dataset, "labels")):
                if len(files) > 0:
                    for i in tqdm(range(0, len(files), K)):
                        files_to_process = []
                        for j in range(K):
                            if i+j < len(files) and files[i+j].endswith(".png") and "/labels/" in root:
                                files_to_process.append(files[i+j])
                                os.makedirs(os.path.join(args.dataset, "labels_gs", root.split("/")[-1]), exist_ok=True)
                        # files_to_process = files[i:i+K]
                        gs_img_paths = [os.path.join(args.dataset, "labels_gs", root.split("/")[-1], file) for file in files_to_process]
                        if files_to_process[0] in os.listdir(os.path.dirname(gs_img_paths[0])):
                            continue
                        # Start timing empty tensor creation
                        start_time_empty_tensor = time.time()
                        imgs = torch.empty((K*dims[0], dims[1], 3), dtype=torch.uint8).cuda()
                        print(f"Time taken for empty tensor creation: {time.time() - start_time_empty_tensor} seconds")

                        # Start timing image reading and processing
                        start_time_img_read = time.time()
                        for j, file in enumerate(files_to_process):
                            if file.endswith(".png") and "/labels/" in root and "gs" not in root:
                                try:
                                    img = torchvision.io.read_image(os.path.join(root, file), ImageReadMode.RGB).cuda()
                                except:
                                    print(os.path.join(root, file))
                                    raise
                                img.transpose_(2, 0)
                                img.transpose_(0, 1)
                                imgs[j*dims[0]:(j+1)*dims[0], :, :] = img
                        print(f"Time taken for image reading and processing: {time.time() - start_time_img_read} seconds")

                        # Start timing conversion to grayscale
                        start_time_gs_conversion = time.time()
                        gs_imgs = torch.zeros((K*dims[0], dims[1])).cuda()
                        for i, color in enumerate(palette_tensor):
                            gs_imgs[(imgs == color).all(axis=2)] = i                   
                        # reshape the images to give to multiprocessing
                        gs_img_list = torch.split(gs_imgs.cpu(), dims[0], dim=0)
                        # gs_imgs = gs_imgs.reshape((K, dims[0], dims[1]))
                        # gs_img_list = [gs_imgs[i*dims[0]:(i+1)*dims[0], :] for i in range(len(files_to_process))]
                        print(f"Time taken for grayscale conversion: {time.time() - start_time_gs_conversion} seconds")

                        # # Start timing image writing
                        start_time_img_moving = time.time()
                        # Reshape gs_imgs into a list of smaller tensors using torch.split
                        
                        # Prepare gs_img_paths
                        
                        
                        # with mp.Pool(16) as pool:
                        #     gs_img_list = pool.map(move_tensor_to_cpu, gs_img_list)
                        # print(f"Time taken for moving to cpu: {time.time() - start_time_img_moving} seconds")

                        # # Start timing image writing
                        start_time_img_write = time.time()
                        # Use multiprocessing to save images
                        with mp.Pool() as pool:
                            pool.map(save_image, zip(gs_img_list, gs_img_paths))

                        # with Pool() as pool:
                        #     pool.map(save_image, zip(gs_img_list, gs_img_paths))
                        # for i, file in enumerate(files_to_process):
                        #     gs_img_path = os.path.join(args.dataset, "labels_gs", root.split("/")[-1], file)
                        #     os.makedirs(os.path.dirname(gs_img_path), exist_ok=True)
                        #     cv2.imwrite(gs_img_path, gs_imgs[i*dims[0]:(i+1)*dims[0], :].cpu().numpy())
                        print(f"Time taken for image writing: {time.time() - start_time_img_write} seconds")
