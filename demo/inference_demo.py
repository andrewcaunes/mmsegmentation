import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from tqdm import tqdm
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmengine import Config
import os
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--config_file', help='Input file', default='configs/mask2former/mask2former_r50_8xb2-90k_apolloscape.py')
    parser.add_argument('--checkpoint_file', help='Input file', default='work_dirs/m2f_AS/best_mIoU_iter_90000.pth')
    parser.add_argument('--mmseg_path', help='Input file', default='/home/andrew/jean-zay/projects/mmsegmentation')
    parser.add_argument('--folder_path', help='Input file', default='data/test_images/test_image14.png')
    parser.add_argument('--save_dir', help='output images save dir', default=None)
    parser.add_argument('--device', help='device', default='cuda')
    

    args = parser.parse_args()

    config_file = os.path.join(args.mmseg_path, args.config_file)
    # show config
    logging.info(Config.fromfile(config_file))
    checkpoint_file = os.path.join(args.mmseg_path, args.checkpoint_file)
    device = args.device
    if args.folder_path.startswith('/'):
        folder_path = args.folder_path
    else:
        folder_path = os.path.join(args.mmseg_path, args.folder_path)

    folder_mode = False
    if os.path.isdir(folder_path):
        folder_mode = True
        imgs_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]

    if os.path.isdir(checkpoint_file):
        ckpts = sorted([os.path.join(checkpoint_file, file) for file in os.listdir(checkpoint_file) if file.endswith('.pth')])
    else :
        ckpts = [checkpoint_file]
    logging.info("Checkpoint files : {}".format(ckpts))
        

    for ckpt in ckpts:
        logging.info('ckpt=\n%s',ckpt)
        if args.save_dir is not None:
            save_dir = args.save_dir
        else:
            save_dir = os.path.join(os.path.dirname(ckpt), 'inference_results', os.path.basename(ckpt)[:-4])
            if folder_path.endswith('/'):
                save_dir = os.path.join(save_dir, os.path.basename(folder_path[:-1]))
            else:
                save_dir = os.path.join(save_dir, os.path.basename(folder_path))
            # logging.info('os.path.basename(checkpoint_file)[:-4]=\n%s',os.path.basename(checkpoint_file)[:-4])
            logging.info('save_dir=\n%s',save_dir)
            os.makedirs(save_dir, exist_ok=True)

        logging.info("Loading checkpoint {}".format(ckpt))
        model = init_model(config_file, ckpt, device=device)
        if not device=='cuda':
            model = revert_sync_batchnorm(model)
        # logging.info('imgs_path=\n%s',imgs_path)
        if folder_mode:
            try:
                imgs_path = sorted(imgs_path, key=lambda x: int(os.path.basename(x).split('.')[0]))
            except:
                imgs_path = sorted(imgs_path)
            np.random.shuffle(imgs_path)
            for img_path in tqdm(imgs_path):
                # logging.info('img_path=\n%s',img_path)
                img_path = os.path.join(args.mmseg_path, args.folder_path, img_path)
                img_save_path = os.path.join(save_dir, os.path.basename(img_path))
                # logging.info('img_save_path=\n%s',img_save_path)
                if os.path.exists(img_save_path):
                    continue
                # test a single image and show the results
                result = inference_model(model, img_path)
                # logging.info('result=\n%s',result)

                # show the results
                vis_result = show_result_pyplot(model, img_path, result, show=False)
                plt.imshow(vis_result)
                # print("Saving to {}".format(save_dir))
                plt.imsave(img_save_path, vis_result)
        else:
            result = inference_model(model, img_path)

            # show the results
            vis_result = show_result_pyplot(model, img_path, result, show=False)
            print(vis_result.shape)
            plt.imshow(vis_result)
            plt.show()
            if args.save_dir is not None:
                img_save_path = os.path.join(args.save_dir, os.path.basename(img_path))
                # logging.info("Saving to {}".format(img_save_path))
                plt.imsave(os.path.join(args.save_dir, os.path.basename(img_path)), vis_result)