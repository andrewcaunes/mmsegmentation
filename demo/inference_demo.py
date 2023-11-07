import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--config_file', help='Input file', default='configs/mask2former/mask2former_r50_8xb2-90k_apolloscape.py')
    parser.add_argument('--checkpoint_file', help='Input file', default='work_dirs/m2f_AS/best_mIoU_iter_90000.pth')
    parser.add_argument('--mmseg_path', help='Input file', default='/home/andrew/jean-zay/projects/mmsegmentation')
    parser.add_argument('--img_path', help='Input file', default='data/test_images/test_image14.png')
    parser.add_argument('--save_dir', help='output images save dir', default=None)
    

    args = parser.parse_args()

    config_file = os.path.join(args.mmseg_path, args.config_file)
    checkpoint_file = os.path.join(args.mmseg_path, args.checkpoint_file)
    img = os.path.join(args.mmseg_path, args.img_path)

    folder_mode = False
    if os.path.isdir(img):
        folder_mode = True
        imgs = sorted(os.listdir(img))

    if os.path.isdir(checkpoint_file):
        ckpts = sorted([os.path.join(checkpoint_file, file) for file in os.listdir(checkpoint_file) if file.endswith('.pth')])
    else :
        ckpts = [checkpoint_file]
    logging.info("Checkpoint files : {}".format(ckpts))
        


    # build the model from a config file and a checkpoint file
    device = 'cuda'

    for ckpt in ckpts:

        if args.save_dir is not None:
            save_dir = args.save_dir
        else:
            save_dir = os.path.join(os.path.dirname(ckpt), 'inference_results', os.path.basename(ckpt)[:-4])
            # logging.info('ckp=\n%s',ckp)
            # logging.info('os.path.basename(checkpoint_file)[:-4]=\n%s',os.path.basename(checkpoint_file)[:-4])
            # logging.info('save_dir=\n%s',save_dir)
            os.makedirs(save_dir, exist_ok=True)

        logging.info("Loading checkpoint {}".format(ckpt))
        model = init_model(config_file, ckpt, device=device)
        if not device=='gpu':
            model = revert_sync_batchnorm(model)

        if folder_mode:
            for img in imgs:
                img = os.path.join(args.mmseg_path, args.img_path, img)
                # test a single image and show the results
                result = inference_model(model, img)

                # show the results
                vis_result = show_result_pyplot(model, img, result, show=False)
                print(vis_result.shape)
                plt.imshow(vis_result)
                print("Saving to {}".format(save_dir))
                plt.imsave(os.path.join(save_dir, os.path.basename(img)), vis_result)
        else:
            result = inference_model(model, img)

            # show the results
            vis_result = show_result_pyplot(model, img, result, show=False)
            print(vis_result.shape)
            plt.imshow(vis_result)
            plt.show()
            if args.save_dir is not None:
                print("Saving to {}".format(args.save_dir))
                plt.imsave(os.path.join(args.save_dir, os.path.basename(img)), vis_result)