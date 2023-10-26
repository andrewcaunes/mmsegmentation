# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules
from mmengine.registry import init_default_scope

from mmseg.visualization import SegLocalVisualizer
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmseg.utils as mmseg_utils
import mmcv
import torch
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def get_dataset_info(dataset_name):
    # Dynamically fetch the classes and palette based on the dataset_name
    classes = getattr(mmseg_utils, f"{dataset_name}_classes", None)
    palette = getattr(mmseg_utils, f"{dataset_name}_palette", None)
    
    if classes is None or palette is None:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")
    
    return classes, palette

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()
    # init_default_scope('mmseg')

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    # print dataset attributes
    metainfo = dataset.METAINFO
    # visualizer = VISUALIZERS.build(cfg.visualizer)
    # print(f'visualizer: {visualizer}')
    # print(f'visualizer cfg: {cfg.visualizer}')
    # visualizer = SegLocalVisualizer(
    #     vis_backends=[dict(type='LocalVisBackend')],
    #     save_dir=save_dir,
    #     alpha=opacity)
    # visualizer.dataset_meta = dict(
    #     classes=model.dataset_meta['classes'],
    #     palette=model.dataset_meta['palette'])
    # visualizer.add_datasample(
    #     name=title,
    #     image=image,
    #     data_sample=result,
    #     draw_gt=draw_gt,
    #     draw_pred=draw_pred,
    #     wait_time=wait_time,
    #     out_file=out_file,
    #     show=show,
    #     withLabels=withLabels)

    progress_bar = ProgressBar(len(dataset))
    import numpy as np
    for item in dataset:
        # print(f'item inputs: {item["inputs"].shape}')
        # img = item['inputs'].permute(1, 2, 0).numpy()
        # img = img[..., [2, 1, 0]]  # bgr to rgb
        # print(f'img: {img.shape}')
        data_sample = item['data_samples'].numpy()
        # print(f'data_sample: {data_sample}')
        # print(f'data_sample: {data_sample}')
        print(f'data_sample: {data_sample}')
        print(f'data_sample: {data_sample.gt_sem_seg.cpu().data}')
        # mask = np.array(data_sample.gt_sem_seg.cpu().data).transpose(1, 2, 0)
        # # plot mask
        # # import matplotlib.pyplot as plt
        # # plt.imshow(mask)
        # # plt.show()
        # img_path = osp.basename(item['data_samples'].img_path)
        # print(f'img_path: {img_path}')

        # out_file = osp.join(
        #     args.output_dir,
        #     osp.basename(img_path)) if args.output_dir is not None else None

        # visualizer.add_datasample(
        #     name=osp.basename(img_path),
        #     image=img,
        #     data_sample=data_sample,
        #     draw_gt=True,
        #     draw_pred=False,
        #     wait_time=args.show_interval,
        #     out_file=out_file,
        #     show=not args.not_show)
        # model = init_model(args.config, device="cpu")
        # if hasattr(model, 'module'):
            # model = model.module
        # print("model", model)
        # if args.device == 'cpu':
        #     model = revert_sync_batchnorm(model)
        # test a single image
        # result = inference_model(model, args.img)
        # show the results

        # Get dataset dependant classes and palette functions
        # classes, palette = metainfo['classes'], metainfo['palette']
        # model.dataset_meta = dict(
        #     classes=classes,
        #     palette=palette)
        img = mmcv.imread(
            item['data_samples'].img_path,
            'color')
        sem_seg = mmcv.imread(
            item['data_samples'].seg_map_path,
            'unchanged')
        sem_seg = torch.from_numpy(sem_seg)
        gt_sem_seg_data = dict(data=sem_seg)
        gt_sem_seg = PixelData(**gt_sem_seg_data)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_sem_seg

        visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
            save_dir=args.output_dir)
        visualizer.dataset_meta = dict(
            classes=metainfo['classes'],
            palette=metainfo['palette'])
        visualizer.add_datasample(name="test",
            image=img,
            data_sample=data_sample,
            # draw_gt=draw_gt,
            # draw_pred=draw_pred,
            # wait_time=wait_time,
            out_file=args.output_dir,
            show=True,
            # withLabels=withLabels
            )
        progress_bar.update()


if __name__ == '__main__':
    main()
