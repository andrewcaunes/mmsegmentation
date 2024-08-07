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
    print(f'dataset: {dataset}')
    metainfo = dataset.METAINFO

    progress_bar = ProgressBar(len(dataset))
    import numpy as np
    for item in dataset:
        img = item['inputs'].permute(1, 2, 0).numpy()
        img = img[..., [2, 1, 0]] 
         # bgr to rgb
        data_sample = item['data_samples'].numpy()
        sem_seg = torch.from_numpy(data_sample.gt_sem_seg.cpu().data.transpose(1, 2, 0)).squeeze()
        gt_sem_seg_data = dict(data=sem_seg)
        gt_sem_seg = PixelData(**gt_sem_seg_data)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_sem_seg
        print(f'img: {img.shape}')
        print(f'data_sample: {data_sample}')

        visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
            save_dir=args.output_dir)
        visualizer.dataset_meta = dict(
            classes=metainfo['classes'],
            palette=metainfo['palette'])
        visualizer.add_datasample(name="test",
            image=img,
            data_sample=data_sample,
            # draw_gt=False,
            # draw_pred=draw_pred,
            # wait_time=wait_time,
            out_file=args.output_dir,
            show=True,
            # withLabels=withLabels
            )
        progress_bar.update()


if __name__ == '__main__':
    main()
