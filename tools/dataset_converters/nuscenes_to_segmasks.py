"""
Script made by Andrew Caunes.
Generate nuscenes annotations as segmentation masks (images with grescale values corresponding to classes) + a classes dict in json format.
Example use:
python nuscenes_to_segmasks.py --root_path /path/to/nuscenes --version v1.0-trainval --output_path /path/to/output
"""
import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np
from nuscenes.nuscenes import NuScenes

def main(args):
    logging.info("args = %s", args)
    kwargs = vars(args)
    generate_segmasks(**kwargs)

def generate_segmasks(**kwargs):
    root_path = kwargs['root_path']
    version = kwargs['version']
    output_path = kwargs['output_path']
    if not os.path.exists(root_path):
        raise ValueError("root_path does not exist")
    if not os.path.exists(output_path):
        logging.info("output_path does not exist, creating it")
        os.makedirs(output_path)

    if output_path is None:
        output_path = os.path.join(root_path, 'segmasks')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
        
    nusc.list_lidarseg_categories(sort_by='count')
    sample = nusc.sample[0]
    sample_token = sample['token']
    nusc.get_sample_lidarseg_stats(sample_token, sort_by='count')
    sample_lidarseg = nusc.get('lidarseg', sample_token)
    # for sample in nusc.sample:
    #     sample_token = sample['token']
    #     sample_name = sample['filename']
    #     height = sample['height']
    #     width = sample['width']

    #     # get annotation
    #     nusc_ann = nusc.get('lidarseg', sample_token)
    #     logging.info('nusc_ann=\n%s',nusc_ann)

#     sample = nusc.sample[0]
    # # go to next sample
    # for i in range(1):
    #     sample = nusc.get('sample', sample['next'])
    # lidar_token = sample['data']['LIDAR_TOP']
    # lidar_data = nusc.get('sample_data', lidar_token)
    # lidar_pointcloud = LidarPointCloud.from_file(f"{nusc.dataroot}/{lidar_data['filename']}")
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(lidar_pointcloud.points[:3, :].T)
    # intensities = lidar_pointcloud.points[3, :]

    # max_cap = 200
    # intensities = np.clip(intensities, 0, max_cap)



    # normalized_intensities = np.interp(intensities, (intensities.min(), intensities.max()), (0, 1)).astype(float)
    # colors = np.zeros((intensities.shape[0], 3))
    # colors[:, :] = normalized_intensities.reshape(-1, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--root_path', help='', default=None)
    parser.add_argument('--output_path', help='', default=None)
    parser.add_argument('--version', help='', default="v1.0-mini")
    args = parser.parse_args()

    main(args)