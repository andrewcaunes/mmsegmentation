# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import numpy as np
color_palette = {i: (np.array([0,0,0])) for i in range(24)}
color_palette.update({0:  (np.array([1,1,1])), # color, is_instance_class
                1:  (np.array([0,1,0])/1.4),
                2:  (np.array([0,1,1])/1.4),
                3:  (np.array([1,0,0])/1.4),
                4:  (np.array([1,0,1])/1.4),
                5:  (np.array([1,1,0])/1.4),
                6:  (np.array([0,0.7,0.7])/1.2),
                7:  (np.array([0.8,0,0.8])/1.2),
                13: (np.array([0.5,0.2,0.7])),
                16: (np.array([0,0,0])),
                17 : (np.array([0.4,0,0.6])),
                18 : (np.array([0.6,0.1,0.4])),
                19 : (np.array([0.4,0.7,0.4])),
                20 : (np.array([0.6,0.4,0.6])),
                21 : (np.array([0.4,0.3,0.2])),
                22 : (np.array([0.6,0.7,0.2])),
                23 : (np.array([0.4,0.1,0.8])),
                })

@DATASETS.register_module()
class PC2DNusc(BaseSegDataset):
    """Mapillary Vistas Dataset.

    Dataset paper link:
    http://ieeexplore.ieee.org/document/8237796/

    v1.2 contain 66 object classes.
    (37 instance-specific)

    v2.0 contain 124 object classes.
    (70 instance-specific, 46 stuff, 8 void or crowd).

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png' for Mapillary Vistas Dataset.
    """
    METAINFO = dict(
        classes=('background', 
                'road', 
                'sidewalk', 
                'building', 
                'vegetation', 
                'terrain',
                'road marking', 
                'traffic sign',#, 'person', 'rider', 'car', 'truck', 'bus', 'train',
                "traffic light",
                "pothole",
                "manhole",
                "street light",
                "pole",
                "car",
                "wall",
                "sky",
                "pedestrian",
                "bicycle",
                "motorcycle",
                "truck",
                "bus",
                "traffic cone",
                "construction vehicle",
                "barrier"),
                 #'motorcycle', 'bicycle'),
        palette=[list((color*255).astype(float)) for ind,color in color_palette.items()],
        # palette = list(np.random.rand(24,3)*255),
        label_map={
            0:0, # initially background is all classes not in global dict
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            8:8,
            9:9,
            10:10,
            11:11,
            12:12,
            13:13,
            14:14,
            15:15,
            16:16,
            17:17,
            18:18,
            19:19,
            20:20,
            21:21,
            22:22,
            23:23,
            255:0 # make the actual background (non point pixels) also background
        }
    )
        
    
        # classes=('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail',
        #          'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain',
        #          'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track',
        #          'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building',
        #          'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
        #          'Other Rider', 'Lane Marking - Crosswalk',
        #          'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
        #          'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
        #          'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
        #          'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole',
        #          'Phone Booth', 'Pothole', 'Street Light', 'Pole',
        #          'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
        #          'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
        #          'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
        #          'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
        #          'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled'),
        # palette=[[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
        #          [180, 165, 180], [90, 120, 150], [102, 102, 156],
        #          [128, 64, 255], [140, 140, 200], [170, 170, 170],
        #          [250, 170, 160], [96, 96, 96],
        #          [230, 150, 140], [128, 64, 128], [110, 110, 110],
        #          [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90],
        #          [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200],
        #          [200, 128, 128], [255, 255, 255], [64, 170,
        #                                             64], [230, 160, 50],
        #          [70, 130, 180], [190, 255, 255], [152, 251, 152],
        #          [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
        #          [100, 140, 180], [220, 220, 220], [220, 128, 128],
        #          [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33],
        #          [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100],
        #          [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30],
        #          [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32],
        #          [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
        #          [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
        #          [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10,
        #                                                  10], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

