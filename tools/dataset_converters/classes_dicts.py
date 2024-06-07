import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)

mapillary_nbs_classes = ('background', 
                            'road', 
                            'sidewalk', 
                            'building', 
                            'vegetation', 
                            'terrain', 
                            'sky')

mapillary_ext_classes=('background',
            'road',
            'sidewalk',
            'building',
            'vegetation',
            'terrain',
            'sky',
            'road marking',
            'traffic sign',
            'traffic light',
            'pothole',
            'manhole',
            'street light',
            'pole',
            'vehicle',
            'wall')

global_classes_dict={
    "background":0,
    "road":1,
    "sidewalk":2,
    "building":3,
    "vegetation":4,
    "terrain":5,
    "road marking":6,
    "traffic sign":7,
    "traffic light":8,
    "pothole":9,
    "manhole":10,
    "street light":11,
    "pole":12,
    "vehicle":13,
    "wall":14,
    "sky":15,
    "pedestrian":16,
}

uda_classes_dict={
    "background":0, # not reported
    "car":1,
    "bicycle":2,
    "motorcycle":3,
    "truck":4,
    "other vehicle":5, # In T-UDA, but o-vehicle in LIDAR-UDA
    "person":6,
    "drivable":7,
    "sidewalk":8,
    "terrain":9,
    "vegetation":10,
    "manmade": 11, # in T-UDA only
    # "fence"
}

global_ext_classes_dict={
    "background":0,
    "road":1,
    "sidewalk":2,
    "building":3,
    "vegetation":4,
    "terrain":5,
    "road marking":6,
    "traffic sign":7,
    "traffic light":8,
    "pothole":9,
    "manhole":10,
    "street light":11,
    "pole":12,
    "car":13,
    "wall":14,
    "sky":15,
    "pedestrian":16,
    "bicycle":17,
    "motorcycle":18,
    "truck":19,
    "bus":20,
    "traffic cone":21,
    "construction vehicle":22,
    "barrier":23,
}

# for evaluating on merged classes
global_to_merged_dict={i:i for i in range(32)}
global_to_merged_dict = {2:2, 5:2, 15:0, 10:1, 9:1, 14:3}

# global_classes_dict={
#     "background":0,
#     "road":1,
#     "sidewalk":2,
#     "building":3,
#     "vegetation":4,
#     "terrain":5,
#     "road marking":6,
#     "traffic sign":7,
#     "traffic light":8,
#     "pothole":9,
#     "manhole":10,
#     "street light":11,
#     "pole":12,
#     "vehicle":13,
#     "wall":14,
#     "sky":15
# }
global_classes_dict_reversed={v:k for k,v in global_classes_dict.items()}

# global_to_lidarseg_dict = {
#     0:27, # background -> flat.other
#     1:24, # road -> flat.driveable_surface
#     2:25, # sidewalk -> flat.sidewalk
#     3:28, # building -> static.manmade
#     4:29, # vegetation -> static.vegetation 
#     5:26, # terrain -> flat.terrain
#     6:24, # road marking -> flat.driveable_surface # Nuscenes does not annotate road markings
#     7:28, # traffic sign -> static.manmade
#     8:28, # traffic light -> static.manmade
#     9:24, # pothole -> flat.driveable_surface
#     10:24, # manhole -> flat.driveable_surface
#     11:28, # street light -> static.manmade
#     12:28, # pole -> static.manmade
#     13:1, # vehicle -> vehicle
#     14:28, # wall -> static.manmade
#     15:30 # sky -> flat.other
# }
global_to_lidarseg_dict = {
    # Built using the mmdet3d mappings because
    # nuscenes annotator instructions seem outdated
    0:25, # background -> flat.other
    1:24, # road -> flat.driveable_surface
    2:26, # sidewalk -> flat.sidewalk
    3:28, # building -> static.manmade
    4:30, # vegetation -> static.vegetation 
    5:27, # terrain -> flat.terrain
    6:24, # road marking -> flat.driveable_surface # Nuscenes does not annotate road markings
    7:28, # traffic sign -> static.manmade
    8:28, # traffic light -> static.manmade
    9:24, # pothole -> flat.driveable_surface
    10:24, # manhole -> flat.driveable_surface
    11:28, # street light -> static.manmade
    12:28, # pole -> static.manmade
    13:1, # vehicle -> vehicle
    14:28, # wall -> static.manmade
    15:25 # sky -> flat.other
}

lidarseg_classes_dict_reversed={i:"background" for i in range(32)}
# lidarseg_classes_dict_reversed.update({
#     "1" : "vehicle",
#     "2" : "vehicle",
#     "3" : "vehicle",
#     "4" : "vehicle",
#     "5" : "vehicle",
#     "6" : "vehicle",
#     "7" : "vehicle",
#     "8" : "vehicle",
#     "9" : "vehicle",
#     "10" : "vehicle",
#     "11" : "vehicle",
#     "24" : "road",
#     "25" : "sidewalk",
#     "26" : "terrain",
#     "27" : "terrain",
#     "28" : "building",
#     "29" : "vegetation",
#     "30" : "vegetation",
# })
lidarseg_classes_dict_reversed.update({
    "2": "pedestrian",
    "3": "pedestrian",
    "4": "pedestrian",
    "6": "pedestrian",
    "14" : "vehicle", # bicycle
    "15" : "vehicle", # bus
    "16" : "vehicle", # bus
    "17" : "vehicle", # car
    "18" : "vehicle", # construction vehicle
    "21" : "vehicle", # motorcycle
    "22" : "vehicle", # trailer
    "23" : "vehicle", # truck
    "24" : "road", # driveable surface
    "26" : "sidewalk", # sidewalk
    "25" : "terrain", # other_flat
    "27" : "terrain", # terrain
    "28" : "building", # manmade
    "29" : "vegetation", # other static
    "30" : "vegetation", # vegetation
})

lidarseg_ext_classes_dict_reversed={i:"background" for i in range(32)}
lidarseg_ext_classes_dict_reversed.update({
    "2": "pedestrian",
    "3": "pedestrian",
    "4": "pedestrian",
    "6": "pedestrian",
    "9": "barrier",
    "12": "traffic cone",
    "14" : "bicycle", # bicycle
    "15" : "bus", # bus
    "16" : "bus", # bus
    "17" : "car", # car
    "18" : "construction vehicle", # construction vehicle
    "21" : "motorcycle", # motorcycle
    "22" : "truck", # trailer
    "23" : "truck", # truck
    "24" : "road", # driveable surface
    "26" : "sidewalk", # sidewalk
    "25" : "terrain", # other_flat
    "27" : "terrain", # terrain
    "28" : "building", # manmade
    "29" : "vegetation", # other static
    "30" : "vegetation", # vegetation
})

semantickitti_classes_dict_reversed={i:"background" for i in range(32)}
semantickitti_classes_dict_reversed.update({
    0 : "background", #"unlabeled",
    1 : "background",#"outlier",
    10: "car",#"car",
    11: "bicycle",#"bicycle",
    13: "bus",#"bus",
    15: "motorcycle",#"motorcycle",
    16: "truck",#"on-rails",
    18: "truck",#"truck",
    20: "truck",#"other-vehicle",
    30: "pedestrian",#"person",
    31: "bicycle",#"bicyclist",
    32: "motorcycle",#"motorcyclist",
    40: "road",#"road",
    44: "road",#"parking",
    48: "sidewalk",#"sidewalk",
    49: "terrain",#"other-ground",
    50: "building",#"building",
    51: "barrier",#"fence",
    52: "building",#"other-structure",
    60: "road marking",#"lane-marking",
    70: "vegetation",#"vegetation",
    71: "vegetation",#"trunk",
    72: "terrain",#"terrain",
    80: "pole",#"pole",
    81: "traffic sign",#"traffic-sign",
    99: "building",#"other-object",
    252: "car",#"moving-car",
    253: "bicycle",#"moving-bicyclist",
    254: "pedestrian",#"moving-person",
    255: "motorcycle",#"moving-motorcyclist",
    256: "truck",#"moving-on-rails",
    257: "bus",#"moving-bus",
    258: "truck",#"moving-truck",
    259: "truck",#"moving-other-vehicle",
})

mapillary_nbs_to_global_dict={i:global_classes_dict[mapillary_nbs_classes[i]] for i in range(len(mapillary_nbs_classes))}
mapillary_ext_to_global_dict={i:global_classes_dict[mapillary_ext_classes[i]] for i in range(len(mapillary_ext_classes))}

lidarseg_to_global_dict={int(i):global_classes_dict[lidarseg_classes_dict_reversed[i]] for i in lidarseg_classes_dict_reversed}
lidarseg_ext_to_global_dict={int(i):global_ext_classes_dict[lidarseg_ext_classes_dict_reversed[i]] for i in lidarseg_ext_classes_dict_reversed}
semantickitti_to_global_dict={int(i):global_ext_classes_dict[semantickitti_classes_dict_reversed[i]] for i in semantickitti_classes_dict_reversed}

import numpy as np
color_palette_float = {0:  (np.array([0,0,0])), # color, is_instance_class
                1:  (np.array([0,1,0])/1.4),
                2:  (np.array([0,1,1])/1.4),
                3:  (np.array([1,0,0])/1.4),
                4:  (np.array([1,0,1])/1.4),
                5:  (np.array([1,1,0])/1.4),
                6:  (np.array([0.8,1.1,1.1])/1.2),
                7:  (np.array([0.8,0,0.8])/1.2),
                8:  (np.array([0.8,0.8,0])/1.2),
                9:  (np.array([0.8,0.5,0])/1.2),
                10: (np.array([0.8,0,0])/1.2),
                11: (np.array([0.8,0.5,0.5])/1.2),
                12: (np.array([0.2,0.8,0.4])),
                13: (np.array([0.5,0.2,0.7])),
                14: (np.array([0.2,0.8,0.8])),
                15: (np.array([0.8,0.8,0.8])),
                16: (np.array([0.1,0.9,0.3])),
                17 : (np.array([0.4,0,0.6])),
                18 : (np.array([0.6,0.1,0.4])),
                19 : (np.array([0.4,0.7,0.4])),
                20 : (np.array([0.6,0.4,0.6])),
                21 : (np.array([0.4,0.3,0.2])),
                22 : (np.array([0.6,0.7,0.2])),
                23 : (np.array([0.4,0.1,0.8])),
            }
color_palette = np.array([[  0,   0,   0],  # color,  is_instance_class
                          [  0, 182,   0],  
                          [  0, 182, 182],  
                          [182,   0,   0],  
                          [182,   0, 182],  # was a problem with 182 -> 82
                          [182, 182,   0],  
                          [170, 233, 233],  
                          [170,   0, 170],  
                          [170, 170,   0],  
                          [170, 127,   0],  
                          [170,   0,   0],  
                          [170, 127, 127],  
                          [ 51, 204, 102],  
                          [128,  51, 178],  # 127 -> 128
                          [ 51, 204, 204],  
                          [204, 204, 204],  
                          [ 26, 229,  77],  # 25 -> 26, 76 -> 77
                          [102,   0, 153], 
                          [153,  26, 102],  # 25 -> 26
                          [102, 178, 102], 
                          [153, 102, 153], 
                          [102,  77,  51], # 76 -> 77 
                          [153, 178,  51], 
                          [102,  25, 204]], dtype=np.uint8) 
            

# color_palette = {0:  (np.array([  0,   0,   0], dtype=np.uint8)),  # color,  is_instance_class
#                 1:  (np.array( [  0, 182,   0], dtype=np.uint8)), 
#                 2:  (np.array( [  0, 182, 182], dtype=np.uint8)), 
#                 3:  (np.array( [182,   0,   0], dtype=np.uint8)), 
#                 4:  (np.array( [182,   0,  82], dtype=np.uint8)), 
#                 5:  (np.array( [182, 182,   0], dtype=np.uint8)), 
#                 6:  (np.array( [170, 233, 233], dtype=np.uint8)), 
#                 7:  (np.array( [170,   0, 170], dtype=np.uint8)), 
#                 8:  (np.array( [170, 170,   0], dtype=np.uint8)), 
#                 9:  (np.array( [170, 127,   0], dtype=np.uint8)), 
#                 10: (np.array( [170,   0,   0], dtype=np.uint8)), 
#                 11: (np.array( [170, 127, 127], dtype=np.uint8)), 
#                 12: (np.array( [ 51, 204, 102], dtype=np.uint8)), 
#                 13: (np.array( [127,  51, 178], dtype=np.uint8)), 
#                 14: (np.array( [ 51, 204, 204], dtype=np.uint8)), 
#                 15: (np.array( [204, 204, 204], dtype=np.uint8)), 
#                 16: (np.array( [ 25, 229,  76], dtype=np.uint8)), 
#                 17 : (np.array([102,   0, 153], dtype=np.uint8)), 
#                 18 : (np.array([153,  25, 102], dtype=np.uint8)), 
#                 19 : (np.array([102, 178, 102], dtype=np.uint8)), 
#                 20 : (np.array([153, 102, 153], dtype=np.uint8)), 
#                 21 : (np.array([102,  76,  51], dtype=np.uint8)), 
#                 22 : (np.array([153, 178,  51], dtype=np.uint8)), 
#                 23 : (np.array([102,  25, 204], dtype=np.uint8)), 
#             }