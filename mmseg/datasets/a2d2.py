# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ApolloscapeDataset(BaseSegDataset):
    """ApolloScape dataset."""
    METAINFO = dict(
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
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_bin.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
