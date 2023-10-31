import torch
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot

config_file = '../configs/mask2former/mask2former_r50_8xb2-90k_apolloscape.py'
checkpoint_file = '../work_dirs/m2f_AS/best_mIoU_iter_90000.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda')

# test a single image
import os
imgs_path = '../data/test_images'
img_list = os.listdir(imgs_path)
img = os.path.join(imgs_path, "test_image14.png")
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)
result = inference_model(model, img)

# show the results
vis_result = show_result_pyplot(model, img, result, show=True)
print(vis_result.shape)
plt.imshow(vis_result)