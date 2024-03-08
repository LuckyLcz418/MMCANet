import torch
import torch.nn.functional as F
import numpy as np

import cv2

from Code.lib.MMCANet import MMCANet

from Code.utils.data import test_dataset
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# /media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT821_unalign/
parser = argparse.ArgumentParser()
# parser.add_argument('--trainsize', type=int, default=256, help='testing size')
parser.add_argument("--testsize", type=int, default=256, help="testing size")
# parser.add_argument('--gpu_id', type=str, default='3', help='select gpu id')
parser.add_argument(
    "--test_path",
    type=str,
    default="/media/data2/lcl_e/wkp/datasets/SOD/RGBT/",
    help="test dataset path",
)


# parser.add_argument('--test_path', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/challenge-scene/', help='challenge test dataset path')

# parser.add_argument('--testsize', type=int, default=256, help='testing size')

# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
# parser.add_argument('--test_path', type=str, default='/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/',
#                      help='test dataset path')

# parser.add_argument('--test_path', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/',
#  help='test dataset path')

# parser.add_argument('--test_path', type=str, default='/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/datasets/visual_image/',
#                      help='test dataset path')
# parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
opt = parser.parse_args()

dataset_path = opt.test_path

# load the model
model = MMCANet()
# model = CMCSNet()
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(
    torch.load(
        "/media/data2/lcl_e/lcz/lcz/MMCANet/models/MMCANet.pth"
    )
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids)
if torch.cuda.is_available():
    model = model.to(device)
model.to(device)
model.eval()

# test
# test_datasets = [
#     "VT821_unalign",
#     "VT1000_unalign",
#     "VT5000-Test_unalign",
#     "VT2000_unalign",
# ]
test_datasets = [
    "VT821",
    "VT1000",
    "VT5000-Test",
]
# test_datasets = ['BSO', 'CB', 'CIB', 'IC', 'LI', 'MSO', 'OF', 'SSO', 'SA', 'TC', 'BW']
# test_datasets = ['NJU2K']

for dataset in test_datasets:
    save_path = (
        "/media/data2/lcl_e/lcz/lcz/MMCANet/testmaps_/"
        + dataset
        + "/"
    )
    # save_path = './test_maps_RGBD_1/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + "/" + "RGB" + "/"
    gt_root = dataset_path + dataset + "/" + "GT" + "/"
    depth_root = dataset_path + dataset + "/" + "T" + "/"
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= gt.max() + 1e-8
        image = image.to(device)
        depth = depth.to(device)
        res = model(image, depth)
        res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print("save img to: ", save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    print("Test Done!")
