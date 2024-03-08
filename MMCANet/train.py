import torch
import logging
import numpy as np
from PIL import Image
from Code.utils.tools import *
from tqdm import tqdm
import pdb, os, argparse
from datetime import datetime
import torch.nn.functional as F

from Code.lib.MMCANet import MMCANet

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from Code.utils.data import get_loader, test_dataset

from Code.utils.util import clip_gradient, adjust_lr
import Code.utils.logs as log_utils
from torchvision.utils import make_grid

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=100, help="epoch number")
parser.add_argument(
    "--model_name", type=str, default="full_supervised", help="model name"
)  
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--batchsize", type=int, default=22, help="training batch size")
parser.add_argument("--trainsize", type=int, default=256, help="training dataset size")
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
parser.add_argument(
    "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
)
parser.add_argument(
    "--decay_epoch", type=int, default=40, help="every n epochs decay learning rate"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/media/data2/lcl_e/lcz/lcz/MMCANet/models/",
    help="the path to save models and logs",
) 
parser.add_argument(
    "--load",
    type=str,
    default="/media/data2/lcl_e/lcz/lcz/MMCANet/Checkpoints/ckpt_S.pth",
    help="train from checkpoints",
)
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
opt = parser.parse_args()


model_name = opt.model_name
model = MMCANet()

print("加载模型中...")
model.encoder_rgb.load_state_dict(torch.load(opt.load), strict=False)
model.encoder_depth.load_state_dict(torch.load(opt.load), strict=False)
print("模型加载成功！")

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model, device_ids)
if torch.cuda.is_available():
    model = model.to(device)

model.to(device)


image_root = "/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/datasets/RGB/"
depth_root = "/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/datasets/T/"
gt_root = "/media/data2/lcl_e/lcz/lcz/RGBT_scribble/pesudal_labels_generation/final_labels/"  # pseudo labels

gt_scribble_root = (
    "/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/datasets/gt/"  # scribble labels
)
mask_root = "/media/data2/lcl_e/lcz/lcz/RGBT_scribble/train_data/scribble-supervisedRGBT_data/datasets/mask/"

test_image_root = "/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/RGB/"
test_gt_root = "/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/GT/"
test_depth_root = "/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/T/"

save_path = opt.save_path
# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
logger = log_utils.get_logger(
    save_path + "train-{}-{:s}.log".format(model_name, time_str)
)
log_utils.print_config(vars(opt), logger)

print("加载数据中...")
train_loader = get_loader(
    image_root,
    gt_root,
    depth_root,
    gt_scribble_root,
    mask_root,
    batchsize=opt.batchsize,
    trainsize=opt.trainsize,
)
test_loader = test_dataset(
    test_image_root, test_gt_root, test_depth_root, opt.trainsize
)
# test_loader = test_get_loader(test_image_root, test_gt_root, test_depth_root, batchsize=1, testsize=256)
print("数据加载成功！")
total_step = len(train_loader)

best_mae = 1
best_epoch = 0
step = 0

CE = torch.nn.BCELoss()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths, gt_scribbles, masks) in enumerate(
            train_loader, start=1
        ):
            optimizer.zero_grad()

            images = images.to(device)
            gts = gts.to(device)
            depths = depths.to(device)
            gt_scribbles = gt_scribbles.to(device)
            masks = masks.to(device)

            result_final = model(images, depths)

            img_size = images.size(2) * images.size(3) * images.size(0)  # 图片的像素数量
            ratio = img_size / torch.sum(masks)  # 总像素数量除以有效像素点数量
            final_prob = torch.sigmoid(result_final)
            final_prob = final_prob * masks  # 过滤背景像素
            pce_loss = ratio * CE(final_prob, gt_scribbles * masks)

            sal_loss = structure_loss(result_final, gts)

      

            loss = sal_loss + pce_loss

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f}, pce_loss:{:4f}, sl_loss:{:4f}".format(
                        datetime.now(),
                        epoch,
                        opt.epoch,
                        i,
                        total_step,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        loss.data,
                        pce_loss.data,
                        sal_loss.data,
                    )
                )
                logger.info(
                    "#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f}, pce_loss:{:4f}, sl_loss:{:4f} ".format(
                        epoch,
                        opt.epoch,
                        i,
                        total_step,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        loss.data,
                        pce_loss.data,
                        sal_loss.data,
                    )
                )
                

        loss_all /= epoch_step
        logger.info(
            "#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}".format(
                epoch, opt.epoch, loss_all
            )
        )

        if epoch >= 30:
            torch.save(
                model.state_dict(),
                save_path + "{}_epoch_{}.pth".format(model_name, epoch),
            )

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt: save model and exit.")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(
            model.state_dict(),
            save_path + "{}_epoch_{}.pth".format(model_name, epoch + 1),
        )
        logger.info("save checkpoints successfully!")
        raise


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8
            image = image.to(device)
            # depth = depth.repeat(1, 3, 1, 1, ).cuda()
            depth = depth.to(device)
            res = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print(
            "Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}".format(
                epoch, mae, best_mae, best_epoch
            )
        )
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    save_path + "{}_epoch_best.pth".format(model_name),
                )
                print("best epoch:{}".format(epoch))
        logger.info(
            "#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}".format(
                epoch, mae, best_epoch, best_mae
            )
        )


print("Starting!")
for epoch in range(1, opt.epoch + 1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch, save_path)
    if epoch >= 30:
        test(test_loader, model, epoch, save_path)
