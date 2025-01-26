import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
# from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
# import networks.MedSAM_Inference
from networks.MedSAM_Inference import medsam_model, medsam_inference
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
# from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC_pCE', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
# progressmix unet
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')

parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()

def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")

    

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            # # med-sam guided
            processed_images = []
            for img_np in volume_batch.cpu().numpy():
                img_np =img_np.reshape(256, 256)
            

                if len(img_np.shape) == 2:
                    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_np
                H, W, _ = img_3c.shape
                # Image preprocessing
                from skimage import io, transform
                img_1024 = transform.resize(
                    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                ).astype(np.uint8)
                img_1024 = (img_1024 - img_1024.min()) / np.clip(
                    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
                )  # Normalize to [0, 1], (H, W, 3)
                # Convert the shape to (3, H, W)
                img_1024_tensor = (
                    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).cuda()
                )
                # generate processed_images
                processed_images.append(img_1024_tensor)

            volume_batch = volume_batch.cuda()
            label_batch = label_batch.cuda()

            # ProgressMix model
            # 检查模型输入是否包含 NaN 或 Inf 值
            if torch.isnan(volume_batch).any() or torch.isinf(volume_batch).any():
                logging.error("Model input contains NaN or Inf values")
                raise ValueError("Model input contains NaN or Inf values")

            outputs = model(volume_batch)  # generate coarse prediction
            outputs_soft = torch.softmax(outputs, dim=1)

            # 提取 coarse prediction 的边界框和边缘检测
            outputs_np = torch.argmax(outputs_soft, dim=1).cpu().numpy()  # Shape: (B, H, W)
            bounding_boxes = []
            edges_list = []

            medsam_seg_list = []
            for class_id in range(num_classes):  # 遍历每个类别
                medsam_seg_batch_list = []
                for batch_idx in range(outputs_np.shape[0]):  # 遍历 batch 中的每个样本
                    class_mask = (outputs_np[batch_idx] == class_id).astype(np.uint8)  # 类别 mask

                    if class_mask.any():  # 如果类别 mask 不为空
                        # 提取 bounding box
                        import cv2
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x, y, w, h = cv2.boundingRect(contours[0])
                        bounding_boxes.append([batch_idx, class_id, x, y, w, h])

                        # 边缘检测
                        edges = cv2.Canny(class_mask * 255, 100, 200)
                        edges_list.append((batch_idx, class_id, edges))

                        # 记录到 TensorBoard
                        box_np = np.array([x, y, x + w, y + h])
                        box_1024 = box_np / np.array([W, H, W, H]) * 1024  # 转换到 1024x1024 尺度

                        # 使用 med-sam 生成分割结果
                        with torch.no_grad():
                            image_embedding = medsam_model.image_encoder(processed_images[batch_idx])  
                            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024.reshape(1,4), H, W)

                        # 将结果记录到 TensorBoard
                        medsam_seg_tensor = torch.tensor(medsam_seg).unsqueeze(0).float().to(label_batch.device)  # (1, H, W)
                        medsam_seg_batch_list.append(medsam_seg_tensor)
                        writer.add_image(f'train/medsam_seg_class_{class_id}_batch_{batch_idx}', medsam_seg_tensor,
                                         iter_num)

                        print(f"Batch {batch_idx}, Class {class_id}: Bounding Box (1024x1024 scale): {box_1024}")
                    else:
                        # 如果类别 mask 为空，添加一个全零张量作为占位符
                        empty_seg_tensor = torch.zeros((1, H, W), dtype=torch.float).to(label_batch.device)
                        medsam_seg_batch_list.append(empty_seg_tensor)
                medsam_seg_list.append(medsam_seg_batch_list)
            # 记录边缘检测到 TensorBoard
            for batch_idx, class_id, edges in edges_list:
                edges_tensor = torch.tensor(edges).unsqueeze(0).float()  # (1, 1, H, W)
                writer.add_image(f'train/edges_class_{class_id}_batch_{batch_idx}', edges_tensor, iter_num)

            # 损失计算和优化
            loss_ce = ce_loss(outputs, label_batch[:].long())

            # Add medsam segmentation loss to the total loss
            medsam_seg_output = torch.cat([torch.cat(medsam_seg_class_list, dim=0) for medsam_seg_class_list in medsam_seg_list], dim=0).view(batch_size, num_classes, 256, 256).to(label_batch.device)
            medsam_seg_loss = ce_loss(medsam_seg_output, label_batch[:].long())

            # Total loss includes both coarse prediction and medsam_seg_loss
            loss = loss_ce + medsam_seg_loss
            # loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/medsam_seg_loss', medsam_seg_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, medsam_seg_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), medsam_seg_loss.item())
            )
            # logging.info(
            #     'iteration %d : loss : %f, loss_ce: %f' %
            #     (iter_num, loss.item(), loss_ce.item())
            # )

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    def remove_readonly(func, path, excinfo):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code',onerror=remove_readonly)
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
