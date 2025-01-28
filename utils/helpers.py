import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from skimage import transform

import os
import shutil

import logging
import sys


def set_seed(seed, deterministic):
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_snapshot_path(args):
    snapshot_path = "./model/{}_{}/{}".format(args.exp, args.fold, args.sup_type)
    return snapshot_path

def setup_snapshot(args):
    snapshot_path = get_snapshot_path(args)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # def remove_readonly(func, path, excinfo):
    #     import stat

    #     os.chmod(path, stat.S_IWRITE)
    #     func(path)

    # if os.path.exists(snapshot_path + "/code"):
    #     shutil.rmtree(snapshot_path + "/code", onerror=remove_readonly)
    # shutil.copytree(
    #     ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    # )
    return snapshot_path

def setup_logging(args):
    snapshot_path = get_snapshot_path(args)
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
def process_images(volume_batch, device):
    processed_images = []
    for img_np in volume_batch.cpu().numpy():
        img_np =img_np.reshape(256, 256)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # Normalize to [0, 1], (H, W, 3)
        # Convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        processed_images.append(img_1024_tensor)
    return processed_images

def check_for_nan(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logging.error("Model input contains NaN or Inf values")
        raise ValueError("Model input contains NaN or Inf values")


def write_scalars(writer, path, scalars, iter_num):
    for key in scalars:
        writer.add_scalar(f'{path}/{key}', scalars[key], iter_num)
    
def log_epoch(path, writer, iter_num, scalars, volume_batch, outputs_soft, label_batch, edges_list):
    write_scalars(writer, path, scalars, iter_num)
    if path == "train":        
        for batch_idx, class_id, edges in edges_list:
            if class_id == 0 and batch_idx == 0 and iter_num % 20 == 0:
                edges_tensor = torch.tensor(edges).to(torch.uint8)
                writer.add_image(f'{path}/edges_class_{class_id}_batch_{batch_idx}', edges_tensor, iter_num, dataformats='HW')
                break
    if iter_num % 20 == 0:
        image = volume_batch[1, 0:1, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        writer.add_image(f'{path}/Image', image, iter_num)
        outputs_vis = torch.argmax(outputs_soft, dim=1)
        writer.add_image(f'{path}/Prediction', outputs_vis[1, ...].unsqueeze(0) * 50, iter_num)
        labs = label_batch[1, ...].unsqueeze(0) * 50
        writer.add_image(f'{path}/GroundTruth', labs, iter_num)
        
        