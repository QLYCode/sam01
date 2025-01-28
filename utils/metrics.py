import numpy as np
from medpy import metric

import torch
import torch.nn.functional as F


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_score(output, target, epsilon=1e-6):
    B, C, H, W = output.shape
    
    target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    if output.shape[1] == C:
        output_softmax = output
    else:
        output_softmax = F.softmax(output, dim=1)

    output_flat = output_softmax.view(B, C, -1)
    target_flat = target_one_hot.view(B, C, -1)

    intersection = (output_flat * target_flat).sum(dim=2)
    union = output_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    mean_dice = dice.mean().item()

    return mean_dice

import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage

def compute_surface_distances(seg, gt):
    seg = seg.cpu().numpy()
    gt = gt.cpu().numpy()

    seg_boundary = seg - ndimage.binary_erosion(seg)
    gt_boundary = gt - ndimage.binary_erosion(gt)

    # Compute distance transforms
    seg_dist = ndimage.distance_transform_edt(1 - seg_boundary)
    gt_dist = ndimage.distance_transform_edt(1 - gt_boundary)
    return torch.tensor(seg_dist[seg_boundary == 1]), torch.tensor(gt_dist[gt_boundary == 1])

def hausdorff_distance(seg, gt):
    seg_to_gt, gt_to_seg = compute_surface_distances(seg, gt)
    if seg_to_gt.numel() == 0 or gt_to_seg.numel() == 0:
        return sum(seg.shape)
    return max(seg_to_gt.max().item(), gt_to_seg.max().item())

def assd(seg, gt):
    seg_to_gt, gt_to_seg = compute_surface_distances(seg, gt)
    if seg_to_gt.numel() == 0 or gt_to_seg.numel() == 0:
        return sum(seg.shape)
    return (seg_to_gt.mean().item() + gt_to_seg.mean().item()) / 2.0

def calculate_distances(output, label_batch):
    B, C, H, W = output.shape
    output = torch.randn(B, C, H, W)
    _, indices = torch.max(output, dim=1, keepdim=True)
    output = torch.zeros_like(output)
    output.scatter_(1, indices, 1)
    
    target = F.one_hot(label_batch.long(), num_classes=C).permute(0, 3, 1, 2).float()
    
    hd_scores = []
    assd_scores = []
    for i in range(B):
        for c in range(C):
            pred_binary = output[i][c].float()
            gt_binary = target[i][c].float()
            
            if gt_binary.sum() > 0:
                hd_scores.append(hausdorff_distance(pred_binary, gt_binary))
                assd_scores.append(assd(pred_binary, gt_binary))

    mean_hd = sum(hd_scores) / len(hd_scores)
    mean_assd = sum(assd_scores) / len(assd_scores)
    return mean_hd, mean_assd

