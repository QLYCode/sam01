import random

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

def show_image(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).show()

def save_image(arr, path):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)

def show_rgb(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).show()

def to_onehot(tensor, num_classes=4):
    winning_classes = torch.argmax(tensor, dim=1)
    one_hot_tensor = F.one_hot(winning_classes, num_classes=num_classes).permute(0, 3, 1, 2).float() # (B, C, W, H)
    return one_hot_tensor