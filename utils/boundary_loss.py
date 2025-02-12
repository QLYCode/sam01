
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=0, device="cuda"):
        super(BoundaryLoss, self).__init__()
        self.ignore_index=0
        self.device=device

    def compute_distance_map(self, edge_map):
        dist_maps = np.zeros_like(edge_map)
        for i in range(edge_map.shape[0]):
            for j in range(edge_map.shape[1]):
                if j == self.ignore_index:
                    continue
                dist_maps[i, j] = distance_transform_edt(1 - edge_map[i, j])
        return torch.tensor(dist_maps, dtype=torch.float32).to(self.device)
    
    def boundary_loss(self, predictions, distance_map):
        pred_probs = F.softmax(predictions, dim=1)
        boundary_loss = (distance_map * pred_probs)
        return boundary_loss

    def forward(self, preds, edge_map):
        distance_map = self.compute_distance_map(edge_map.detach().cpu().numpy())
        mean_distance = distance_map.mean()
        normalization_factor = mean_distance
        if normalization_factor.item() > 0:
            distance_map = distance_map / normalization_factor.item()
        else:
            distance_map = torch.zeros_like(distance_map)
        return self.boundary_loss(preds, distance_map)
