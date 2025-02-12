import torch
import torch.nn as nn
import numpy as np

from utils.pcae import PointCloudAccuracyEstimator
import utils.helpers as helpers

class PointPromptGenerator(nn.Module):
    def __init__(self, n, not_n, ignore_index=0, **kwargs):
        self.n = n
        self.not_n = not_n
        self.ignore_index = ignore_index
        self.pcu = PointCloudAccuracyEstimator()

    def log_pcu(self, pcu):
        self.logger.log_scalar("pcu/mean", pcu.mean())
        class_means = pcu.mean(dim=0)
        for i, val in enumerate(class_means):
            self.logger.log_scalar(f"pcu/class_{i}_mean", val)

    def forward_pcu(self, labels, out_sig):
        pcu = self.pcu(labels, out_sig)
        self.log_pcu(pcu)
        return pcu
    
    def _get_points(coords, n):
        point_idxs = np.random.choice(len(coords), n, replace=False)
        return coords[point_idxs]
        
    def _reduce_mask(self, binary_mask, background, k, bg):
        coords = np.column_stack(np.where(binary_mask))
        non_coords = np.column_stack(np.where(background))
        num_points = min(len(coords), k)
        num_negative = k - num_points + bg
        my_points = self._get_points(coords, num_points)
        my_negatives = self._get_points(non_coords, num_negative)
        coords = np.concatenate((my_points, my_negatives), axis=0)
        labels = np.concatenate((np.ones(num_points), np.zeros(num_negative)), axis=0)
        return coords, labels

    def forward(self, scribbles, outputs, pcu):
        B, C, _, _ = outputs.shape
        scribbles = scribbles.detach().cpu().numpy()
        outputs = helpers.to_onehot(outputs).detach().cpu().numpy()
        points = []
        labels = []

        pcu = self.forward_pcu(labels, outputs)
        
        for b in range(B):
            img_prompt = np.zeros((C - 1, self.n + self.not_n, 2))
            img_labels = np.zeros((C - 1, self.n + self.not_n))
            
            for c in range(0, C):
                if c == self.ignore_index:
                    continue
                num_pred = int(round(pcu[b][c].item() * self.n, 0))
                num_scribble = self.n - num_pred

                num_not_pred = int(round((1 - pcu[b][c].item()) * self.not_n, 0))
                num_not_scribble = self.not_n - num_not_pred
                
                points_scrib, labels_scrib = self._reduce_mask(scribbles[b][c], scribbles[b][0], num_scribble, num_not_pred)
                points_out, labels_out = self._reduce_mask(outputs[b][c], outputs[b][0], num_pred, num_not_scribble)
                
                prompt = np.concatenate((points_scrib, points_out), axis=0)
                prompt_labels = np.concatenate((labels_scrib, labels_out), axis=0)
                prompt = prompt[:, [1, 0]]
                
                img_prompt[c - 1] = prompt
                img_labels[c - 1] = prompt_labels
            points.append(img_prompt.astype(np.int64))
            labels.append(img_labels.astype(np.int64))
        return points, labels