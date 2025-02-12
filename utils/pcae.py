import torch
import torch.nn as nn

import utils.helpers as helpers

class PointCloudAccuracyEstimator(nn.Module):
    def __init__(self, k=100, threshold=0.75):
        super(PointCloudAccuracyEstimator, self).__init__()
        self.k = k
        self.threshold = threshold
        self.eps = 1e-6

    def _normalize_across_dim(self, tensor, dim):
        x = tensor.sum(dim=dim)
        norm = x / (x.sum(dim=2, keepdim=True) + self.eps)
        return norm


    def _get_axis_distribution(self, tensor1, tensor2, dim):
        tensor1 = self._normalize_across_dim(tensor1, dim=dim)
        tensor2 = self._normalize_across_dim(tensor2, dim=dim)
        return torch.sum(torch.min(tensor1, tensor2), dim=2)

    def get_agreement(self, tensor1, tensor2):
        sim_H = self._get_axis_distribution(tensor1, tensor2, dim=3)
        sim_W = self._get_axis_distribution(tensor1, tensor2, dim=2)
        agreement_H = torch.sigmoid(self.k * (sim_H - self.threshold))
        agreement_W = torch.sigmoid(self.k * (sim_W - self.threshold))
        agreement = (agreement_H + agreement_W) / 2
        return agreement

    def forward(self, scribbles, model_out):
        with torch.no_grad():
            model_out = helpers.to_onehot(model_out)
            return self.get_agreement(scribbles, model_out)
 

