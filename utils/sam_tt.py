import math

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.ppg import PointPromptGenerator
import utils.helpers as helpers
from utils.logger import TensorBoardLogger



class SAM(nn.Module):
    def __init__(self, image_size, model_cfg, checkpoint, num_classes, embedding_size, n=40, not_n=10, device="cuda"):
        super(SAM, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.num_classes = num_classes
        self.reduce_embeddings = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(image_size, embedding_size, kernel_size=1, device=self.device),
        )
        self.resize = transforms.Compose(
            [
                transforms.Resize((256, 256)),
            ]
        )
        self.model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.n = n
        self.not_n = not_n
        self.logger = TensorBoardLogger().instance()
        self.ppg = PointPromptGenerator(n=self.n, not_n=self.not_n).to(self.device)

    def embed(self, images):
        with torch.no_grad():
            features = self.model.image_encoder.forward(images)["vision_features"]
            features = self.reduce_embeddings(features).squeeze()
            return features

    def _get_images(self, images):
        images = images.detach().cpu().numpy()
        image_list = [(images[b].transpose(1, 2, 0) * 255.0).astype(np.uint8) for b in range(images.shape[0])]
        return image_list
    
    
    def log_images(self, image, sam, points, labels):
        image = image.transpose(2, 0, 1)
        self.logger.log_image("sam/img", image, dataformats="HW")
        scale_factor = 1 / (self.num_classes - 1)
        vis_sam = torch.argmax(sam, dim=0) * scale_factor
        self.logger.log_image("sam/sam", vis_sam, dataformats="HW")
        _, H, W = image.shape
        point_map = torch.zeros((H, W), dtype=torch.float32) + 0.15
        for c in range(self.num_classes-1):
            for n in range(points.shape[1]):
                x = int(points[c, n, 0].item())
                y = int(points[c, n, 1].item())
                label = int(labels[c, n].item())
                if 0 <= x < W and 0 <= y < H:
                    point_map[y, x] = label * (c+1) * scale_factor
        self.logger.log_image("sam/points", point_map, dataformats="HW")

    def _process_sam_output(self, logits):
        prediction = [torch.Tensor(pred) for pred in logits]
        prediction = torch.stack(prediction).to(self.device)
        if self.num_classes > 2:
            prediction = prediction.squeeze(2)
        background = torch.zeros(prediction.shape[0], 1, prediction.shape[2], prediction.shape[3], device=self.device)
        prediction = torch.sigmoid(prediction)
        for b in range(prediction.shape[0]):
            pred = prediction[b]
            sum = pred.sum(dim=0, keepdim=True).float()
            background[b][0] = torch.clamp(1 - sum, 0, 1)
        prediction = torch.cat((background, prediction), dim=1)
        prediction = helpers.to_onehot(prediction)

    def forward(self, images, scribbles, outputs):
        with torch.no_grad():
            images = self._get_images(images)
            self.predictor.set_image_batch(images)
            points, labels = self.ppg(scribbles, outputs)
            _, _, logits = self.predictor.predict_batch(points, labels, box_batch=None, multimask_output=False)
            prediction = self._process_sam_output(logits)
            if self.logger.step % 50 == 0:
                i = 0
                self.log_images(images[i], prediction[i], points[i], labels[i])
            return prediction
