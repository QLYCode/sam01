import torch.nn as nn
from networks.mamba_components.encoder import Encoder
from networks.mamba_components.decoder import Decoder
from networks.mamba_components.evss import EVSS


class EfficientUMamba(nn.Module):
    def __init__(self, in_chns, class_num):
        super(EfficientUMamba, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.EVSS = EVSS()
        self.combiner = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        feature = self.encoder(x)
        evss_out = self.EVSS(x)
        combined_feature  = feature[-1] + evss_out.permute(0, 3, 1, 2)
        combined_feature = self.combiner(combined_feature)

        feature.append(combined_feature)

        output = self.decoder(feature)
        return output

