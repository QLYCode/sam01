import torch.nn as nn
from networks.mamba_components.down_vss_block import DownVSSBlock

class EVSS(nn.Module):
    def __init__(self):
        super(EVSS, self).__init__()
        self.evss1 = DownVSSBlock(hidden_dim=1,out_channels=16,down_srides = 1)
        self.evss2 = DownVSSBlock(hidden_dim=16,out_channels=32,down_srides = 2)
        self.evss3 = DownVSSBlock(hidden_dim=32,out_channels=64,down_srides = 2)
        self.evss4 = DownVSSBlock(hidden_dim=64,out_channels=128,down_srides = 2)
        self.evss5 = DownVSSBlock(hidden_dim=128,out_channels=256,down_srides = 2)
    def forward(self, x):
        x=x.permute(0, 2, 3, 1)
        x1 = self.evss1(x)
        x2 = self.evss2(x1)
        x3 = self.evss3(x2)
        x4 = self.evss4(x3)
        x5 = self.evss5(x4)
        return x5
