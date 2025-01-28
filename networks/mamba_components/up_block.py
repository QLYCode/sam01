import torch
import torch.nn as nn

from networks.mamba_components.conv_block import ConvBlock

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
   
   
class UpBlock0(nn.Module):
    """Upsampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p):
        super(UpBlock0, self).__init__()
        # 使用 1x1 卷积层来调整通道数
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        # 使用卷积层而不是上采样层
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        # 不进行上采样，直接拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
