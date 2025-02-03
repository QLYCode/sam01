# from networks.CycleMix import CycleMix2D
# from networks.ProgressMix import ProgressMix
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, DynamicMix, UNet_CCT_3H
# from networks.EfficientUMamba import EfficientUMamba
import os
import random
import argparse
import torch.backends.cudnn as cudnn
import numpy as np
import torch
# CNN
from networks.unet import UNet, UNet_DS, UNet_CCT_3H
from networks.bisenet import BiSeNet
from networks.ghostnet import ghostnet
from networks.mobilenet import MobileNetV2
from networks.alignseg import AlignSeg
from networks.ccnet import CCNet_Model
from networks.ecanet import ECA_MobileNetV2
# Mamba
from networks.msvmunet import build_msvmunet_model
from networks.segmamba import SegMamba
from networks.SwinUMamba import SwinUMamba
from networks.em_net_model import EMNet
from networks.UMambaBot_2d import UMambaBot
from networks.lkmunet import LKMUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "dynamicmix":
        net = DynamicMix(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "ccnet":
        net = CCNet_Model(num_classes=class_num).cuda()
    elif net_type == "alignseg":
        net = AlignSeg(num_classes=class_num).cuda()
    elif net_type == "bisenet":
        net = BiSeNet(in_channel=in_chns, nclass=class_num, backbone='resnet18').to(device)
    elif net_type == "mobilenet":
        net = MobileNetV2(input_channel=in_chns, n_class=class_num).cuda()
    elif net_type == "ecanet":
        net = ECA_MobileNetV2(n_class=4, width_mult=1).to(device)
    elif net_type == "msvmunet":
        net = build_msvmunet_model(in_channels=3, num_classes=class_num).cuda()
    elif net_type == "segmamba":
        net = SegMamba(in_chns, class_num).cuda()
    elif net_type == "swinumamba":
        net = SwinUMamba(in_chns, class_num).cuda()
    elif net_type == "emnet":
        net = EMNet(in_chns, class_num).cuda()
    elif net_type == "umamba":
        net = UMambaBot(input_channels=in_chns,
                        n_stages=4,  # 包含stem的4个编码阶段
                        features_per_stage=[32, 64, 128, 256],  # 各阶段特征通道数
                        conv_op=nn.Conv2d,
                        kernel_sizes=[[3, 3], [3, 3], [3, 3], [3, 3]],  # 每个阶段的卷积核尺寸
                        strides=[1, 2, 2, 2],  # 下采样策略（第一次保持分辨率）
                        n_conv_per_stage=[2, 2, 2, 2],  # 编码器各阶段卷积层数
                        num_classes=class_num,
                        n_conv_per_stage_decoder=[2, 2, 2],  # 解码器各阶段卷积层数（比编码器少1阶）
                        conv_bias=False,
                        norm_op=nn.BatchNorm2d,
                        norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
                        nonlin=nn.LeakyReLU,
                        nonlin_kwargs={"inplace": True},
                        deep_supervision=False
                        ).cuda()
    elif net_type == "lkmunet":
        net = LKMUNet(in_channels=in_chns, out_channels=class_num, kernel_sizes=[21, 15, 9, 3]).cuda()
    # elif net_type == "cyclemix":
    #     net = CycleMix2D(feature_scale=4, n_classes=class_num, in_channels=in_chns, is_batchnorm=True)
    # elif net_type == "progressmix":
    #     net = ProgressMix(in_chns=in_chns, class_num=class_num).cuda()
    # elif net_type == "efficientumamba":
    #     net = EfficientUMamba(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
