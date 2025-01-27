from networks.CycleMix import CycleMix2D
from networks.ProgressMix import ProgressMix
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, DynamicMix, UNet_CCT_3H
from networks.EfficientUMamba import EfficientUMamba


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
    elif net_type == "cyclemix":
        net = CycleMix2D(feature_scale=4, n_classes=class_num, in_channels=in_chns, is_batchnorm=True)
    elif net_type == "progressmix":
        net = ProgressMix(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficientumamba":
        net = EfficientUMamba(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
