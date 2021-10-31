from typing import List

from fvcore.nn import c2_xavier_fill
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec,BatchNorm2d, NaiveSyncBatchNorm,FrozenBatchNorm2d
from detectron2.utils import env
from functools import partial


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        kwargs: Additional parameters in normalization layers,
            such as, eps, momentum

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (
                1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)


def get_activation(activation):
    """
    Only support `ReLU` and `LeakyReLU` now.

    Args:
        activation (str or callable):

    Returns:
        nn.Module: the activation layer
    """

    act = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
    }[activation]
    if activation == "LeakyReLU":
        act = partial(act, negative_slope=0.1)
    return act(inplace=True)

class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    # def __init__(self, cfg, input_shape: List[ShapeSpec]):
    def __init__(self,in_channels = 1024, num_residual_blocks = 4, block_dilations = [2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        self.backbone_level = 'res5'    # res5
        self.in_channels = in_channels[0]      # 2048
        self.encoder_channels = 512    # 512
        self.block_mid_channels = 128   # 128
        self.num_residual_blocks = num_residual_blocks# 4
        self.block_dilations = block_dilations       # [2, 4, 6, 8]
        self.norm_type = 'BN'                # BN
        self.act_type = "ReLU"             # Relu

        # fmt: on
        # assert input_shape[self.backbone_level].channels == self.in_channels
        # assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,  # 1024
                                      self.encoder_channels,  # 512
                                      kernel_size=1)
        self.lateral_norm = get_norm(self.norm_type, self.encoder_channels)  # BN 512
        self.fpn_conv = nn.Conv2d(self.encoder_channels,  # 512
                                  self.encoder_channels,  # 512
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = get_norm(self.norm_type, self.encoder_channels)
        encoder_blocks = []

        # 连续4个空洞卷积残差块
        for i in range(self.num_residual_blocks):  # num_residual_blocks = 4
            dilation = self.block_dilations[i]  # [2,4,6,8]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,      # 512
                    self.block_mid_channels,    # 128
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # feature.shape = (batch,1024,20,20)
        out = self.lateral_norm(self.lateral_conv(feature))     # 用1*1的卷积把1024调整成512，然后BN
        out = self.fpn_norm(self.fpn_conv(out))     # 用512个3*3的，然后BN
        return self.dilated_encoder_blocks(out)     # 通过4个空洞卷积残差块


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 norm_type: str = 'BN',
                 act_type: str = 'ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


if __name__ == "__main__":
    from thop import profile

    m = DilatedEncoder()

    # m.init_weights()
    m.eval()
    inputs = torch.rand(512, 1024, 1, 1)
    total_ops, total_params = profile(m, (inputs,))
    print("total_ops {}G, total_params {}M".format(total_ops/1e9, total_params/1e6))
