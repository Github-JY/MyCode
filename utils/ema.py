# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 19:42
# @Author  : MingZhang
# @Email   : zhangming210426@credithc.com

import math
from copy import deepcopy
import torch
import torch.nn as nn

from .data_parallel import _DataParallel


def is_parallel(model):
    # 判断是不是有多个GPU
    """check if model is in parallel mode."""

    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
        _DataParallel
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9998, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        # 先把模型复制一份
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)     # ema里的数据不保留梯度

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            decay_value = self.decay(self.updates)

            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    # EMA的公式
                    v *= decay_value
                    v += (1.0 - decay_value) * state_dict[k].detach()
