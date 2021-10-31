# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 22:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import time
import numpy as np
import torchvision
import torch
import torch.nn as nn

# from models.backbone.resnet import ResNet
from models.backbone.resnet_PR import ResNet,Bottleneck
from models.backbone.csp_darknet import CSPDarknet
from models.backbone.darknet53_C5 import *
from models.neck.Dilated_Encoder import DilatedEncoder
from models.head.YOLOF_Decoder import YOLOF_Head
from models.losses.YOLOF_LOSS import YOLOFLoss
from models.post_process import yolox_post_process
from models.ops import fuse_model
from data.data_augment import preproc
from utils.model_utils import load_model
from utils.util import sync_time




def get_model(opt):

    num_residual_blocks = 4
    block_dilations = [2, 4, 6, 8]

    if opt.backbone == "Darknet53":
        backbone = Darknet53_C5(out_indices=(5))
        in_channel = [1024]
    elif opt.backbone == "CSPDarknet":
        backbone = CSPDarknet(out_indices=(5))
        in_channel = [1024]
        num_residual_blocks = 8
        block_dilations = [1,2,3,4,5,6,7,8]
    elif opt.backbone == "Res101" or opt.backbone == "res101":
        backbone = ResNet(depth=101, out_indices=[3])
        in_channel = [2048]
    elif opt.backbone == "Res50" or opt.backbone == "res50":
        # backbone = ResNet(depth=50, out_indices=[3])            # 以前的版本
        # num_residual_blocks = 6
        # block_dilations = [1, 2, 5, 1, 2, 5]

        in_channel = [2048]
        backbone = ResNet(Bottleneck, [3,4,6,3])
        if opt.pretrain:        # 使用预训练模型
            # 加载预训练模型
            pre_model = torchvision.models.resnet50(pretrained=True)

            # 读取参数
            pretrained_dict = pre_model.state_dict()
            model_dict = backbone.state_dict()

            # 将pretrained_dict里不属于model_dict的键剔除掉
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 更新现有的model_dict
            model_dict.update(pretrained_dict)

            # 加载我们真正需要的state_dict
            backbone.load_state_dict(model_dict)


    else:
        print("请输入正确的backbone")
        exit()

    # define neck
    neck = DilatedEncoder(in_channels = in_channel, num_residual_blocks = num_residual_blocks, block_dilations = block_dilations)
    # define head
    head = YOLOF_Head(num_classes=opt.num_classes, depthwise=opt.depth_wise)
    # define loss
    loss = YOLOFLoss(opt.label_name,  strides=[32], in_channels=in_channel)

    # define network
    model = YOLOF(opt, backbone=backbone, neck=neck, head=head, loss=loss)

    return model



class YOLOF(nn.Module):
    def __init__(self, opt, backbone, neck, head, loss):
        super(YOLOF, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        if not opt.pretrain:
            self.backbone.init_weights()
        self.neck._init_weight()
        self.head.init_weights()

    def forward(self, inputs, targets=None, show_time=False):
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            if show_time:
                s1 = sync_time(inputs)

            body_feats = self.backbone(inputs)      # [batch, 1024, 20, 20]
            neck_feats = self.neck(body_feats)      # [batch, 512, 20, 20]
            yolo_outputs = self.head(neck_feats)    # 是一个list，里面存了一个tensor，shape为[batch, class+5, 20, 20], class+5中，前四个是BBOX，然后是obj，最后是class



            # print('yolo_outputs:', [[i.shape, i.dtype, i.device] for i in yolo_outputs])  # float16 when use_amp=True

            if show_time:
                s2 = sync_time(inputs)
                print("[inference] batch={} time: {}s".format("x".join([str(i) for i in inputs.shape]), s2 - s1))

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)
                # for k, v in loss.items():
                #     print(k, v, v.dtype, v.device)  # always float32

        if targets is not None:
            return yolo_outputs, loss
        else:
            return yolo_outputs


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        print("Loading model {}".format(self.opt.load_model))
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()
        if "fuse" in self.opt and self.opt.fuse:
            print("==>> fuse model's conv and bn...")
            self.model = fuse_model(self.model)

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:    #如果图片的维度为3，说明是单张图片
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()

            img_ratios, img_shape = [], []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_shape.append(image.shape[:2])

                # 预处理，调整成需要的尺寸，补零
                img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                s2 = time.time()
                print("[pre_process] time {}".format(s2 - s1))

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            yolo_outputs = self.model(inp_imgs, show_time=show_time)    # 得到输出结果


            if show_time:
                s3 = sync_time(inp_imgs)

            # 后处理
            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
            if show_time:
                s4 = sync_time(inp_imgs)
                print("[post_process] time {}".format(s4 - s3))
        if batch_img:
            return predicts
        else:
            return predicts[0]





if __name__ == "__main__":
    from thop import profile
    from config import opt

    feats0 = torch.zeros([1,3, 640, 640])
    feats1 = torch.zeros([1,8, 640, 640])
    feats2 = torch.zeros([1,1024,40,40])
    feats3 = torch.zeros([1, 256, 40, 40])



    backbone = CSPDarknet(out_indices=(5))
    # define neck
    neck = DilatedEncoder(in_channels = [1024], num_residual_blocks = 4, block_dilations = [2,4,6,8])
    # define head
    head = YOLOF_Head(num_classes=20,  width=1, in_channels=256,depthwise=False)


    model = get_model(opt)

    data = model(feats0)

    print(data[0].shape)

    backbone.init_weights()
    backbone.eval()
    total_ops, total_params = profile(model, (feats0,))

    print("total_ops {:.2f}G, total_params {:.2f}M".format(total_ops / 1e9, total_params / 1e6))

