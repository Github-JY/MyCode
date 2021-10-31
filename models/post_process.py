# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 23:06
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import torch
import torch.nn.functional as F
import torchvision

# 后处理过程，把网络预测的结果整理成 [label, conf, bbox]
def yolox_post_process(outputs, down_strides, num_classes, conf_thre, nms_thre, label_name, img_ratios, img_shape):
    hw = [i.shape[-2:] for i in outputs]
    grids, strides = [], []

    # zip函数将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    for (hsize, wsize), stride in zip(hw, down_strides):    # down_strides == [8, 16, 32]
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])     # https://blog.csdn.net/weixin_39504171/article/details/106356977
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)  # stack 在第2的位置插入维度，view构造shape为[1,-1,2]的矩阵
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride)) # torch.full((*shape, 1)是输出shape形状的全1数组

    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # bs, all_anchor, 85(+128)
    grids = torch.cat(grids, dim=1).type(outputs.dtype).to(outputs.device)      # 将list中的多个tensor，整合成一个大tensor
    strides = torch.cat(strides, dim=1).type(outputs.dtype).to(outputs.device)  # 将list中的多个tensor，整合成一个大tensor

    # print('outputs.shape= ',outputs.shape)      # 测试单张图片shape=  torch.Size([1, 8400, 85]),85是 x,y,w,h,obj,80个class
    # print('grids.shape= ', grids.shape)         # 测试单张图片shape=  torch.Size([1, 8400, 2])
    # print('strides.shape= ', strides.shape)     # 测试单张图片shape=  torch.Size([1, 8400, 1])

    # x, y
    outputs[..., 0:2] = (outputs[..., 0:2] + grids) * strides
    # w, h
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # obj
    outputs[..., 4:5] = torch.sigmoid(outputs[..., 4:5])
    # 80 class
    outputs[..., 5:5 + num_classes] = torch.sigmoid(outputs[..., 5:5 + num_classes])
    # reid
    reid_dim = outputs.shape[2] - num_classes - 5
    if reid_dim > 0:
        outputs[..., 5 + num_classes:] = F.normalize(outputs[..., 5 + num_classes:], dim=2)

    box_corner = outputs.new(outputs.shape) # 创建个shape相同的tensor
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1   左上角坐标x1 等于 中心x坐标- w/2
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2  # x2   右下角坐标
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2  # y2
    outputs[:, :, :4] = box_corner[:, :, :4]


    output = [[] for _ in range(len(outputs))]   # 存一个batch的数量
    for i, image_pred in enumerate(outputs):
        # Get score and class with highest confidence
        # image_pred.shape = torch.Size([8400, 85])

        # class_conf对应最大值的置信度，class_pred对应最大值的索引
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)  # dim=1是取行最大值


        # 默认conf_thre = 0.3, image_pred[:, 4].shape = torch.Size([8400]), class_conf.shape = torch.Size([8400, 1])
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()    # squeeze()压缩为1的维度
        # conf_mask = tensor([False, False, False, ..., False, False, False], device='cuda:0'),  shape = torch.Size([8400])


        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        if reid_dim > 0:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5 + num_classes:]),
                                   1)
        else:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)


        detections = detections[conf_mask]  # 提取出mask为True对应的数据

        if not detections.size(0):  # size(0)返回的是数字，既第0维度有多少行数据
            continue

        # 对应的是坐标，bbox分数*分类分数，类别对应的索引，nms阈值, 返回的是结果中对应的索引，nms_out_index= tensor([24, 17, 10,  3]
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                    detections[:, 6], nms_thre)

        detections = detections[nms_out_index]

        # img_ratios是图片的放缩比例，在前处理部分计算得到
        detections[:, :4] = detections[:, :4] / img_ratios[i]

        img_h, img_w = img_shape[i]     # 得到原图的尺寸
        for det in detections:
            x1, y1, x2, y2, obj_conf, class_conf, class_pred = det[0:7]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            conf = float(obj_conf * class_conf)
            label = label_name[int(class_pred)]

            # clip bbox，计算得到在预处理后的图片上对应的bbox坐标
            bbox[0] = max(0, min(img_w, bbox[0]))
            bbox[1] = max(0, min(img_h, bbox[1]))
            bbox[2] = max(0, min(img_w, bbox[2]))
            bbox[3] = max(0, min(img_h, bbox[3]))

            if reid_dim > 0:
                reid_feat = det[7:].cpu().numpy().tolist()
                output[i].append([label, conf, bbox, reid_feat])
            else:
                output[i].append([label, conf, bbox])
    return output



