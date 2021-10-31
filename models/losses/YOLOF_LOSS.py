# -*- coding: utf-8 -*-
# @Time    : 21-7-20 20:01
# @Author  : MingZhang
# @Email   : zm19921120@126.com


import numpy as np
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOFLoss(nn.Module):
    def __init__(self, label_name, reid_dim=0, id_nums=None, strides=[8, 16, 32], in_channels=[256, 512, 1024]):
        super().__init__()

        self.n_anchors = 1
        self.label_name = label_name
        self.num_classes = len(self.label_name)
        self.strides = strides
        self.reid_dim = reid_dim

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")   # 适用于多标签分类任务,先对输出向量里的每个元素使用sigmoid函数, 然后使用BCELoss函数
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(in_channels)

        # 目标追踪
        if self.reid_dim > 0:
            assert id_nums is not None, "opt.tracking_id_nums shouldn't be None when reid_dim > 0"
            assert len(id_nums) == self.num_classes, "num_classes={}, which is different from id_nums's length {}" \
                                                     "".format(self.num_classes, len(id_nums))
            # scale_trainable = True
            # self.s_det = nn.Parameter(-1.85 * torch.ones(1), requires_grad=scale_trainable)
            # self.s_id = nn.Parameter(-1.05 * torch.ones(1), requires_grad=scale_trainable)

            self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.classifiers = nn.ModuleList()
            self.emb_scales = []
            for idx, (label, id_num) in enumerate(zip(self.label_name, id_nums)):
                print("{}, tracking label name: '{}', tracking_id number: {}, feat dim: {}".format(idx, label, id_num,
                                                                                                   self.reid_dim))
                self.emb_scales.append(np.math.sqrt(2) * np.math.log(id_num - 1))
                self.classifiers.append(nn.Linear(self.reid_dim, id_num))

    def forward(self, preds, targets, imgs=None):
        # preds是一个list,格式为（1个尺寸的特征图, batch, 类别数+5，高，宽),  其中前4个为BBox，obj，加类别数
        # targets.shape =  torch.Size([8, 120, 5]) ，对应一个batch的数量8，一个图最多的物体数120， class和BBox信息5
        # x_shifts, y_shifts 是一个list,都存了1个tensor ，shape为[1, 400]
        # origin_preds 和 L1 损失有关 ，L1损失默认不开启，存了3个Tensor，shape为 [8, 6400/1600/400, 4]
        outputs, origin_preds, x_shifts, y_shifts, expanded_strides = [], [], [], [], []



        # 计算不同尺寸特征图下的偏移量
        for k, (stride, p) in enumerate(zip(self.strides, preds)):
            # 步长为32时， p.shape分别为 ,[8, 25, 20, 20]
            pred, grid = self.get_output_and_grid(p, k, stride, p.dtype)

            outputs.append(pred)

            # x轴和y轴的偏移量
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])

            # expanded_strides是一个lsit，存了三个Tensor ，shape分别为[1, 6400]，[1, 1600]，[1, 400]， 内部存的分别是 8,16,32
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride).type_as(p))


            if self.use_l1:
                # p.shape分别为[8, 25, 80, 80], [8, 25, 40, 40], [8, 25, 20, 20]
                reg_output = p[:, :4, :, :]  # 取BBOX的信息

                # 修改shape
                batch_size, _, hsize, wsize = reg_output.shape
                reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)   # ( 8, 1, 4, 80/40/20, 80/40/20 )
                reg_output = (reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)) # [8, 6400/1600/400, 4]

                # 存入origin_preds
                origin_preds.append(reg_output.clone())


        # outputs是一个list，len(outputs) = 3 , torch.cat后变成了一个Tensor，torch.Size([8, 400, 25]
        outputs = torch.cat(outputs, 1)

        # 计算得到各种loss
        total_loss, iou_loss, conf_loss, cls_loss, l1_loss, reid_loss, num_fg = self.get_losses(imgs, x_shifts,
                                                                                                y_shifts,
                                                                                                expanded_strides,
                                                                                                targets, outputs,
                                                                                                origin_preds,
                                                                                                dtype=preds[0].dtype)

        losses = {"loss": total_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "iou_loss": iou_loss}
        if self.use_l1:
            losses.update({"l1_loss": l1_loss})
        if self.reid_dim > 0:
            losses.update({"reid_loss": reid_loss})
        losses.update({"num_fg": num_fg})
        return losses

    def get_output_and_grid(self, p, k, stride, dtype):
        # p是predict, p.shape =  torch.Size([8, 25, 20, 20]),k应该是对应特征图的层级，表示当前是哪个特征图
        p = p.clone()

        # grid.shape =  torch.Size([1, 1, 80/40/20, 80/40/20, 2]),最后面的2存的是 x轴和y轴的偏移量
        grid = self.grids[k]


        # 步长分别为8,16,32时， p.shape分别为 [8, 25, 80, 80],[8, 25, 40, 40],[8, 25, 20, 20]
        batch_size, n_ch, hsize, wsize = p.shape

        # 如果形状不匹配或是运行设备不一致，则修改到符合要求，grid.shape = (1, 1, hsize, wsize, 2)
        if grid.shape[2:4] != p.shape[2:4] or grid.device != p.device:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])     #  torch.meshgrid（）的功能是生成网格，可以用于生成坐标
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(p.device)
            self.grids[k] = grid

        pred = p.view(batch_size, self.n_anchors, n_ch, hsize, wsize)   # [8, 1, 25, 80/40/20, 80/40/20]
        pred = (pred.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1))    # [8, 6400/1600/400, 25]


        grid = grid.view(1, -1, 2)  # [1, 6400/1600/400, 2],在输出的三种大小特征图上，分别有 6400.1600.400个锚框


        pred[..., :2] = (pred[..., :2] + grid) * stride
        pred[..., 2:4] = torch.exp(pred[..., 2:4]) * stride



        return pred, grid

    # 计算得到各种的loss
    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, targets, outputs, origin_preds, dtype):
        # outputs.shape = [8, 锚框数（8400）, 25]

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:self.num_classes + 5]  # [batch, n_anchors_all, n_cls]

        if self.reid_dim > 0:   # 是否开启追踪
            reid_preds = outputs[:, :, self.num_classes + 5:]  # [batch, h*w, 128]

        assert targets.shape[2] == 6 if self.reid_dim > 0 else 5

        # 存识别到物体的数量 格式为tensor([10,  3,  6,  8, 25, 16, 14,  3]
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects


        # total_num_anchors =  8400
        total_num_anchors = outputs.shape[1]

        # x_shifts, y_shifts 是一个list,都存了3个tensor ，shape分别为 [1, 6400] ，[1, 1600]，[1, 400]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]

        # expanded_strides是一个lsit，存了三个Tenmsor ，shape分别为[1, 6400]，[1, 1600]，[1, 400]， 内部存的分别是 8,16,32
        expanded_strides = torch.cat(expanded_strides, 1)   # 转换到shape = [1,8400]


        if self.use_l1:
            # [8, 6400/1600/400, 4] ——> [8,8400,4]
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        reid_targets = []
        fg_masks = []

        num_fg = 0.0    # 计算一个Batc中，所有正样本数量的总和
        num_gts = 0.0   # 计算一个Batc中，所有图片中物体的总和

        # 一张一张图片算
        for batch_idx in range(outputs.shape[0]):
            # 当前图片的物体数量
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt

            if num_gt == 0:     # 未识别到物体
                # 所有分类都用 0
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                reid_target = outputs.new_zeros((0, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:               # 识别到了物体
                # 得到真实的分类和位置信息
                gt_classes = targets[batch_idx, :num_gt, 0]             # gt_classes.shape = [单张图片的物体数量]
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]  # gt_bboxes_per_image.shape = [单张图片的物体数量,4]

                if self.reid_dim > 0:
                    gt_tracking_id = targets[batch_idx, :num_gt, 5]

                # 得到单张图片的预测BBOX
                bboxes_preds_per_image = bbox_preds[batch_idx, :, :]

                # --------------------调用get_assignments进行正负样本分配--------------------
                try:    # 尝试用GPU算
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs,
                    )
                except RuntimeError:    #  换成CPU计算
                    print(traceback.format_exc())
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                        # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                        cls_preds, bbox_preds, obj_preds, targets, imgs, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img


                # 根据分配好的正样本来计算数据
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes) * pred_ious_this_matching.unsqueeze(-1)    # unsqueeze(-1)增加一个维度
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                # cls_target.shape = [正样本数,类别总数]
                # obj_target.shape = [锚框总数,1]
                # reg_target.shape = [正样本数,4]


                if self.reid_dim > 0:
                    reid_target = gt_tracking_id[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

            if self.use_l1:
                l1_targets.append(l1_target)
            if self.reid_dim > 0:
                reid_targets.append(reid_target)

        # 将多个目标拼接在一起(包含了多个正样本)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)

        # fg_masks存布尔值，表示对应的预测框中心在GT中，可作为候选框  shape = [batch * 所有的锚框数, ]
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        if self.reid_dim > 0:
            reid_targets = torch.cat(reid_targets, 0).type(torch.int64)

        # ----------------------------------loss计算----------------------------------------

        # 存一个batch正样本数量的总和
        num_fg = max(num_fg, 1)

        # 将预测信息中所有包含正样本的锚框信息与target计算，得到的总和处理正样本总数
        # bbox_preds.shape=  [batch, n_anchors_all, 4]， bbox_preds.view(-1, 4)[fg_masks].shape = [num_fg, 4]
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg    # 针对筛选出来的正样本计算
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg        # 针对的是所有8400个anchor计算
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg   # 针对筛选出来的正样本计算
        loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg if self.use_l1 else 0.  # 针对筛选出来的正样本计算

        reid_loss = 0.
        if self.reid_dim > 0:   # 目标追踪
            reid_feat = reid_preds.view(-1, self.reid_dim)[fg_masks]
            cls_label_targets = cls_targets.max(1)[1]
            for cls in range(self.num_classes):
                inds = torch.where(cls == cls_label_targets)
                if inds[0].shape[0] == 0:
                    continue
                this_cls_tracking_id = reid_targets[inds]
                this_cls_reid_feat = self.emb_scales[cls] * F.normalize(reid_feat[inds])

                reid_output = self.classifiers[cls](this_cls_reid_feat)
                reid_loss += self.reid_loss(reid_output, this_cls_tracking_id)

        # loss合并
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + reid_loss
        fg_r = torch.tensor(num_fg / max(num_gts, 1), device=outputs.device, dtype=dtype)   # 正样本总数 / 物体的总数
        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, reid_loss, fg_r

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()

    # 功能:返回经过simOTA计算后，具体哪些anchor是正样本，哪些是负样本
    def get_assignments(
            self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
            bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
            cls_preds, bbox_preds, obj_preds, targets, imgs, mode="gpu",
    ):
        # num_gt为单张图片的物体数量
        if mode == "cpu":
            # 全换成CPU类型
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # ----------调用get_in_boxes_info确定候选区域,初步筛选出一部分候选的anchor--------------------
        # ----------确定论文中所说的a fixed center region，即缩小anchor考察范围。--------------------
        # -------------------------------------------------------------------------------------
        # fg_mask 存布尔值，表示两个条件至少满足一个，可作为候选框，shape:[num_anchors]， 例如 [8400]
        # is_in_boxes_and_center shape:[num_gt, num_in_anchors],例如 [5, 7900]，表示既在GT，又在扩大框
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------

        # 从所有预测中选出可作为候选的anchor信息  ，  shape：  [anchor总数, 4] -> [符合要求的anchor数量, 4]
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]      # [8, 8400,类别总数] -> [符合的anchor数，类别数]
        obj_preds_ = obj_preds[batch_idx][fg_mask]      # [8, 8400, 1] -> [符合的anchor数，1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]   # num_in_boxes_anchor记录符合要求的anchor数量

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # -------------------------------------计算cost-----------------------------------------
        # -------------------------------------------------------------------------------------
        # 计算所有的gt和bboxes的iou，iou用于dynamic_k的确定
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)     # shape = [目标数，符合的anchor数]

        # gt_classes存的是类别对应的索引，如[3., 2., 6., 0.]
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor,1))
            # gt_cls_per_image.shape = [物体数，符合的anchor数，类别数]

        # 计算匹配IOU损失， IOU越大，loss越小
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)


        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()


        with torch.cuda.amp.autocast(enabled=False):
            # cost中Lcls和Lreg计算,将类别的条件概率和目标的先验概率做乘积，得到目标的类别分数
            cls_preds_ = (
                        cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid() * obj_preds_.unsqueeze(0).repeat(
                    num_gt, 1, 1).sigmoid())
            # cls_preds_.shape = [num_gt,符合的anchor数，类别数]

            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt(), gt_cls_per_image, reduction="none").sum(-1)  # 二元交叉熵损失函数
            # 前面部分shape为 [num_gt,符合的anchor数，类别数]，用了sum(-1)后变成了[num_gt,符合的anchor数]
        del cls_preds_

        cost = (pair_wise_cls_loss              # Lcls
                + 3.0 * pair_wise_ious_loss     # λ*Lreg,实际代码中把λ设置为了3
                + 100000.0 * (~is_in_boxes_and_center)      # ~是取反，把不在考虑范围内的anchor置为很大的数值
                )           # cost.shape = [目标数，符合的anchor数]


        # -----------------给每个gt分配正样本，同时确定每个gt要分配几个正样本--------------------------
        # -------------------------------------------------------------------------------------
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()


        # gt_matched_classes存每个框对应物体的类别编号 , shape = [所有物体动态K的总和]
        # fg_mask存布尔值，表示对应的anchor中心在GT中，可作为候选框 , shape = [所有anchor的总数]
        # pred_ious_this_matching存每个框对应的IOU, shape = [所有物体动态K的总和]
        # matched_gt_inds存的索引是指本张图片上对应的物体编号 , shape = [所有物体动态K的总和]
        # num_fg = 所有物体动态K的总和，是一个int
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    # 功能:确定论文中所说的a fixed center region，即缩小anchor考察范围。返回对应锚框中心是否在GT和GT中心边长为5的正方形中的框中
    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        # gt_bboxes_per_image存 [x_center,y_center，w，h]
        # expanded_strides.shape为[1,num_anchors] ,例如[1,8400]，存了所有锚点对应的的步长，不只是一个特征图的
        expanded_strides_per_image = expanded_strides[0]                    # shape:[num_anchors]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image       # 每个anchor在原图中的偏移量
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        # 计算anchor中心坐标
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        )       # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = ((y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))


        # ------每个gt的left,right,top,bottom与anchor进行比较，计算anchor中心点是否在gt中，得到is_in_boxes_all(shape:[num_anchors])---------------
        # -------------------------------------------------------------------------------------
        # 这里计算出了gt的坐标,是相较于原图的
        # gt_bboxes_per_image存的是框中心坐标和w，h
        gt_bboxes_per_image_l = (   # 左边
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_r = (   # 右边
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_t = (   # 上边
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (   # 下边
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        )

        # 用每个anchor中心减去gt的坐标
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)      # shape = [num_GT , 8400 , 4]


        # 这里求出的是，哪些anchor的中心点是在gt内部的 , is_in_boxes.shape = [num_GT , 8400],存True或者False
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # 内部的4个值都大于0，说明anchor中心点在GT内部
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0    # ,sum求和以后，有True的都为True，shape = [8400], 内部为[False,  ...,  True,  True]

        # in fixed center

        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------
        # 每个gt的cx与cy向外扩展 2.5 * expanded_strides 距离得到 left_b,right_b,top_b,bottom_b，
        # 与anchor进行比较，计算anchor中心点是否包含在left_b,right_b,top_b,bottom_b中，
        # 得到is_in_centers_all(shape:[num_anchors])
        # in fixed center
        center_radius = 2.5

        # 计算向外扩大后的边，在GT中向外扩大成一个边长为5的正方形
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - \
                                center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + \
                                center_radius * expanded_strides_per_image.unsqueeze(0)

        # 计算anchor的中心点是否在GT中心扩大以后的框中
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        # -------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------


        # in boxes and in centers
        # shape:[num_anchors],如 torch.Size([8400])
        # anchor的中心点是在gt内部的 或者 在扩大之后的框里
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all    # 上述两个条件只要满足一个，就成为候选区域。注意！！！这里是“|”求或

        # ！！！shape:[num_gt, num_in_boxes_anchor]，如[6, 3957] 注意：这里是每一个gt与每一个候选区域的关系，
        # 这里一个anchor可能与多个gt存在候选关系
        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])    # 既在GT里，又在扩大的框里

        # is_in_boxes_anchor是至少满足一个条件的， is_in_boxes_and_center是2个条件都满足的
        return is_in_boxes_anchor, is_in_boxes_and_center

    # 给每个gt分配正样本，同时确定每个gt要分配几个正样本
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # cost.shape = [目标数，符合的anchor数]
        # pair_wise_ious.shape = [目标数，符合的anchor数]
        # gt_classes存的是类别对应的索引，如[3., 2., 6., 0.]
        # num_gt = 单张图片的目标数

        # --------------------dynamic_k确定逻辑，取预测值与gt拥有最大iou前10名的iou总和作为dynamic_k---------------------------
        # -------------------------------------------------------------------------------------------------------------
        # 首先按照cost值的大小，新建一个全0变量matching_matrix
        matching_matrix = torch.zeros_like(cost)    # cost.shape = [目标数，符合的anchor数]
        ious_in_boxes_matrix = pair_wise_ious       # pair_wise_ious.shape = [目标数，符合的anchor数]

        # n_candidate_k = 10
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))

        # 取预测值与gt拥有最大iou前10名的iou总和作为dynamic_k
        # topk_ious存最大的k个IOU，shape = [目标数，k]
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)   # 返回前k个最大值 和 索引

        # dynamic_ks 存每个物体前k个IOU的总和并取整，shape= [目标数]，如 [7, 8, 7, 8, 9]
        # min=1,即把dynamic_ks限制最小为1，保证一个gt至少有一个正样本
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)


        for gt_idx in range(num_gt):
            # 取cost排名最小的前dynamic_k个anchor作为postive，记录成本最小的索引
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)   # 返回前k个值和索引，False为从小到大

            # 记录下哪些是成本最小的 ， matching_matrix.shape = [目标数，符合的anchor数]
            matching_matrix[gt_idx][pos_idx] = 1.0
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        del topk_ious, dynamic_ks, pos_idx


        # -----------------------------------过滤共用的候选框-----------------------------------
        # anchor_matching_gt记录每个锚框的总和，用来判断是否有一个候选框对应多个GT的情况，shape = [符合的anchor数]
        anchor_matching_gt = matching_matrix.sum(0)     # 对每一列进行求和

        # 针对一个anchor匹配了2个及以上gt情况进行处理，大于1说明有预测框共用的情况
        if (anchor_matching_gt > 1).sum() > 0:  # 说明此anchor包含了成本最小的
            # 计算哪些锚框对哪些物体成本最小
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            # cost_argmin存取最小值对应 行（物体） 的索引

            # 先将共用anchor全都初始化为0，然后将物体对应cost最小的设为1，就去除了共用的情况
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # ------------------------------------------------------------------------------------



        # fg_mask_inboxes存哪些anchor是成本最小的,是最小的存1，不是的存0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0      # fg_mask_inboxes存布尔值，shape = [符合的anchor数],一般有几千个

        # num_fg = 所有物体动态K的总和
        num_fg = fg_mask_inboxes.sum().item()   # num_fg是一个int,统计有多少个True

        # fg_mask是判断锚框中心是否在GT中，fg_mask_inboxes存成本最低的
        # 从fg_mask中为True的部分中挑选，将成本最低的设为True，其余改为False
        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 把fg_mask中为True的部分

        # matched_gt_inds存的索引是指本张图片上对应的物体编号，用来锚框对应的是哪个物体， shape = [每个物体动态K的总和]
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)     # argmax返回最大值（1）的索引，matching_matrix内部存的都是0和1

        # gt_classes存的是类别（class）对应的索引，例如[3., 2., 6., 0.]，shape = [图片物体的数量]
        # gt_matched_classes存每个框对应物体的类别索引，shape = [所有物体动态K的总和]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # 记录下来每个框对应的IOU
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]


        # num_fg = 所有物体动态K的总和，是一个int
        # gt_matched_classes存每个框对应物体的类别编号,shape = [所有物体动态K的总和]
        # pred_ious_this_matching存每个框对应的IOU,shape = [所有物体动态K的总和]
        # matched_gt_inds存的索引是指本张图片上对应的物体编号,shape = [所有物体动态K的总和]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

# 返回BBOx的坐标
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    # bboxes_a.shape = [物体数量，4]
    # bboxes_b.shape = [符合的anchor数量，4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:    # 传入的是左上右下形式
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  # 比较 a和b的左上角左边，较大的为交叉区域的左上角坐标
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)   # 右下减左上得到宽高，宽高相乘得到面积
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:       # 传入的是中心坐标和宽高
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    # tl.type() =  torch.cuda.FloatTensor
    # tl < br =  tensor([[True, True],[True, True]....])
    # (tl < br).type(tl.type()) =  tensor([[1., 1.],[1., 1.],...])
    # en =  存1或者0
    en = (tl < br).type(tl.type()).prod(dim=2)  # 返回是否有交叉区域


    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())，得到交叉区域面积

    # 返回a和b的IOU
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="giou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # pred中的bbox是  中心店坐标 + w + h 格式
        eps = 1e-7
        assert pred.shape[0] == target.shape[0]



        # 用view函数转换成 多行4列的格式
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # 根据预测值和实际值来计算，交叉区域的左上角和右下角坐标
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))    # 交叉区域左上角坐标
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))    # 交叉区域右下角坐标


        area_p = torch.prod(pred[:, 2:], 1)     # torch.prod()返回 w * h, 得到了BBOX的面积
        area_g = torch.prod(target[:, 2:], 1)   # torch.prod()返回 w * h, 得到了真实框的面积

        # tl.type() =  torch.cuda.FloatTensor
        # tl < br =  tensor([[True, True],[True, True]....])
        # (tl < br).type(tl.type()) =  tensor([[1., 1.],[1., 1.],...])
        # en =  tensor([1., 1., 1., 1.,...])
        en = (tl < br).type(tl.type()).prod(dim=1)      # en应该是判断两个框有没有交集

        # 得到交叉区域的面积
        area_i = torch.prod(br - tl, 1) * en         # 得到交叉区域的面积,torch.prod返回向量的乘积

        # 计算IOU
        iou = area_i / (area_p + area_g - area_i + eps)

        if self.loss_type == "iou":
            # loss为  1 - IOU的平方
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # 计算包围框 C的 左上和右下坐标
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

            # 计算包围框C的面积
            area_c = torch.prod(c_br - c_tl, 1)

            # 计算GIOU的值，此处与公式不太一样
            # giou = iou - (area_c - area_p - area_g + area_i) / area_c.clamp(1e-16)        # 这条符合公式的写法
            giou = iou - (area_c - area_i) / area_c.clamp(eps)        # clamp限制最小值，Giithub代码的写法

            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "ciou":

            b1_x1, b1_x2 = pred[:, 0] - pred[:, 2] / 2, pred[:, 0] + pred[:, 2] / 2
            b1_y1, b1_y2 = pred[:, 1] - pred[:, 3] / 2, pred[:, 1] + pred[:, 3] / 2
            b2_x1, b2_x2 = target[:, 0] - target[:, 2] / 2, target[:, 0] + target[:, 2] / 2
            b2_y1, b2_y2 = target[:, 1] - target[:, 3] / 2, target[:, 1] + target[:, 3] / 2

            # Intersection area
            inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

            # Union Area
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
            union = w1 * h1 + w2 * h2 - inter + eps

            iou = inter / union  # iou

            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + eps
            # centerpoint distance squared
            # rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            rho2 = (((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4).clamp(0)

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (1 + eps - iou + v)
            ciou = iou - (rho2 / c2 + v * alpha)  # CIoU
            loss = 1 - ciou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()





        # print("************************************************************")
        # print("b1_x1 = ", b1_x1)
        # print("b1_x2 = ", b1_x2)
        # print("b1_y1 = ", b1_y1)
        # print("b1_y2 = ", b1_y2)
        # print("b2_x1 = ", b2_x1)
        # print("b2_x2 = ", b2_x2)
        # print("b2_y1 = ", b2_y1)
        # print("b2_y2 = ", b2_y2)
        # print("------------------------------------------------------------")
        # print("cw = ", cw)
        # print("ch = ", ch)
        # print("iou = ", iou)
        # print("iou = ", iou)
        # print("union = ",union)
        # print("ciou = ", ciou)
        # print("rho2 = ", rho2)
        # print("v = ", v)
        # print("alpha = ", alpha)
        # print("------------------------------------------------------------")

        return loss


if __name__ == "__main__":
    from config import opt
    #
    # torch.manual_seed(0)
    # opt.reid_dim = 0  # 0
    # opt.batch_size = 4
    #
    #
    # dummy_input = [torch.rand([2, 85 + opt.reid_dim, i, i]) for i in [64, 32, 16]]
    # dummy_target = torch.rand([2, 3,  5]) * 20  # [bs, max_obj_num, 5]
    # dummy_target[:, :, 0] = torch.randint(10, (2, 3), dtype=torch.int64)
    #
    #
    # yolof_loss = YOLOFLoss(label_name=opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums)
    # # print('input shape:', [i.shape for i in dummy_input])
    # # print("target shape:", dummy_target, dummy_target.shape)
    #
    # print("---------------------------------")
    # loss_status = yolof_loss(dummy_input, dummy_target)
    # for l in loss_status:
    #     print(l, loss_status[l])


    #
    # def Diou_loss(pred, target, eps=1e-7, reduction='mean'):
    #     b1_x1, b1_x2 = pred[:, 0] - pred[:, 2] / 2, pred[:, 0] + pred[:, 2] / 2
    #     b1_y1, b1_y2 = pred[:, 1] - pred[:, 3] / 2, pred[:, 1] + pred[:, 3] / 2
    #     b2_x1, b2_x2 = target[:, 0] - target[:, 2] / 2, target[:, 0] + target[:, 2] / 2
    #     b2_y1, b2_y2 = target[:, 1] - target[:, 3] / 2, target[:, 1] + target[:, 3] / 2
    #
    #     # Intersection area
    #     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
    #             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    #
    #     # Union Area
    #     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    #     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    #     union = (w1 * h1 + 1e-16) + w2 * h2 - inter
    #
    #     iou = inter / union  # iou
    #
    #     cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    #     ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    #
    #     # convex diagonal squared
    #     c2 = cw ** 2 + ch ** 2 + 1e-16
    #     # centerpoint distance squared
    #     rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    #
    #     v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    #     with torch.no_grad():
    #         alpha = v / (1 - iou + v)
    #     loss = iou - (rho2 / c2 + v * alpha)  # CIoU
    #     return loss
    # pred_box = torch.tensor([[4, 6, 4, 4], [9, 10.5, 8, 3]])
    # gt_box = torch.tensor([[5, 6.5, 4, 5]])
    # loss = Diou_loss(pred_box, gt_box)
    # print(loss)
