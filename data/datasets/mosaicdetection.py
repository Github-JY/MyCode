#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random
import cv2
import numpy as np

from ..data_augment import box_candidates, random_perspective
from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # xc和yc是 mosaic的中心坐标， w, h是原图的高和宽，   input_h, input_w 默认都是640
    # TODO update doc
    # index0 左上角的图像
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 右上角的图像
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 左下角的图像
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 右下角的图像
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self, dataset, img_size, mosaic=True, preproc=None, degrees=10.0, translate=0.1, scale=(0.5, 1.5),
                 mscale=(0.5, 1.5), shear=2.0, perspective=0.0, enable_mixup=True, tracking=False):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):   传进来时是 (640, 640)
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):     传进来时是TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120, tracking=do_tracking)
            degrees (float):    10
            translate (float):  0.1
            scale (tuple):      opt.scale = (0.1, 2)
            mscale (tuple):     (0.5, 1.5)
            shear (float):      opt.shear = 2.0
            perspective (float):    opt.perspective = 0.0
            enable_mixup (bool):    opt.enable_mixup = True
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.tracking = tracking

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:  # 进行mosaiic

            # mosaic_labelsyong用来存 Box和cls
            mosaic_labels = []

            input_dim = self._dataset.input_dim     # Return: list: Tuple containing the current width,height
            input_h, input_w = input_dim[0], input_dim[1]   # 默认都是640


            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))  # random.uniform随机生成一个范围内的数
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices    再加上数据集中随机的三张图片的索引, indices的格式为 [1,2,3,4]
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                # i_mosaic是四张图对应的位置编号（左上角为0），index是原图在数据集中的编号

                # res的存BBox（左上右下）和cls ，img_info存原图的高度和宽度
                img, _labels, _, _ = self._dataset.pull_item(index)
                # 返回的是 img, res.copy(), img_info, id_,  _labels 的格式为 [[ 24. 109. 444. 324.  16.]]

                # 得到原始的 h，w
                h0, w0 = img.shape[:2]
                # 计算得到缩放比例
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                # 修改图片
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)    # 用114填满

                # 后缀 l 代表大图像(mosaic)上的坐标,  s 代表小图像（原图）上的坐标.计算对应图片在mosaic图像中的坐标
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                # 把对应的图片放入mosaic图中
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1
                labels = _labels.copy()     # 存BBox（左上右下）和cls，格式为 [[ 24. 109. 444. 324.  16.]]

                # Normalized xywh to pixel xyxy format，修改新的BBOx的坐标
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                # 把4个图片矩阵拼接到一起
                mosaic_labels = np.concatenate(mosaic_labels, 0)

                # clip函数将数值限制在给定的范围内
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            # 随机进行透视变换
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,       # 10
                translate=self.translate,   # 0.1
                scale=self.scale,           # opt.scale = (0.1, 2)
                shear=self.shear,           # 2.0
                perspective=self.perspective,   # 0.0
                border=[-input_h // 2, -input_w // 2],  # [-320,-320]
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------


            if self.enable_mixup and not len(mosaic_labels) == 0:
                # self.input_dim = [640, 640]
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)

            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            return mix_img, padded_labels, img_info, int(idx)

        else:   # 不进行mosaic
            self._dataset._input_dim = self.input_dim
            img, label, img_info, id_ = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, int(id_)

    # 此函数会将两张图片融合，返回新图片和对应的标签
    def mixup(self, origin_img, origin_labels, input_dim):
        # https://github.com/Megvii-BaseDetection/YOLOX/commit/d1e80118e1a6dd9fbde2e2c374f737997f117a59
        # input_dim = [640, 640]

        # self.mixup_scale =  (0.5, 1.5)
        jit_factor = random.uniform(*self.mixup_scale)

        # FLIP决定是否进行翻转
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []

        # 如果cp_labels为空，则随机选一张
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)    # 从数据集中所有图片中选出一张
            cp_labels = self._dataset.load_anno(cp_index)
            # cp_labels的格式为 [[ 25. 110. 293. 497.  11.]]

        # 加载对应的图片和标签，# cp_labels的格式为 [[ 25. 110. 293. 497.  11.]]
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        # 构建padding矩阵
        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0

        # 计算缩放比例，修改图片大小
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        # 给图片padding
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        # 将padding后的图片进行随机放缩
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        # 是否对图片进行翻转
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]

        # 对图片进行padding，方便后面裁切
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0

        # 如果之前的cp_img相比原图是放大的
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)

        # 进行裁切
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        # 计算缩放后的BBox坐标,不包括cls
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )

        # 如果FLIP== TRUE，计算翻转以后的 x坐标
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )

        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()

        # 把 BBOX限定在范围内，数据中不包括cls
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        # 选择出合适的BBOX， keep_lis格式为 [False  True False  True False False]
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)



        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()       # 下标为4的是cls
            box_labels = cp_bboxes_transformed_np[keep_list]    # 提取出符合要求的BOX，不包括cls

            if self.tracking:
                tracking_id_labels = cp_labels[keep_list, 5:6].copy()
                labels = np.hstack((box_labels, cls_labels, tracking_id_labels))
            else:
                # 把 box 和 cls 堆叠在一起
                labels = np.hstack((box_labels, cls_labels))

            # 把原始的标签和转换过的标签堆叠在一起
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)

            # 两张图片融合
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        # 返回融合后的图片和对应的标签
        return origin_img, origin_labels


# 此函数计算缩放后的BBox坐标
def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)   # 把 x坐标值限定在 0 ~ w_max
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)   # 把 y坐标值限定在 0 ~ h_max
    return bbox
