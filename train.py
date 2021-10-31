# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 20:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import print_function, division

import os
import shutil
import random
import time
import numpy as np
from progress.bar import Bar
import torch
import torch.nn as nn

from config import opt
from data.coco_dataset import get_dataloader
from models.YOLOF import get_model
from models.post_process import yolox_post_process
from utils.lr_scheduler import LRScheduler
from utils.util import AverageMeter, write_log, configure_module, occupy_mem
from utils.model_utils import save_model, load_model, clip_grads
from utils.ema import ModelEMA
from utils.data_parallel import set_device, _DataParallel
from utils.logger import Logger


def run_epoch(model_with_loss, optimizer, scaler, ema, phase, epoch, data_iter, num_iter, total_iter,
              train_loader=None, lr_scheduler=None):
    '''
    调用的代码为：
    loss_dict_train, _ = run_epoch(model, optimizer, scaler, ema, "train", epoch, train_iter, iter_per_train_epoch, total_train_iteration,
                train_loader, lr_scheduler)

    引用的部分参数是：
    scaler 和amp有关
    train_iter = iter(train_loader)
    iter_per_train_epoch = len(train_loader)
    total_train_iteration = opt.num_epochs * iter_per_train_epoch
    '''

    if phase == 'train':
        # 将模型设为训练模型，仅当模型中有 Dropout 和 BatchNorm 是才会有影响
        model_with_loss.train()
    else:
        # 将模型设为评价模型，仅当模型中有 Dropout 和 BatchNorm 是才会有影响
        model_with_loss.eval()
        torch.cuda.empty_cache()    # 释放缓存

    results, avg_loss_stats, last_opt_iter = {}, {}, 0
    data_time, batch_time = AverageMeter(), AverageMeter()  # AverageMeter()这个类是用来计算并存储平均值和当前值



    bar = Bar('{}'.format(opt.exp_id), max=num_iter)    # opt.exp_id = "coco_CSPDarknet-s_640x640"
    end = time.time()
    for iter_id in range(1, num_iter + 1):
        inps, targets, img_info, ind = next(data_iter)  # data_iter用了迭代

        # inps.shape = [8, 3, 640, 640],8个图，3通道，后面的图片尺寸会有变化
        inps = inps.to(device=opt.device, non_blocking=True)

        # targets.shape = [8, 120, 5]
        targets = targets.to(device=opt.device, non_blocking=True)


        data_time.update(time.time() - end)     # 记录时间

        if phase == 'train':
            # epoch 是当前训练的第几轮
            iteration = (epoch - 1) * num_iter + iter_id
            optimizer.zero_grad()

            # 得到损失值
            _, loss_stats = model_with_loss(inps, targets=targets)


            loss_stats = {k: v.mean() for k, v in loss_stats.items()}   # 所有的loss都取平均值


            # scaler 与自动混合精盾（amp）有关，将损失反向传播
            scaler.scale(loss_stats["loss"]).backward()

            #   opt.grad_clip是剪辑梯度，让训练更稳定
            if opt.grad_clip is not None and not opt.use_amp:
                # 取消scaler
                scaler.unscale_(optimizer)
                grad_normal = clip_grads(model_with_loss, opt.grad_clip)
                loss_stats['grad_normal'] = grad_normal

            # amp执行并更新
            scaler.step(optimizer)
            scaler.update()

            # ema更新
            ema.update(model_with_loss) if opt.ema else ''

            # 得到学习率
            lr = lr_scheduler.update_lr(iteration)

            for param_group in optimizer.param_groups:
                # param_group是一个字典，包括{params，lr, momentum, dampening, weight_decay, nesterov}
                param_group["lr"] = lr

            if (iteration - 1) % 50 == 0 and epoch <= 15:
                logger.scalar_summary("lr_iter_before_15_epoch", lr, iteration)

        else:   # 此时在验证集
            yolo_outputs, loss_stats = model_with_loss(inps, targets=targets)
            iteration, total_iter, lr = iter_id, num_iter, optimizer.param_groups[0]['lr']

            # 得到图片的比例和尺寸
            img_ratio = [float(min(opt.test_size[0] / img_info[0][i], opt.test_size[1] / img_info[1][i])) for i in
                         range(inps.shape[0])]      # 默认opt.test_size = (640, 640)
            img_shape = [[int(img_info[0][i]), int(img_info[1][i])] for i in range(inps.shape[0])]

            # 对数据进行后处理，predicts的格式为[label, conf, bbox]
            predicts = yolox_post_process(yolo_outputs, opt.stride, opt.num_classes, 0.01, opt.nms_thresh,
                                          opt.label_name, img_ratio, img_shape)

            # 把预测数据存入results
            for img_id, predict in zip(ind.cpu().numpy().tolist(), predicts):
                results[img_id] = predict

        batch_time.update(time.time() - end)
        end = time.time()

        shapes = "x".join([str(i) for i in inps.shape])

        # 记录日志
        Bar.suffix = '{phase}: total_epoch[{0}/{1}] total_batch[{2}/{3}] batch[{4}/{5}] |size: {6} |lr: {7} |Tot: ' \
                     '{total:} |ETA: {eta:} '.format(epoch, opt.num_epochs, iteration, total_iter, iter_id, num_iter,
                                                     shapes, "{:.8f}".format(lr), phase=phase, total=bar.elapsed_td,
                                                     eta=bar.eta_td)

        # loss_stats 包括 'loss', 'conf_loss', 'cls_loss', 'iou_loss', 'num_fg'
        for l in loss_stats:
            if l not in avg_loss_stats:
                avg_loss_stats[l] = AverageMeter()
            # 计算loss的平均值
            avg_loss_stats[l].update(loss_stats[l], inps.size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s |Batch {bt.val:.3f}s'.format(dt=data_time, bt=batch_time)



        # 打印训练信息
        if opt.print_iter > 0 and iter_id % opt.print_iter == 0:
            # print('{}| {}'.format(opt.exp_id, Bar.suffix))
            if iter_id % 10 == 0:       # 每10个打印一次
                print('{}| {}'.format(opt.exp_id, Bar.suffix))
            logger.write('{}| {}\n'.format(opt.exp_id, Bar.suffix))
        bar.next()


        # 随机进行放缩
        if phase == 'train' and opt.random_size is not None and (iteration % 10 == 0 or iteration <= 20):
            tensor = torch.LongTensor(2).to(device=opt.device)
            # 计算长高比
            size_factor = opt.input_size[1] * 1. / opt.input_size[0]

            # opt.random_size = (14, 26)  multi-size train: from 448(14*32) to 832(26*32)
            size = np.random.randint(*opt.random_size)  # 在 14-26之间生成随机的整数
            size = (int(32 * size), 32 * int(size * size_factor))   # 得到新的尺寸
            tensor[0], tensor[1] = size[0], size[1]
            if iteration <= 10:
                # 用最大的数值初始化，防止训练期间内存不足
                tensor[0], tensor[1] = int(max(opt.random_size) * 32), int(max(opt.random_size) * 32)

            # 修改参数
            train_loader.change_input_dim(multiple=(tensor[0].item(), tensor[1].item()), random_range=None)

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.

    # ret 是一个字典，格式为{'loss': 14.526, 'conf_loss': 8.50, 'cls_loss': 1.462, 'iou_loss': 4.562, 'num_fg': 1.3697, 'time': 0.133}
    # results评价时用
    return ret, results


def train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, no_aug):
    #  调用的指令为 train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, no_aug)

    best = -1
    iter_per_train_epoch = len(train_loader)
    iter_per_val_epoch = len(val_loader)


    # initialize data loader
    train_iter = iter(train_loader)
    total_train_iteration = opt.num_epochs * iter_per_train_epoch

    # exponential moving average    滑动平均（EMA）
    ema = ModelEMA(model)
    ema.updates = iter_per_train_epoch * start_epoch

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        # 如果到了最后几轮，则不开启数据增强
        if epoch == opt.num_epochs - opt.no_aug_epochs or no_aug:   # 默认最后15轮没有数据增强
            logger.write("--->No mosaic aug now! epoch {}\n".format(epoch))
            train_loader.close_mosaic()

            # 开启L1损失
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, _DataParallel):
                model.module.loss.use_l1 = True
            else:
                model.loss.use_l1 = True
            opt.val_intervals = 1       # 改为每一轮评价一次
            logger.write("--->Add additional L1 loss now! epoch {}\n".format(epoch))

        logger.scalar_summary("lr_epoch", optimizer.param_groups[0]['lr'], epoch)

        # ！！！！！！！！！！！！！！！！！！！！！！！运行一个epoch ！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        loss_dict_train, _ = run_epoch(model, optimizer, scaler, ema, "train", epoch, train_iter, iter_per_train_epoch,
                                       total_train_iteration, train_loader, lr_scheduler)
        logger.write('train epoch: {} |'.format(epoch))
        write_log(loss_dict_train, logger, epoch, "train")

        # 保存模型
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch,
                   ema.ema, logger=logger) if epoch % opt.save_epoch == 0 else ""
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, ema.ema, optimizer, scaler, logger=logger)


        # 根据设定的参数，评价验证集上的效果，保存最好的模型
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            # 开始评价
            logger.write('----------epoch {} start evaluate----------\n'.format(epoch))
            with torch.no_grad():
                loss_dict_val, preds = run_epoch(ema.ema, optimizer, None, None, "val", epoch, iter(val_loader),
                                                 iter_per_val_epoch, iter_per_val_epoch)
            logger.write('----------epoch {} evaluating ----------\n'.format(epoch))
            logger.write('val epoch: {} |'.format(epoch))

            # 记录ap
            ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, r = val_loader.dataset.run_coco_eval(preds, opt.save_dir)



            loss_dict_val["AP"], loss_dict_val["AP_0.5"], loss_dict_val["AP_0.75"] = ap, ap_0_5, ap_7_5
            loss_dict_val["AP_small"], loss_dict_val["AP_medium"] = ap_small, ap_medium
            loss_dict_val["AP_large"] = ap_large
            write_log(loss_dict_val, logger, epoch, "val")
            logger.write("\n{}\n".format(r))

            # 效果超过之前最好的就保存
            if ap >= best:
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, ema.ema, logger=logger)
                best = ap
            del loss_dict_val, preds



    logger.write("training finished... please use 'evaluate.sh' to get the final mAP on val dataset\n")
    logger.close()


def main():
    # define model with loss
    model = get_model(opt)      # 获取到模型

    # define optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():      # 返回网络中所有模块的迭代器，同时产生模块的名称以及模块本身
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):     # isinstance判断是否为子类
            pg2.append(v.bias)  # biases

        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):   # hasattr() 函数用于判断对象是否包含对应的属性
            pg1.append(v.weight)  # apply decay


    # 刚开始时 warmup_lr = 0, warmup_epochs = 5
    lr = opt.warmup_lr if opt.warmup_epochs > 0 else opt.basic_lr_per_img * opt.batch_size
    optimizer = torch.optim.SGD(pg0, lr=lr, momentum=opt.momentum, nesterov=True)   # momentum = 0.9
    optimizer.add_param_group({"params": pg1, "weight_decay": opt.weight_decay})  # add pg1 with weight_decay ， opt.weight_decay = 5e-4
    optimizer.add_param_group({"params": pg2})


    # Automatic mixed precision自动混合精度，默认是关的，训练开始前创建一次
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp, init_scale=2. ** 16)    # opt.use_amp = False


    # fine-tune or resume 微调或继续训练
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch, scaler = load_model(model, opt.load_model, optimizer, scaler, opt.resume)


    # define loader
    no_aug = start_epoch >= opt.num_epochs - opt.no_aug_epochs      # opt.num_epochs = 30, opt.no_aug_epochs = 15

    # ！！！！！！！！！！！！！！！！！！！！！！测试用的！！！！！！！！！！！！！！！！！！！！！！
    no_aug = True


    # 加载数据
    train_loader, val_loader = get_dataloader(opt, no_aug=no_aug)   # 加载训练集时有数据处理！！！！！

    # 提取所有类的名字，dataset_label的格式为['dog', 'person', 'train', 'sofa', 'chair'....]
    dataset_label = val_loader.dataset.classes

    # 判断类的名字是否一致
    assert opt.label_name == dataset_label, "[ERROR] 'opt.label_name' should be the same as dataset's {} != {}".format(
        opt.label_name, dataset_label)


    # learning ratio scheduler
    base_lr = opt.basic_lr_per_img * opt.batch_size
    lr_scheduler = LRScheduler(opt.scheduler, base_lr, len(train_loader), opt.num_epochs,
                               warmup_epochs=opt.warmup_epochs, warmup_lr_start=opt.warmup_lr,
                               no_aug_epochs=opt.no_aug_epochs, min_lr_ratio=opt.min_lr_ratio)

    # DP
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    if opt.occupy_mem and opt.device.type != 'cpu': # occupy_mem默认是False
        occupy_mem(opt.device)

    # 设置设备，CPU或者GPU，是否多GPU并行处理
    model, optimizer = set_device(model, optimizer, opt)

    # 开始训练
    train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, no_aug)


if __name__ == "__main__":
    configure_module()      # 配置环境

    if opt.seed is not None:    # seed默认是None
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    torch.backends.cudnn.benchmark = opt.cuda_benchmark     # 默认是True

    logger = Logger(opt)    # 生成一系列的日志文件
    shutil.copyfile("./config.py", logger.log_path + "/config.py")  # 把训练的config拷贝到每一个log文件夹中
    main()
