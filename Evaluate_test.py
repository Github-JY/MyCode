import torch
from config import opt
from utils.ema import ModelEMA
import torch.nn as nn
from utils.model_utils import save_model, load_model, clip_grads
from models.YOLOF import get_model
from data.coco_dataset import get_dataloader
from models.post_process import yolox_post_process

if __name__ == "__main__":


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

    optimizer = torch.optim.SGD(pg0, lr=opt.basic_lr_per_img, momentum=opt.momentum, nesterov=True)  # momentum = 0.9
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": opt.weight_decay})  # add pg1 with weight_decay ， opt.weight_decay = 5e-4
    optimizer.add_param_group({"params": pg2})

    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp, init_scale=2. ** 16)    # opt.use_amp = False



    # 加载数据集和模型
    train_loader, val_loader = get_dataloader(opt, no_aug=False)
    iter_per_val_epoch = len(val_loader)
    model_with_loss, optimizer, start_epoch, scaler = load_model(model, opt.load_model, optimizer, scaler, opt.resume)



    # exponential moving average    滑动平均（EMA）model
    ema = ModelEMA(model)
    ema.updates = 0




    with torch.no_grad():
        model_with_loss.eval()
        torch.cuda.empty_cache()

        results, avg_loss_stats, last_opt_iter = {}, {}, 0

        data_iter = iter(val_loader)
        num_iter = len(val_loader)
        for iter_id in range(1, num_iter + 1):
            inps, targets, img_info, ind = next(data_iter)  # data_iter用了迭代

            # inps.shape = [8, 3, 640, 640],8个图，3通道，后面的图片尺寸会有变化
            inps = inps.to(device="cuda:0" , non_blocking=True)


            # targets.shape = [8, 120, 5]
            targets = targets.to(device="cuda:0" , non_blocking=True)


            #  --------------------------------------------------

            model = model_with_loss.cuda()
            yolo_outputs, loss_stats = model(inps, targets=targets)
            iteration, total_iter, lr = iter_id, num_iter, optimizer.param_groups[0]['lr']

            # 得到图片的比例和尺寸
            img_ratio = [float(min(opt.test_size[0] / img_info[0][i], opt.test_size[1] / img_info[1][i])) for i in
                         range(inps.shape[0])]  # 默认opt.test_size = (640, 640)
            img_shape = [[int(img_info[0][i]), int(img_info[1][i])] for i in range(inps.shape[0])]

            # 对数据进行后处理，predicts的格式为[label, conf, bbox]
            predicts = yolox_post_process(yolo_outputs, opt.stride, opt.num_classes, 0.001, opt.nms_thresh,
                                          opt.label_name, img_ratio, img_shape)

            # 把预测数据存入results
            for img_id, predict in zip(ind.cpu().numpy().tolist(), predicts):
                results[img_id] = predict






    # 记录ap
    ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, r = val_loader.dataset.run_coco_eval(results, opt.save_dir)

