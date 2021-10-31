import matplotlib.pyplot as plt
import math

def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr


def new_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """Cosine learning rate with warm up."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr
iters = []
LR1 = []
LR2 = []
LR3 = []
LR4 = []
LR5 = []
LR6 = []
for i in range(1200):
    iters.append(i)

    lr1 = yolox_warm_cos_lr(0.01/8.0,  0.05, 1200, 500, 0, 0, i)
    LR1.append(lr1)

    lr2 = yolox_warm_cos_lr(0.01/16.0, 0.05, 1200, 300, 0, 0, i)
    LR2.append(lr2)

    lr3 = yolox_warm_cos_lr(0.01 / 8.0, 0.05, 1200, 100, 0, 0, i)
    LR3.append(lr3)

    lr4 = yolox_warm_cos_lr(0.01 / 64.0, 0.05, 1200, 500, 0, 0, i)
    LR4.append(lr4)

    lr5 = yolox_warm_cos_lr(0.01 / 40, 0.05, 1200, 200, 0, 0, i)
    LR5.append(lr5)

    lr6 = new_lr(0.01 / 40, 0.05, 1200, 200, 0, 100, i)
    LR6.append(lr6)


# 画图
plt.plot(iters,LR1)
plt.plot(iters,LR2)
plt.plot(iters,LR3)
plt.plot(iters,LR4)
plt.plot(iters,LR5)
plt.plot(iters,LR6)



plt.show()