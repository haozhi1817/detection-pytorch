"""
Author: HaoZhi
Date: 2022-10-24 14:05:42
LastEditors: HaoZhi
LastEditTime: 2022-10-24 14:12:58
Description: 
"""
import os
import time

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import TrainDataSet, collect_fn
from dataset.augmentation import transform
from model.backbone import SSD300, MultiBoxLoss

train_root = "/disk2/haozhi/tmp/ssd/data"
num_class = 3
batch_size = 128
lr = 1e-3
wd = 5e-4
num_epoch = 600
device = torch.device("cuda:1")

log_root = "/disk2/haozhi/tmp/ssd/code/log"
ckpt_root = "/disk2/haozhi/tmp/ssd/code/ckpt"
reseum_path = '/disk2/haozhi/tmp/ssd/code/ckpt/model_124.pth'
cudnn.benchmark = True


def main():
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)

    train_dataset = TrainDataSet(root=train_root, data_aug=transform, aug_mode= 'train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collect_fn,
    )

    num_batch = len(train_loader)

    model = SSD300(num_class, device)
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith(".bias"):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(
        params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}],
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=lr, eta_min=lr * 1e-3
    )

    model = model.to(device)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

    if reseum_path:
        checkpoint = torch.load(reseum_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = SummaryWriter(log_root)

    for epoch in range(num_epoch):
        model.train()
        start = time.time()
        for idx, (imgs, boxes, labels, _) in enumerate(train_loader):
            data_time = time.time() - start
            imgs = imgs.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(imgs)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step= epoch * num_batch + idx)
            writer.add_scalar('loss', loss.item(), global_step= epoch * num_batch + idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time = time.time() - start
            start = time.time()
            print(
                "Epoch: [{:>3d}][{:>3d}/{:>3d}]\t"
                "Batch Time: {:.3f}\t"
                "Data Time: {:.3f}\t"
                "Loss: {:.4f}".format(
                    epoch, idx, num_batch, batch_time, data_time, loss.item()
                )
            )
        lr_scheduler.step()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(ckpt_root, 'model_' + str(epoch) + '.pth'))

if __name__ == '__main__':
    main()
