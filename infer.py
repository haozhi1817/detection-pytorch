import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.dataset import ValidDataSet, collect_fn_valid
from model.backbone import SSD300

valid_root = "/disk2/haozhi/tmp/ssd/data/test_images"
checkpoint_path = "/disk2/haozhi/tmp/ssd/code/ckpt/model_124.pth"
result_root = "/disk2/haozhi/tmp/ssd/code/result"
num_classes = 3
batch_size = 64

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

plot_cls_dict = {0: "background", 1: "helmet", 2: "head"}
plot_color_dict = {'helmet': (0, 255, 0), 'head': (0, 0, 255)}

def draw_anno(img, boxes, labels, scores):
    for (box, label, score) in zip(boxes, labels, scores):
        lx, ly, rx, ry = list(map(int, box))
        score = np.around(score.item(), decimals= 2)
        color = plot_color_dict[label]
        img = cv2.rectangle(img, (lx, ly), (rx, ry), color, 1, 4)
        t_size, baseline = cv2.getTextSize(
            " ".join([label, str(score)]),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            thickness=1,
        )
        img = cv2.rectangle(
            img, (lx, ly - t_size[1] - baseline), (lx + t_size[0], ly), color, -1
        )
        img = cv2.putText(
            img,
            " ".join([label, str(score)]),
            (lx, ly - baseline),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(0, 0, 0),
            thickness=1,
        )
    return img


def main(min_score, max_overlap, top_k):
    valid_dataset = ValidDataSet(root=valid_root)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collect_fn_valid,
    )
    model = SSD300(n_classes=num_classes, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for idx, (ori_imgs, aug_imgs, pathes) in enumerate(valid_dataloader):
            aug_imgs = aug_imgs.to(device)
            predicted_locs, predicted_scores = model(aug_imgs)
            det_boxes, det_labels, det_scores = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=min_score,
                max_overlap=max_overlap,
                top_k=top_k,
            )
            for (ori_img, det_box, det_label, det_score, path) in zip(
                ori_imgs, det_boxes, det_labels, det_scores, pathes
            ):
                det_box = det_box.to("cpu")
                ori_dims = torch.FloatTensor(
                    [ori_img.width, ori_img.height, ori_img.width, ori_img.height]
                ).unsqueeze(0)
                det_box = det_box * ori_dims
                det_label = [plot_cls_dict[l] for l in det_label.to("cpu").tolist()]
                ori_img = np.array(ori_img)
                if det_label == ["background"]:
                    anno_img =  ori_img
                else:
                    anno_img = ori_img
                    anno_img = draw_anno(anno_img, det_box, det_label, det_score)
                plt.imsave(os.path.join(result_root, path.split(os.sep)[-1]), anno_img)           

if __name__ == '__main__':
    main(min_score= 0.23, max_overlap= 0.4, top_k= 100)