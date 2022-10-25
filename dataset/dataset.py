"""
Author: HaoZhi
Date: 2022-10-24 09:24:15
LastEditors: HaoZhi
LastEditTime: 2022-10-24 11:13:25
Description: 
"""
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from dataset.augmentation import transform

class_dict = {"helmet": 1, "head": 2}


class TrainDataSet(Dataset):
    def __init__(self, root, data_aug, aug_mode) -> None:
        super().__init__()
        self.img_folder = os.path.join(root, "train_images")
        self.anno_folder = os.path.join(root, "train_anns")
        self.file_names = list(
            map(lambda x: os.path.splitext(x)[0], os.listdir(self.img_folder))
        )
        self.transform = data_aug
        self.aug_mode = aug_mode

    def parse_img(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img

    def parse_anno(self, anno_path):
        labels = []
        boxes = []
        with open(anno_path, "r") as f:
            info = f.readlines()
        for f in info:
            f = f.strip().split(" ")
            if f[0] == "person":
                continue
            label = class_dict[f[0]]
            box = list(map(int, f[1:]))
            labels.append(label)
            boxes.append(box)
        return labels, boxes

    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder, self.file_names[index] + ".png")
        anno_path = os.path.join(self.anno_folder, self.file_names[index] + ".txt")
        img = self.parse_img(img_path)
        labels, boxes = self.parse_anno(anno_path)
        labels = torch.LongTensor(labels)
        boxes = torch.FloatTensor(boxes)
        new_img, new_boxes, new_labels = self.transform(img, boxes, labels, mode = self.aug_mode)
        return new_img, new_boxes, new_labels, img_path

    def __len__(self):
        return len(self.file_names)


def collect_fn(batch):
    images = list()
    boxes = list()
    labels = list()
    pathes = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
        pathes.append(b[3])

    images = torch.stack(images, dim=0)

    return images, boxes, labels, pathes


class ValidDataSet(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.img_folder = os.path.join(root, "test_images")
        self.file_names = list(
            map(lambda x: os.path.splitext(x)[0], os.listdir(self.img_folder))
        )
        self.file_names = sorted(self.file_names)
        self.transform = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def parse_img(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, index):
        img_path = os.path.join(self.img_folder, self.file_names[index] + ".png")
        img = self.parse_img(img_path)
        new_img = self.transform(img)
        return img, new_img, img_path

    def __len__(self):
        return len(self.file_names)
    
def collect_fn_valid(batch):
    ori_imgs = list()
    aug_imgs = list()
    pathes = list()
    for b in batch:
        ori_imgs.append(b[0])
        aug_imgs.append(b[1])
        pathes.append(b[2])
    aug_imgs = torch.stack(aug_imgs, dim = 0)
    return ori_imgs, aug_imgs, pathes


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    root = r"D:\workspace\tmp\安全帽佩戴位置识别挑战赛公开数据\安全帽佩戴位置识别挑战赛公开数据"
    plot_class_dict = {1: "helmet", 2: "head"}

    dataset = TrainDataSet(root=root, data_aug=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collect_fn)
    for idx, (imgs, boxes, labels, pathes) in enumerate(dataloader):
        for (img, box, label, path) in zip(imgs, boxes, labels, pathes):
            img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8").copy()
            for b, l in zip(box, label):
                l = plot_class_dict[l.item()]
                lx, ly, rx, ry = list(map(int, b))
                img = cv2.rectangle(img, (lx, ly), (rx, ry), (0, 255, 0), 1, 4)
                t_size, baseline = cv2.getTextSize(
                    l, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1
                )
                img = cv2.rectangle(
                    img,
                    (lx, ly - t_size[1] - baseline),
                    (lx + t_size[0], ly),
                    (0, 255, 0),
                    -1,
                )
                img = cv2.putText(
                    img,
                    l,
                    (lx, ly - baseline),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=1,
                )
            plt.imsave(
                os.path.join(
                    r"D:\workspace\detection\SSD\debug", path.split(os.sep)[-1]
                ),
                img,
            )
        break
