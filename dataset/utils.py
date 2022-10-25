"""
Author: HaoZhi
Date: 2022-10-22 17:45:52
LastEditors: HaoZhi
LastEditTime: 2022-10-22 18:12:17
Description: 
"""
import os
import glob
from collections import Counter

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def statistic_img(data_folder):
    imgs = glob.glob(os.path.join(data_folder, "*.png"))
    img_sizes = list(map(lambda x: Image.open(x).size, imgs))
    img_sizes = np.array(img_sizes)
    print("img_size: ", img_sizes.shape)
    print("min_x: ", np.min(img_sizes[:, 0]), "max_x: ", np.max(img_sizes[:, 0]))
    print("min_y: ", np.min(img_sizes[:, 1]), "max_y: ", np.max(img_sizes[:, 1]))


def parse_img(img_file):
    img = Image.open(img_file).convert('RGB')
    img = np.array(img)
    return img

def parse_anno(anno_file):
    with open(anno_file, "r") as f:
        annos = f.readlines()
    num_annos = len(annos)
    cls_list = list(map(lambda x: x.strip().split(" ")[0], annos))
    lx_list = list(map(lambda x: int(x.strip().split(" ")[1]), annos))
    ly_list = list(map(lambda x: int(x.strip().split(" ")[2]), annos))
    rx_list = list(map(lambda x: int(x.strip().split(" ")[3]), annos))
    ry_list = list(map(lambda x: int(x.strip().split(" ")[4]), annos))
    return num_annos, cls_list, lx_list, ly_list, rx_list, ry_list


def statistic_annos(data_folder):
    annos = glob.glob(os.path.join(data_folder, "*.txt"))
    print('num_files: ', len(annos))
    total_anno_list = []
    total_cls_list = []
    total_lx_list = []
    total_ly_list = []
    total_rx_list = []
    total_ry_list = []
    for anno in annos:
        num_annos, cls_list, lx_list, ly_list, rx_list, ry_list = parse_anno(anno)
        total_anno_list.append(num_annos)
        total_cls_list.extend(cls_list)
        total_lx_list.extend(lx_list)
        total_ly_list.extend(ly_list)
        total_rx_list.extend(rx_list)
        total_ry_list.extend(ry_list)
    min_annos = np.min(total_anno_list)
    max_annos = np.max(total_anno_list)
    num_annos = np.sum(total_anno_list)
    min_lx = np.min(total_lx_list)
    max_lx = np.max(total_lx_list)
    min_ly = np.min(total_ly_list)
    max_ly = np.max(total_ly_list)
    min_rx = np.min(total_rx_list)
    max_rx = np.max(total_rx_list)
    min_ry = np.min(total_ry_list)
    max_ry = np.max(total_ry_list)
    print(
        "min_annos: ",
        min_annos,
        "max_annos: ",
        max_annos,
        "num_annos: ",
        num_annos,
        "min_lx: ",
        min_lx,
        "max_lx: ",
        max_lx,
        "min_ly: ",
        min_ly,
        "max_ly: ",
        max_ly,
        "min_rx: ",
        min_rx,
        "max_rx: ",
        max_rx,
        "min_ry: ",
        min_ry,
        "max_ry: ",
        max_ry,
    )
    cls_count = Counter(total_cls_list)
    print('cls_count: ', cls_count)
    min_w = np.min(np.array(total_rx_list) - np.array(total_lx_list))
    max_w = np.max(np.array(total_rx_list) - np.array(total_lx_list))
    min_h = np.min(np.array(total_ry_list) - np.array(total_ly_list))
    max_h = np.max(np.array(total_ry_list) - np.array(total_ly_list))
    print('min_w: ', min_w, 'max_w: ', max_w, 'min_h: ', min_h, 'max_h: ', max_h)

def draw_box(img_file, anno_file):
    img = parse_img(img_file)
    _, cls_list, lx_list, ly_list, rx_list, ry_list = parse_anno(anno_file)
    for (cls_label, lx, ly, rx, ry) in zip(cls_list, lx_list, ly_list, rx_list, ry_list):
        img = cv2.rectangle(img, (lx, ly), (rx, ry), (0, 255, 0), 1, 4)
        t_size, baseline = cv2.getTextSize(cls_label, fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1, thickness = 1)
        img = cv2.rectangle(img, (lx, ly - t_size[1] - baseline), (lx + t_size[0], ly), (0, 255, 0), -1)
        img = cv2.putText(img, cls_label, (lx, ly - baseline), fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1, color = (0, 0, 0), thickness = 1)
    return img
    




if __name__ == "__main__":
    # debug statistic_img
    #data_folder = r"D:\workspace\tmp\安全帽佩戴位置识别挑战赛公开数据\安全帽佩戴位置识别挑战赛公开数据\train_images"
    #statistic_img(data_folder)

    # debug statistic_anno
    # data_folder = r'D:\workspace\tmp\安全帽佩戴位置识别挑战赛公开数据\安全帽佩戴位置识别挑战赛公开数据\train_anns\train_anns'
    # statistic_annos(data_folder)

    #debug draw_box
    # img_folder = ''
    # anno_folder = ''
    # save_folder = ''
    # file_names = os.listdir(img_folder)
    # for file_name in file_names:
    #     img_path = os.path.join(img_folder, file_name)
    #     anno_path = os.path.join(img_folder, file_name.replace('.png', '.txt'))
    #     img = draw_box(img_path, anno_path)
    #     save_path = os.path.join(save_folder, file_name)
    #     plt.imsave(save_path, img)
    img = draw_box(r'd:\workspace\tmp\安全帽佩戴位置识别挑战赛公开数据\安全帽佩戴位置识别挑战赛公开数据\train_images\hard_hat_workers2788.png', r'D:\workspace\tmp\安全帽佩戴位置识别挑战赛公开数据\安全帽佩戴位置识别挑战赛公开数据\train_anns\train_anns\hard_hat_workers2788.txt')
    plt.imsave('test.jpg', img)
