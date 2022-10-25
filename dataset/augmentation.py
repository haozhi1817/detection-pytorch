"""
Author: HaoZhi
Date: 2022-10-24 09:46:42
LastEditors: HaoZhi
LastEditTime: 2022-10-24 10:59:53
Description: 
"""
import random

import torch
import torchvision.transforms.functional as FT

def find_intersection(set1, set2):
    lower_bounds = torch.max(set1[:, :2].unsqueeze(1), set2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set1[:, 2:].unsqueeze(1), set2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set1, set2):
    intersection = find_intersection(set1, set2)
    areas_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    areas_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])
    union = areas_set1.unsqueeze(1) + areas_set2.unsqueeze(0) - intersection
    return intersection / union


def photometric_distort(image):
    new_img = image
    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue,
    ]

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is "adjust_hue":
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_img = d(new_img, adjust_factor)
    return new_img


def expand(image, boxes, filler):
    ori_h = image.size(1)
    ori_w = image.size(2)
    max_scale = 3
    scale = random.uniform(1, max_scale)
    new_h = int(scale * ori_h)
    new_w = int(scale * ori_w)
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(
        1
    ).unsqueeze(1)
    left = random.randint(0, new_w - ori_w)
    right = left + ori_w
    top = random.randint(0, new_h - ori_h)
    bottom = top + ori_h
    new_image[:, top:bottom, left:right] = image
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    return new_image, new_boxes


def random_crop(image, boxes, labels):
    ori_h = image.size(1)
    ori_w = image.size(2)
    while True:
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])
        if min_overlap is None:
            return image, boxes, labels

        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * ori_h)
            new_w = int(scale_w * ori_w)
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left = random.randint(0, ori_w - new_w)
            right = left + new_w
            top = random.randint(0, ori_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)
            if overlap.max().item() < min_overlap:
                continue
            new_image = image[:, top:bottom, left:right]
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bottom)
            )
            if not centers_in_crop.any():
                continue
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]
            return new_image, new_boxes, new_labels

def flip(image, boxes):
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:,0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:,2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes

def resize(image, boxes, dims = (300, 300), return_percent_coords = True):
    new_image = FT.resize(image, dims)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
    
    return new_image, new_boxes


def transform(img, boxes, labels, mode):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_img = img
    new_boxes = boxes
    new_labels = labels

    if mode == 'train':

        new_img = photometric_distort(new_img)
        new_img = FT.to_tensor(new_img)

        if random.random() < 0.5:
            new_img, new_boxes = expand(new_img, boxes, filler=mean)
        
        new_img, new_boxes, new_labels = random_crop(new_img, new_boxes, new_labels)

        new_img = FT.to_pil_image(new_img)

        if random.random() < 0.5:
            new_img, new_boxes = flip(new_img, new_boxes)
    
    new_img, new_boxes = resize(new_img, new_boxes, dims = (300, 300), return_percent_coords= True)
    new_img = FT.to_tensor(new_img)
    new_img = FT.normalize(new_img, mean = mean, std = std)
    return new_img, new_boxes, new_labels
