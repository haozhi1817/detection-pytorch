import torch
from torch.utils.data import DataLoader

from dataset.dataset import TrainDataSet, collect_fn
from dataset.augmentation import transform
from model.backbone import SSD300

num_classes = 3
batch_size = 64
data_root = ""
checkpoint_path = ""


device = torch.device("cuad:1" if torch.cuda.is_available() else "cpu")

plot_cls_dict = {1: 'helmet', 2: 'head'}

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

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels):
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)
    true_imgs = list()
    for i in range(len(true_labels)):
        true_imgs.extend([i] * true_labels[i].size(0))
    true_imgs = torch.LongTensor(true_imgs).to(device)#img_id
    true_boxes = torch.cat(true_boxes, dim = 0) #(total_num_true_obj)
    true_labels = torch.cat(true_labels, dim = 0) #(total_num_true_obj)

    det_imgs = list()
    for i in range(len(det_labels)):
        det_imgs.extend([i] * det_labels[i].size(0))
    det_imgs = torch.LongTensor(det_imgs).to(device) #img_id
    det_boxes = torch.cat(det_boxes, dim = 0)
    det_labels = torch.cat(det_labels, dim = 0)
    det_scores = torch.cat(det_scores, dim = 0)

    average_precisions = torch.zeros((num_classes - 1), dtype = torch.float)
    for c in range(1, num_classes):
        true_class_imgs = true_imgs[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_boxes_detected = torch.zeros(true_class_boxes.size(0), dtype = torch.uint8).to(device)
        num_class_objects = true_class_boxes.size(0)

        det_class_imgs = det_imgs[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue
        
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim = 0, descending= True)
        det_class_imgs = det_class_imgs[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(n_class_detections, dtype = torch.float).to(device)
        false_positives = torch.zeros(n_class_detections, dtype = torch.float).to(device)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_imgs[d]
            object_boxes = true_class_boxes[true_class_imgs == this_image]
            # 如果obj中没有该类别的box，则直接将这个pred置为fp
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim = 0)
            ori_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_imgs == this_image][ind]
            if max_overlap.item() > 0.5:
                if true_class_boxes_detected[ori_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[ori_ind] = 1
                else:
                    false_positives[d] = 1
        cumul_true_positives = torch.cumsum(true_positives, dim = 0)
        cumul_false_positives = torch.cumsum(false_positives, dim = 0)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_true_positives / num_class_objects

        recall_thresholds = torch.arange(start= 0, end = 1.1, step= .1)  .tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype = torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c-1] = precisions.mean()
    mean_average_precision = average_precisions.mean().item()
    average_precision = {plot_cls_dict[c+1]:v for c, v in enumerate(average_precisions.tolist())}
    return average_precision, mean_average_precision

def main():
    valid_dataset = TrainDataSet(root=data_root, data_aug=transform, aug_mode="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collect_fn,
    )

    model = SSD300(n_classes=num_classes, device=device)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        model.eval()

        for idx, (imgs, boxes, labels, pathes) in enumerate(valid_dataloader):
            imgs = imgs.to(device)
            predicted_locs, predicted_scores = model(imgs)
            det_boxes_batch, det_labels_batch, det_score_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
            )
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_score_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    print(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
