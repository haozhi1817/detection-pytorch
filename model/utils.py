"""
Author: HaoZhi
Date: 2022-10-20 14:44:09
LastEditors: HaoZhi
LastEditTime: 2022-10-20 14:44:23
Description: 
"""
import torch


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat(
        [
            gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:],
        ],
        1,
    )


def cxcy_to_xy(cxcy):
    return torch.cat(
        [cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1
    )


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat(
        [
            (cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
            torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5,
        ],
        1,
    )


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
