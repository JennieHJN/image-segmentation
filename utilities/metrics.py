import numpy as np

import torch
import torch.nn.functional as F


def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True:  # not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins=true_objects)[0]
        area_pred = np.histogram(y_pred, bins=pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:, 1:]
        union = union[1:, 1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1  # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0


def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:, 0, :, :]
    target = target[:, 0, :, :]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def voe(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    input_1 = output[:, 0, :, :]
    input_2 = output[:, 1, :, :]

    target_1 = target[:, 0, :, :]
    target_2 = target[:, 1, :, :]
    input_1 = input_1 > 0.5
    target_1 = target_1 > 0.5

    intersection_1 = (input_1 & target_1).sum()
    union_1 = (input_1 | target_1).sum()

    input_2 = input_2 > 0.5
    target_2 = target_2 > 0.5

    intersection_2 = (input_2 & target_2).sum()
    union_2 = (input_2 | target_2).sum()

    voe_1 = 1 - (intersection_1 + smooth) / (union_1 + smooth)

    voe_2 = 1 - (intersection_2 + smooth) / (union_2 + smooth)

    return voe_1, voe_2


def dice_coef(output, target):
    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:, 0, :, :]
    input_2 = output[:, 1, :, :]

    target_1 = target[:, 0, :, :]
    target_2 = target[:, 1, :, :]

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (input_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (input_2.sum() + target_2.sum() + smooth)

    return dice_1, dice_2


def rvd(output, target):
    smooth = 1e-5
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:, 0, :, :]
    input_2 = output[:, 1, :, :]

    target_1 = target[:, 0, :, :]
    target_2 = target[:, 1, :, :]

    rvd_score_1 = (target_1.sum() - input_1.sum() + smooth) / (input_1.sum() + smooth)
    rvd_score_2 = (target_2.sum() - input_2.sum() + smooth) / (input_2.sum() + smooth)

    return rvd_score_1, rvd_score_2

def hd95(output, target, spacing=(1.0, 1.0), threshold_liver=0.5, threshold_tumor=0.5,
         max_dist=20.0, verbose=False):
    """
    Calculate HD95 for liver and tumor based on 2D CT slice data.

    Args:
        output: Predicted logits, shape (batch_size, num_channels, H, W)
        target: Ground truth, shape (batch_size, num_channels, H, W)
        spacing: Voxel spacing in mm, default (1.0, 1.0) for x and y directions
        threshold_liver: Threshold for liver binarization, default 0.5
        threshold_tumor: Threshold for tumor binarization, default 0.5
        max_dist: Maximum distance cap in mm, default 20.0
        verbose: Whether to print debug info, default False

    Returns:
        tuple: (hd95_liver, hd95_tumor) mean HD95 values in mm for liver and tumor
    """
    # 输入验证
    if not (torch.is_tensor(output) or isinstance(output, np.ndarray)) or \
            not (torch.is_tensor(target) or isinstance(target, np.ndarray)):
        raise ValueError("Output and target must be PyTorch tensors or NumPy arrays")

    if len(spacing) != 2:
        raise ValueError("Spacing must be a tuple of length 2 (x, y)")

    if output.shape[1] < 2 or target.shape[1] < 2:
        raise ValueError("Input and target must have at least 2 channels (liver and tumor)")

    if output.shape[0] == 0:
        raise ValueError("Batch size cannot be 0")

    # 转换为 NumPy 数组
    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    # 二值化
    input_1 = (output[:, 0, :, :] > threshold_liver).astype(bool)  # Liver
    input_2 = (output[:, 1, :, :] > threshold_tumor).astype(bool)  # Tumor
    target_1 = (target[:, 0, :, :] > threshold_liver).astype(bool)
    target_2 = (target[:, 1, :, :] > threshold_tumor).astype(bool)

    hd95_liver_list, hd95_tumor_list = [], []
    batch_size = output.shape[0]

    for b in range(batch_size):
        pred_1, gt_1 = input_1[b], target_1[b]
        pred_2, gt_2 = input_2[b], target_2[b]

        # 计算距离变换
        dist_pred_1 = distance_transform_edt(~pred_1, sampling=spacing)
        dist_gt_1 = distance_transform_edt(~gt_1, sampling=spacing)
        dist_pred_2 = distance_transform_edt(~pred_2, sampling=spacing)
        dist_gt_2 = distance_transform_edt(~gt_2, sampling=spacing)

        # 肝脏 HD95
        if gt_1.sum() > 0 and pred_1.sum() > 0:
            distances_1 = np.concatenate([dist_gt_1[pred_1].flatten(), dist_pred_1[gt_1].flatten()])
            hd95_1 = np.percentile(distances_1, 95) if distances_1.size > 0 else max_dist
            hd95_1 = min(hd95_1, max_dist)
        elif gt_1.sum() == 0 and pred_1.sum() == 0:
            hd95_1 = 0
        else:
            hd95_1 = max_dist
        hd95_liver_list.append(hd95_1)

        # 肿瘤 HD95
        if gt_2.sum() > 0 and pred_2.sum() > 0:
            distances_2 = np.concatenate([dist_gt_2[pred_2].flatten(), dist_pred_2[gt_2].flatten()])
            hd95_2 = np.percentile(distances_2, 95) if distances_2.size > 0 else max_dist
            hd95_2 = min(hd95_2, max_dist)
        elif gt_2.sum() == 0 and pred_2.sum() == 0:
            hd95_2 = 0
        else:
            hd95_2 = max_dist
        hd95_tumor_list.append(hd95_2)

        if verbose:
            print(f"Batch {b}: Liver HD95: {hd95_1:.4f} mm, Tumor HD95: {hd95_2:.4f} mm")

    hd95_liver = np.mean(hd95_liver_list)
    hd95_tumor = np.mean(hd95_tumor_list)

    if verbose:
        print(f"Mean HD95 - Liver: {hd95_liver:.4f} mm, Tumor: {hd95_tumor:.4f} mm")

    return hd95_liver, hd95_tumor



def jaccard_index(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:, 0, :, :]  # Liver prediction
    input_2 = output[:, 1, :, :]  # Tumor prediction

    target_1 = target[:, 0, :, :]  # Liver ground truth
    target_2 = target[:, 1, :, :]  # Tumor ground truth

    # Threshold to binary masks (if necessary)
    input_1 = input_1 > 0.5
    target_1 = target_1 > 0.5

    input_2 = input_2 > 0.5
    target_2 = target_2 > 0.5

    # Jaccard index (IoU) for liver
    intersection_1 = (input_1 & target_1).sum()
    union_1 = (input_1 | target_1).sum()
    jaccard_1 = (intersection_1 + smooth) / (union_1 + smooth)

    # Jaccard index (IoU) for tumor
    intersection_2 = (input_2 & target_2).sum()
    union_2 = (input_2 | target_2).sum()
    jaccard_2 = (intersection_2 + smooth) / (union_2 + smooth)

    return jaccard_1, jaccard_2


# batch_size = output.shape[0]
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

def asd(output, target, spacing=(1.0, 1.0), threshold_liver=0.5, threshold_tumor=0.5,
        max_dist=20.0, verbose=False):
    """
    计算肝脏和肿瘤的 ASD，同步优化。
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = (output[:, 0, :, :] > threshold_liver).astype(bool)
    input_2 = (output[:, 1, :, :] > threshold_tumor).astype(bool)
    target_1 = (target[:, 0, :, :] > threshold_liver).astype(bool)
    target_2 = (target[:, 1, :, :] > threshold_tumor).astype(bool)

    asd_liver_list = []
    asd_tumor_list = []
    batch_size = output.shape[0]

    for b in range(batch_size):
        pred_1, gt_1 = input_1[b], target_1[b]
        pred_2, gt_2 = input_2[b], target_2[b]

        dist_pred_1 = distance_transform_edt(~pred_1) * spacing[0]
        dist_gt_1 = distance_transform_edt(~gt_1) * spacing[0]
        dist_pred_2 = distance_transform_edt(~pred_2) * spacing[0]
        dist_gt_2 = distance_transform_edt(~gt_2) * spacing[0]

        # 肝脏 ASD
        if pred_1.sum() > 0 and gt_1.sum() > 0:
            asd_1 = (dist_gt_1[pred_1].mean() + dist_pred_1[gt_1].mean()) / 2
            asd_1 = min(asd_1, max_dist)
        else:
            asd_1 = 0 if pred_1.sum() == 0 and gt_1.sum() == 0 else max_dist
        asd_liver_list.append(asd_1)

        # 肿瘤 ASD
        if pred_2.sum() > 0 and gt_2.sum() > 0:
            asd_2 = (dist_gt_2[pred_2].mean() + dist_pred_2[gt_2].mean()) / 2
            asd_2 = min(asd_2, max_dist)
        else:
            asd_2 = 0 if pred_2.sum() == 0 and gt_2.sum() == 0 else max_dist
        asd_tumor_list.append(asd_2)

        if verbose:
            print(f"Batch {b}: Liver ASD: {asd_1:.4f} mm, Tumor ASD: {asd_2:.4f} mm")

    asd_liver = np.mean(asd_liver_list)
    asd_tumor = np.mean(asd_tumor_list)

    if verbose:
        print(f"Mean ASD - Liver: {asd_liver:.4f} mm, Tumor: {asd_tumor:.4f} mm")

    return asd_liver, asd_tumor

def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)


def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / \
           (output.sum() + smooth)


def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
           (target.sum() + smooth)
