'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:16:07
Description: file function description
'''
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def ConfusionMatrix(num_classes, pres, gts):
    def __get_hist(pre, gt):
        # pre = pre.cpu().detach().numpy()
        # gt = gt.cpu().detach().numpy()
        pre = np.where(pre >= 0.5, 1, 0)
        gt = np.where(gt >= 0.5, 1, 0)
        mask = (gt >= 0) & (gt < num_classes)
        label = num_classes * gt[mask].astype(int) + pre[mask].astype(int)
        hist = np.bincount(label, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(pres, gts):
        cm += __get_hist(lt.flatten(), lp.flatten())
    return cm

def get_score(confusionMatrix):
    precision = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=0) + np.finfo(np.float32).eps)
    recall = np.diag(confusionMatrix) / (confusionMatrix.sum(axis=1) + np.finfo(np.float32).eps)
    f1score = 2 * precision * recall / ((precision + recall) + np.finfo(np.float32).eps)
    iou = np.diag(confusionMatrix) / (
            confusionMatrix.sum(axis=1) + confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) + np.finfo(
        np.float32).eps)
 
    po = np.diag(confusionMatrix).sum() / (confusionMatrix.sum() + np.finfo(np.float32).eps)
    pe = np.sum(confusionMatrix.sum(axis=0) * confusionMatrix.sum(axis=1)) / (confusionMatrix.sum() ** 2 + np.finfo(np.float32).eps)
    kappa = (po - pe) / (1 - pe + np.finfo(np.float32).eps)
    acc = np.diag(confusionMatrix).sum() / (confusionMatrix.sum() + np.finfo(np.float32).eps)
    mIoU = np.nanmean(iou)
    mF1 = np.nanmean(f1score)
    return precision, recall, f1score, iou, kappa, acc, mIoU, mF1

def get_score_sum(confusionMatrix):
    num_classes = confusionMatrix.shape[0]
    precision, recall, f1score, iou, kappa, acc, mIoU, mF1 = get_score(confusionMatrix)
    cls_precision = dict(zip(['precision_' + str(i) for i in range(num_classes)], precision))
    cls_recall = dict(zip(['recall_' + str(i) for i in range(num_classes)], recall))
    cls_f1 = dict(zip(['f1_' + str(i) for i in range(num_classes)], f1score))
    cls_iou = dict(zip(['iou_' + str(i) for i in range(num_classes)], iou))
    return cls_precision, cls_recall, cls_f1, cls_iou, mIoU, acc, mF1, kappa
