'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:16:07
Description: file function description
'''
from skimage import io
import os
import numpy as np

def save_preds_visualization(val_preds, val_labels, save_path, image_names):
    val_preds = np.squeeze(val_preds, axis=1)
    val_labels = np.squeeze(val_labels, axis=1)
    # 定义颜色
    colors = {
        'TP': [255, 255, 255],  # 白色
        'TN': [0, 0, 0],        # 黑色
        'FP': [255, 0, 0],      # 红色
        'FN': [0, 255, 0]       # 绿色
    }
    
    for i, (val_pred, val_label) in enumerate(zip(val_preds, val_labels)):
        colored_image = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
        val_pred = np.uint8(np.where(val_pred > 0, 1, 0))
        val_label = np.uint8(np.where(val_label > 0, 1, 0))

        # 计算 TP, TN, FP, FN 并填充颜色
        tp = (val_pred == 1) & (val_label == 1)
        tn = (val_pred == 0) & (val_label == 0)
        fp = (val_pred == 1) & (val_label == 0)
        fn = (val_pred == 0) & (val_label == 1)
        colored_image[tp] = colors['TP']
        colored_image[tn] = colors['TN']
        colored_image[fp] = colors['FP']
        colored_image[fn] = colors['FN']
        
        io.imsave(os.path.join(save_path, image_names[i]), colored_image, check_contrast=False)
        # print(image_names[i])
    return 0

def save_preds_binary_visualization(val_preds, save_path, image_names):
    val_preds = np.squeeze(val_preds, axis=1)
    for i, val_pred in enumerate(val_preds):
        pred_image = np.uint8(np.where(val_pred > 0, 255, 0))
        io.imsave(os.path.join(save_path, image_names[i]), pred_image, check_contrast=False)
        # print(image_names[i])
    return 0

def save_preds_binary(val_preds, save_path, image_names):
    val_preds = np.squeeze(val_preds, axis=1)
    for i, val_pred in enumerate(val_preds):
        pred_image = np.uint8(np.where(val_pred > 0, 1, 0))
        io.imsave(os.path.join(save_path, image_names[i]), pred_image, check_contrast=False)
    return 0