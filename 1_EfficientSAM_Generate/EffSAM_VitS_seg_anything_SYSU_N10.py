'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:20:42
Description: file function description
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
#from PIL import Image
from skimage import io
import cv2
import os

GRID_SIZE = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
from torchvision.ops.boxes import batched_nms, box_area
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)

def process_small_region(rles):
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.7
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    predicted_masks, predicted_iou = model(img[None, ...], points, point_labels)
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2)
    predicted_masks = predicted_masks[0]
    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]
    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]
    masks = torch.ge(masks, 0.0)
    return masks, iou_

def run_everything_ours(image_path, model):
    # model = model.cuda()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)
    _, original_image_h, original_image_w = img_tensor.shape
    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy))
    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1)
    with torch.no_grad():
      predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
              img_tensor.cuda(),
              points.reshape(1, num_pts, 1, 2).cuda(),
              point_labels.reshape(1, num_pts, 1).cuda(),
              model)
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle)
    return predicted_masks

# def show_anns(mask):
#     img = np.ones((mask[0].shape[0], mask[0].shape[1], 3))*255
#     #img[:,:,3] = 0
#     for ann in mask:
#         m = ann
#         color = list(np.random.choice(range(256), size=3))
#         #color_mask = np.concatenate([np.random.random(3), [0.5]])
#         img[m] = color
#     return img.astype(np.uint8)

def show_anns(mask):
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    colored_image = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)*255
    for label in unique_labels:
        color = np.random.randint(0, 255, size=3).tolist()
        colored_image[mask == label] = color
    return colored_image.astype(np.uint8)

def index_mask(mask):
    img = np.zeros((mask[0].shape[0], mask[0].shape[1])).astype(np.uint8)
    #img[:,:,3] = 0
    for idx, ann in enumerate(mask):
        m = ann
        img[m] = idx+1
    return img
    

def main():
    sam_model = build_efficient_sam_vits()
    # sam_model = sam_model.cuda()
    sam_model = torch.nn.DataParallel(sam_model)
    sam_model = sam_model.cuda()
    sam_model.eval()

    # Utilize your own path here
    SrcDir = r'D:\2_Experiments\3_Change_Detection\1_dataset\SYSU-CD\st_train\A'
    DstDir_masks  = './genarate_results/SYSU/st_train/masks_vits/' + str(GRID_SIZE) + '/'
    DstDir_colors = './genarate_results/SYSU/st_train/colors_vits/'+ str(GRID_SIZE) + '/'
    DstDir_img_colors = './genarate_results/SYSU/st_train/img_colors_vits/'+ str(GRID_SIZE) + '/'

    # SrcDir = r'D:\2_Experiments\3_Change_Detection\20_EfficientSAM_Generate\data'
    # DstDir  = './genarate_results/CLCD/train/masksA_vitt/'
    if not os.path.exists(DstDir_masks): os.makedirs(DstDir_masks)
    if not os.path.exists(DstDir_colors): os.makedirs(DstDir_colors)
    if not os.path.exists(DstDir_img_colors): os.makedirs(DstDir_img_colors)
    
    data_list = os.listdir(SrcDir)
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            img_src_path = os.path.join(SrcDir, it)
            img_dst_path_masks = os.path.join(DstDir_masks, it)
            img_dst_path_colors = os.path.join(DstDir_colors, it)
            img_dst_path_img_colors = os.path.join(DstDir_img_colors, it)
            image = io.imread(img_src_path)
            
            masks = run_everything_ours(img_src_path, sam_model)
            torch.cuda.empty_cache()
            
            if len(masks)>255: masks=masks[:255]            
            masks_int = index_mask(masks)
            unique_values = np.unique(masks_int)
            remapped_masks_int = np.searchsorted(unique_values, masks_int).astype(np.uint8)
            io.imsave(img_dst_path_masks, remapped_masks_int, check_contrast=False)
            
            mask_color = show_anns(remapped_masks_int)
            io.imsave(img_dst_path_colors, np.uint8(mask_color), check_contrast=False)

            image = image*0.5 + mask_color*0.5
            io.imsave(img_dst_path_img_colors, np.uint8(image), check_contrast=False)
            
            print('%d/%d image processed, %d masks generated.'%(idx+1, len(data_list), len(masks)))
    

if __name__ == '__main__':
    main()
















