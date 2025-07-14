'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:16:07
Description: file function description
'''
import torchvision
import albumentations as A
import albumentations.pytorch as AP
import torch.utils.data as Data
import torch
import numpy as np
from PIL import Image
from skimage import io
import os
import random
import cv2

#LEVIR
mean_all, std_all = [0.39398650, 0.38862084, 0.33166478], [0.19670198, 0.18713744, 0.17206211]
mean_A, std_A = [0.44647953, 0.44253506, 0.37772246], [0.21731772, 0.20347068, 0.18538859]
mean_B, std_B = [0.34149347, 0.33470662, 0.28560711], [0.15698670, 0.15107087, 0.14347913]
mean_general, std_general = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Utilize your own path here
root_bt = r'D:\2_Experiments\3_Change_Detection\1_dataset\LEVIR_CD\LEVIR_CD_1024'
root_st = r'D:\2_Experiments\3_Change_Detection\1_dataset\LEVIR_CD\LEVIR_CD_1024'

class ChangeDataset_BCD_ST(Data.Dataset):
    def __init__(self, mode):
        assert mode in ['st_train']
        self.mode = mode
        self.edge_threshold = 0  # 边缘距离阈值
        self.num_factor = [5,10,15]
        self.image1_dir = os.path.join(root_st, self.mode, 'A')
        self.label1_dir = os.path.join(root_st, self.mode, 'EfficientSAM_VitS_mask_N5_to_N30', 'masks_vits', '30')
        self.ids = os.listdir(self.label1_dir)
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0, value=0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5)
        ], p=0.8, additional_targets={'image2': 'image', 'mask2': 'mask'})
        self.normalize1 = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_A, std=std_A)
        ])
        self.normalize2 = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_B, std=std_B)
        ])
        self.normalize_all = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_all, std=std_all)
        ])
        self.normalize_general = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_general, std=std_general)
        ])
        self.colorjit = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
        self.affine = torchvision.transforms.RandomAffine(degrees=(-5, 5), scale=(1, 1.02),translate=(0.02, 0.02), shear=(-5, 5))

    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, index):
        image_name = self.ids[index]
        image1 = np.array(Image.open(os.path.join(self.image1_dir, image_name)))
        label1 = np.array(Image.open(os.path.join(self.label1_dir, image_name)).convert('L'))
        return {'image1':image1, 'label1':label1, 'image_name':image_name}
    
    def change_synthesis(self, batchlist):
        image1 = []
        image2 = []
        label_change = []
        image_name_list = []
        num_objects_to_copy = random.choice(self.num_factor)
        for batch in range(0, len(batchlist)):
            origin_image1 = np.array(batchlist[batch]["image1"])
            origin_label1 = np.array(batchlist[batch]["label1"])
            image_name = batchlist[batch]["image_name"]
            image_name_list.append(image_name)

            generate_image2, generate_label2 = self.generate_change_detection_sample_intra_image(origin_image1, origin_label1, num_objects_to_copy)
            
            image_t1 = origin_image1
            image_t2 = np.array(self.colorjit(Image.fromarray(generate_image2)))
            label_t1 = origin_label1
            label_t2 = generate_label2
            
            augmented = self.transform(image = image_t1, mask = label_t1, image2 = image_t2, mask2 = label_t2)
            image_t1, image_t2, label_t1, label_t2 = augmented['image'], augmented['image2'], augmented['mask'], augmented['mask2']
            image_t1 = self.normalize_all(image_t1)
            image_t2 = self.normalize_all(image_t2)
            label_t1 = torch.tensor(label_t1)
            label_t2 = torch.tensor(label_t2)
            image1.append(image_t1)
            image2.append(image_t2)
            label_change.append((label_t1 != label_t2).unsqueeze(0).type(torch.float32))
        return {'image1':torch.stack(image1), 'image2':torch.stack(image2), 'label':torch.stack(label_change), 'image_name':image_name_list}
    
    def extract_instances(self, image, label, object_labels_to_copy, num_objects_to_copy):
        instances = []
        for label_value in object_labels_to_copy:
            mask = (label == label_value).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if self.edge_threshold == 0 or self.is_within_edge(contour, image.shape[1], image.shape[0]):
                    instance = image[y:y + h, x:x + w]
                    instances.append((instance, label_value, (x, y, w, h), contour))
        return random.sample(instances, min(num_objects_to_copy, len(instances)))
    
    def is_within_edge(self, contour, width, height):
        for point in contour:
            x, y = point[0]
            if x < self.edge_threshold or y < self.edge_threshold or x >= (width - self.edge_threshold) or y >= (height - self.edge_threshold):
                return False
        return True

    def paste_instances_intra_image(self, image, instances, label):
        new_image = image.copy()
        semantic_label_t2 = label.copy()
        for instance, label_value, (x, y, w, h), contour in instances:
            valid_position = False
            attempts = 0
            max_attempts = 100
            while not valid_position and attempts < max_attempts:
                new_x = 0
                new_y =0
                if w != image.shape[1]:
                    new_x = np.random.randint(self.edge_threshold, image.shape[1] - w - self.edge_threshold)
                if h != image.shape[0]:
                    new_y = np.random.randint(self.edge_threshold, image.shape[0] - h - self.edge_threshold)
                roi = new_image[new_y:new_y + h, new_x:new_x + w]
                roi_origin_label = label[new_y:new_y + h, new_x:new_x + w]
                roi_label = semantic_label_t2[new_y:new_y + h, new_x:new_x + w]

                valid_position = True
                mask = np.zeros_like(new_image)
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED, offset=(new_x - x, new_y - y))
                mask = mask[new_y:new_y + h, new_x:new_x + w]
                new_image[new_y:new_y + h, new_x:new_x + w] = np.where(mask == 255, instance, roi)
                semantic_label_t2[new_y:new_y + h, new_x:new_x + w] = np.where(mask[:, :, 0] == 255, label_value, roi_label)
                attempts += 1
        return new_image, semantic_label_t2

    def generate_change_detection_sample_intra_image(self, image_t1, semantic_label_t1, num_objects_to_copy):
        unique_labels = np.unique(semantic_label_t1)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            image_t2 = image_t1
            semantic_label_t2 = semantic_label_t1
        else:
            if len(unique_labels) == 1 and unique_labels[0] == 255:
                object_labels_to_copy = [255]
            else:
                object_labels_to_copy = random.sample(list(unique_labels), min(num_objects_to_copy, len(unique_labels)))
            instances = self.extract_instances(image_t1, semantic_label_t1, object_labels_to_copy, num_objects_to_copy)
            modified_image, semantic_label_t2 = self.paste_instances_intra_image(image_t1, instances, semantic_label_t1)
            image_t2 = modified_image
        return image_t2, semantic_label_t2

class ChangeDataset_BCD_BT(Data.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.image1_dir = os.path.join(root_bt, self.mode, 'A')
        self.image2_dir = os.path.join(root_bt, self.mode, 'B')
        self.label_dir = os.path.join(root_bt, self.mode, 'label')
        self.ids = os.listdir(self.label_dir)
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0, value=0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5)
        ], p=0.8, additional_targets={'image2': 'image'})
        self.normalize1 = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_A, std=std_A)
        ])
        self.normalize2 = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_B, std=std_B)
        ])
        self.normalize_all = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_all, std=std_all)
        ])
        self.normalize_general = torchvision.transforms.Compose([
             #这里除以255，scale至[0,1]
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean_general, std=std_general)
        ])

    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, index):
        image_name = self.ids[index]
        image1 = np.array(Image.open(os.path.join(self.image1_dir, image_name)))
        image2 = np.array(Image.open(os.path.join(self.image2_dir, image_name)))
        label = np.array(Image.open(os.path.join(self.label_dir, image_name)).convert('L'))
        label = label.clip(max=1)
        if self.mode == 'train':
            augmented = self.transform(image = image1, mask = label, image2 = image2)
            image1, image2, label = augmented['image'], augmented['image2'], augmented['mask']
        image1 = self.normalize1(image1)
        image2 = self.normalize2(image2)
        label = torch.from_numpy(label).unsqueeze(0).float()
        return {'image1':image1, 'image2':image2, 'label':label, 'image_name':image_name}

if __name__ == '__main__':
    train_data = ChangeDataset_BCD_ST(mode='st_train')
    train_dataloader = Data.DataLoader(train_data, num_workers=4, batch_size=16, shuffle=True, collate_fn=train_data.change_synthesis, drop_last=True)
    val_data = ChangeDataset_BCD_BT(mode='test')
    val_dataloader = Data.DataLoader(val_data, num_workers=4, batch_size=16, shuffle=False)

    for i, data in enumerate(train_dataloader):
        image_t1 = data['image1'].cuda()
        image_t2 = data['image2'].cuda()
        label = data['label'].cuda()
        print("label.shape", label.shape)
        train_name = data['image_name']
    for j, valdata in enumerate(val_dataloader):
        val_t1 = valdata['image1'].cuda()
        val_t2 = valdata['image2'].cuda()
        val_label = valdata['label'].cuda()
        val_names = valdata['image_name']