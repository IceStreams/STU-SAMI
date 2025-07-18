# STU-SAMI
Pytorch codes of **Integrating Segment Anything Model with Instance-Level Change Generation for Single-Temporal Unsupervised Change Detection** [[paper](https://ieeexplore.ieee.org/document/11071869)]

<img width="1585" height="1065" alt="image" src="https://github.com/user-attachments/assets/e3459799-9a4b-4b29-b3a9-453387a784f4" />

## Environment

```
  System: Windows10
  
  Python: 3.8.19
  
  torch: 2.1.2+cu118
  
  torchaudio: 2.1.2+cu118
  
  torchvision: 0.16.2+cu118
  
  numpy: 1.24.1
  
  albumentations: 1.4.3
  
  opencv-python: 4.9.0.80
  
  etc
```

## Dataset Download

In the following, we summarize the processed change detection data set used in this paper:

* [SECOND (Baidu)](https://pan.baidu.com/s/1RFhlO9_1KaFcIdTqblJIbA?pwd=dn84)
* SYSU-CD
* LEVIR-CD

## How to Use

1. Dataset preparation.
   * Please split the data into training and testing sets and organize them as follows:
```
      YOUR_DATA_DIR
      ├── ...
      ├── train
      │   ├── A
      │   ├── B
      │   ├── label(label_bn_vis for SECOND)
      ├── test
      │   ├── A
      │   ├── B
      │   ├── label(label_bn_vis for SECOND)
```

2. Construct single temporal data. (SYSU-CD as an example)

```
cd 0_Construct_Single_Temporal/SYSU-CD
   
python SYSU_construct_single_temporal_set.py
```

3. EfficientSAM mask generate. (SYSU-CD as an example)
   
```
cd 1_EfficientSAM_Generate
   
python EffSAM_VitS_seg_anything_SYSU_N10.py
```

(The pretrained weight of EfficientSAM-Vits is visible in the links below:

   [Baidu](https://pan.baidu.com/s/1yKN5yMVEPQEFRS_z3SAOHw?pwd=ymih))

4. Training
   
```
cd STUSAMI
   
bash my_train.sh
```

5. Inference and evaluation
   
```
bash my_inference.sh
```

## Pretrained Models

The reproducible weights of STU_SAMI on the three benchmark datasets are visible in the links below: [Baidu](https://pan.baidu.com/s/1o1s6pP2-ipGnoarKmvLOnA?pwd=awhr)

## Cite STU_SAMI

If you find this work useful or interesting, please consider citing the following BibTeX entry.

```
@ARTICLE{11071869,
  author={Zuo, Xibing and Rui, Jie and Ding, Lei and Jin, Fei and Lin, Yuzhun and Wang, Shuxiang and Liu, Xiao and Lei, Juan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Integrating Segment Anything Model With Instance-Level Change Generation for Single-Temporal Unsupervised Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-17},
  keywords={Training;Feature extraction;Image segmentation;Buildings;Adaptation models;Unsupervised learning;Remote sensing;Optimization;Generative adversarial networks;Annotations;Instance-level change generation;remote sensing;segment anything model (SAM);single-temporal unsupervised change detection (CD)},
  doi={10.1109/TGRS.2025.3586102}}
```



