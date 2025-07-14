'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:16:07
Description: file function description
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import torch.nn as nn
from utils import metrics, visualization
from datasets import ST_SYSU as dataset
from models.MyNet import changeNet as Net
import argparse
from thop import profile
import copy

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="ST_SYSU")
        parser.add_argument("--net_name", type=str, default="MyNet")
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--load_from", type=str, default=r'checkpoints\ST_SYSU_MyNet\Val_best_ST_SYSU_MyNet_epoch68_OA80.25_F62.51_mF74.55_IoU45.46_mIoU60.91_Recall69.83_Precision56.58_Kappa49.30_loss0.4779.pth')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

class Preder:
    def __init__(self, args):
        self.args = args
        self.working_path = os.path.dirname(os.path.abspath(__file__))
        self.pred_dir = os.path.join(self.working_path, 'pred_results', self.args.data_name + '_' + self.args.net_name)
        self.pred_save_path_bn = os.path.join(self.pred_dir, 'pred_bn')
        self.pred_save_path = os.path.join(self.pred_dir, 'pred_vis')
        if not os.path.exists(self.pred_save_path_bn): os.makedirs(self.pred_save_path_bn)
        if not os.path.exists(self.pred_save_path): os.makedirs(self.pred_save_path)

        testset = dataset.ChangeDataset_BCD_BT(mode='test')
        self.testloader = Data.DataLoader(testset, batch_size=self.args.test_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)
        self.model = Net(backbone='resnet18')
        if self.args.load_from:
            self.model.load_state_dict(torch.load(self.args.load_from), strict=True)
        self.model = self.model.cuda()
    
    def inference(self):
        self.model.eval()

        # calculate Pamrams and FLOPs
        for j, data in enumerate(self.testloader):
            if j == 0:
                image_t1 = data['image1'].cuda()
                image_t2 = data['image2'].cuda()
                break
        model_copy = copy.deepcopy(self.model)
        FLOPs, Params = profile(model_copy, (image_t1, image_t2))
        print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

        test_start = time.time()
        cm_total = np.zeros((2, 2))
        for j, data in enumerate(tqdm(self.testloader)):
            with torch.no_grad():
                image_t1 = data['image1'].cuda()
                image_t2 = data['image2'].cuda()
                change_label = data['label'].cuda()
                image_name = data['image_name']
                change_out = self.model(image_t1, image_t2)

                test_preds = (change_out.cpu().numpy() >=0.5).astype('uint8') * 255
                test_labels = (change_label.cpu().numpy() > 0).astype('uint8') * 255
                cm = metrics.ConfusionMatrix(2, test_preds, test_labels)
                cm_total += cm
                visualization.save_preds_binary_visualization(test_preds, self.pred_save_path_bn, image_name)
                visualization.save_preds_visualization(test_preds, test_labels, self.pred_save_path, image_name)
        precision_total, recall_total, f1_total, iou_total, mIoU, acc, mF1, kappa = metrics.get_score_sum(cm_total)
        Precision = precision_total['precision_1']
        Recall = recall_total['recall_1']
        F1 = f1_total['f1_1']
        IoU = iou_total['iou_1']
        OA = acc
        mIoU = mIoU
        mF1 = mF1
        Kappa = kappa
        cost_time = time.time() - test_start
        print("[test], [cost time %.1fs], [OA %.2f], [F %.2f], [mF %.2f], [IoU %.2f], [mIoU %.2f], [Recall %.2f], [Precision %.2f], [Kappa %.2f]" % \
                        (cost_time, OA * 100, F1 * 100, mF1 * 100, IoU * 100, mIoU * 100, Recall * 100, Precision * 100, Kappa * 100))

        metric_file = os.path.join(self.pred_dir, str(self.args.data_name)+'_'+str(self.args.net_name)+'_metric.txt')
        f = open(metric_file, 'w', encoding='utf-8')
        f.write("{:<20} {:<20} {:<15} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<10}\n".format(
            "Data", "Model", "Infer_time(s)", "Params(Mb)", "FLOPs(Gbps)", "OA", "F", "mF", "IoU", "mIoU", "Recall", "Precision", "Kappa"))
        f.write("{:<20} {:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<12.2f} {:<10.2f}\n".format(
            str(self.args.data_name), str(self.args.net_name), cost_time, Params/1e6, FLOPs/1e9, OA * 100, F1 * 100, mF1 * 100, IoU * 100, mIoU * 100, Recall * 100, Precision * 100, Kappa * 100))
        f.close()
        print('Inference finished.')

if __name__ == "__main__":
    args = Options().parse()
    preder = Preder(args)
    preder.inference()



