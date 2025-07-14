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
import random
import numpy as np
from tqdm import tqdm
import torch.utils.data as Data
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import metrics
from utils.utils import AverageMeter, PolyScheduler, get_logger, seed_torch
from datasets import ST_LEVIR1024 as dataset
from models.MyNet import changeNet as Net
import argparse
import glob
import datetime

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="ST_LEVIR1024")
        parser.add_argument("--net_name", type=str, default="MyNet")
        parser.add_argument("--image_size", type=int, default=1024)
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=8)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=5e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--pretrain_from", type=str, help='train from a checkpoint')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

class Trainer:
    def __init__(self, args):
        self.args = args
        self.log_dir = os.path.join(working_path, 'logs', self.args.data_name + '_' + self.args.net_name, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.logger = get_logger(os.path.join(self.log_dir, 'train_'+str(self.args.data_name)+'_'+str(self.args.net_name)+'.log'))
        self.logger.info("Config Information: {}".format(self.args))
        
        trainset = dataset.ChangeDataset_BCD_ST(mode='st_train')
        valset = dataset.ChangeDataset_BCD_BT(mode='test')
        self.trainloader = Data.DataLoader(trainset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=trainset.change_synthesis, num_workers=8, drop_last=False)
        self.valloader = Data.DataLoader(valset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=8, drop_last=False)
        self.model = Net(backbone='resnet18')
        if self.args.pretrain_from:
            self.model.load_state_dict(torch.load(self.args.pretrain_from), strict=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, dampening=0, weight_decay=1e-3, nesterov=False)
        self.scheduler = PolyScheduler(self.optimizer, self.args.epochs*len(self.trainloader), power=0.9)
        self.model = self.model.cuda()
        self.criterion_bn = nn.BCELoss().cuda()
        self.best_metrics = {
            'best_epoch': 0,
            'best_OA': 0.0,
            'best_F': 0.0,
            'best_mF': 0.0,
            'best_IoU': 0.0,
            'best_mIoU': 0.0,
            'best_Recall': 0.0,
            'best_Precision': 0.0,
            'best_Kappa': 0.0,
            'best_val_loss_change': 10.0}

    def training(self, epoch):
        torch.cuda.empty_cache()
        self.model.train()
        curr_epoch = epoch + 1
        tbar = tqdm(self.trainloader)
        train_loss_total = AverageMeter()
        train_loss_change = AverageMeter()
        train_loss_metric = AverageMeter()
        all_iters = float(self.args.epochs * len(self.trainloader))
        curr_iter = curr_epoch * len(self.trainloader)

        for i, data in enumerate(tbar):
            running_iter = curr_iter + i + 1
            image_t1 = data['image1'].cuda()
            image_t2 = data['image2'].cuda()
            change_label = data['label'].cuda()
            self.optimizer.zero_grad()
            change_out = self.model(image_t1, image_t2)
            loss_change = self.criterion_bn(change_out, change_label)
            loss_metric = 0.0
            loss_total = loss_change
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss_change.update(loss_change.item())
            train_loss_metric.update(loss_metric)
            train_loss_total.update(loss_total.item())
            
            tbar.set_description("[Training], [epoch %d/%d], [iter %d/%d], [lr: %f], [loss_change %.4f], [loss_metric %.4f], [loss_total %.4f]" %
        (curr_epoch, self.args.epochs, i + 1, len(self.trainloader), self.optimizer.param_groups[0]["lr"], train_loss_change.avg, train_loss_metric.avg, train_loss_total.avg))

            self.writer.add_scalar('train_loss_change', train_loss_change.avg, running_iter)
            self.writer.add_scalar('train_loss_metric', train_loss_metric.avg, running_iter)
            self.writer.add_scalar('train_loss_total', train_loss_total.avg, running_iter)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)
        
        self.logger.info("[Training], [epoch %d/%d], [iter %d/%d], [lr: %f], [loss_change %.4f], [loss_metric %.4f], [loss_total %.4f]" %
        (curr_epoch, self.args.epochs, i + 1, len(self.trainloader), self.optimizer.param_groups[0]["lr"], train_loss_change.avg, train_loss_metric.avg, train_loss_total.avg))

    def validation(self, epoch):
        torch.cuda.empty_cache()
        self.model.eval()
        curr_epoch = epoch + 1
        tbar = tqdm(self.valloader)
        cm_total = np.zeros((2, 2))
        val_loss_change = AverageMeter()

        for j, data in enumerate(tbar):
            with torch.no_grad():
                image_t1 = data['image1'].cuda()
                image_t2 = data['image2'].cuda()
                change_label = data['label'].cuda()
                change_out = self.model(image_t1, image_t2)
                loss_change = self.criterion_bn(change_out, change_label)
                val_loss_change.update(loss_change.item())

                val_preds = (change_out.cpu().numpy() >=0.5).astype('uint8') * 255
                val_labels = (change_label.cpu().numpy() > 0).astype('uint8') * 255
                cm = metrics.ConfusionMatrix(2, val_preds, val_labels)
                cm_total += cm
        precision_total, recall_total, f1_total, iou_total, mIoU, acc, mF1, kappa = metrics.get_score_sum(cm_total)
        Precision = precision_total['precision_1']
        Recall = recall_total['recall_1']
        F1 = f1_total['f1_1']
        IoU = iou_total['iou_1']
        OA = acc
        mIoU = mIoU
        mF1 = mF1
        Kappa = kappa
        self.logger.info("[Validate], [epoch %d/%d], [OA %.2f], [F %.2f], [mF %.2f], [IoU %.2f], [mIoU %.2f], [Recall %.2f], [Precision %.2f], [Kappa %.2f], [val_loss_change %.4f]" % \
                             (curr_epoch, self.args.epochs, OA * 100, F1 * 100, mF1 * 100, IoU * 100, mIoU * 100, Recall * 100, Precision * 100, Kappa * 100, val_loss_change.avg))

        if F1 >= self.best_metrics['best_F']:
            model_path = "checkpoints/%s" % (self.args.data_name + '_' + self.args.net_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            else:
                files = glob.glob(os.path.join(model_path, '*'))
                for f in files:
                    os.remove(f)
            torch.save(self.model.state_dict(), "%s/Val_best_%s_%s_epoch%i_OA%.2f_F%.2f_mF%.2f_IoU%.2f_mIoU%.2f_Recall%.2f_Precision%.2f_Kappa%.2f_loss%.4f.pth" %
                (model_path, self.args.data_name, self.args.net_name, curr_epoch, OA * 100, F1 * 100, mF1 * 100, IoU * 100, mIoU * 100, Recall * 100, Precision * 100, Kappa * 100, val_loss_change.avg))
            self.best_metrics.update({
                'best_epoch': curr_epoch,
                'best_OA': OA,
                'best_F': F1,
                'best_mF': mF1,
                'best_IoU': IoU,
                'best_mIoU': mIoU,
                'best_Recall': Recall,
                'best_Precision': Precision,
                'best_Kappa': Kappa,
                'best_val_loss_change': val_loss_change.avg})

        self.writer.add_scalar('val_OA', OA, curr_epoch)
        self.writer.add_scalar('val_F1', F1, curr_epoch)
        self.writer.add_scalar('val_mF1', mF1, curr_epoch)
        self.writer.add_scalar('val_IoU', IoU, curr_epoch)
        self.writer.add_scalar('val_mIoU', mIoU, curr_epoch)
        self.writer.add_scalar('val_Recall', Recall, curr_epoch)
        self.writer.add_scalar('val_Precision', Precision, curr_epoch)
        self.writer.add_scalar('val_Kappa', Kappa, curr_epoch)
        self.writer.add_scalar('val_loss_change', val_loss_change.avg, curr_epoch)

if __name__ == "__main__":
    #set random seed
    seed_torch(seed = 42)
    working_path = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args = Options().parse()
    trainer = Trainer(args)
    train_begin_time = time.time()
    for epoch in range(args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
        trainer.logger.info('[Val_best], [epoch %d/%d], [OA %.2f], [F %.2f], [mF %.2f], [IoU %.2f], [mIoU %.2f], [Recall %.2f], [Precision %.2f], [Kappa %.2f], [val_loss_change %.4f]\n' \
            % (trainer.best_metrics['best_epoch'], trainer.args.epochs, trainer.best_metrics['best_OA'] * 100, trainer.best_metrics['best_F'] * 100, trainer.best_metrics['best_mF'] * 100, 
               trainer.best_metrics['best_IoU'] * 100, trainer.best_metrics['best_mIoU'] * 100, trainer.best_metrics['best_Recall'] * 100, trainer.best_metrics['best_Precision'] * 100, 
               trainer.best_metrics['best_Kappa'] * 100, trainer.best_metrics['best_val_loss_change']))
    trainer.writer.close()
    train_end_time = time.time()
    trainer.logger.info('Training cost time: %.2f hours' % ((train_end_time - train_begin_time) / 3600))
    trainer.logger.info('Training finished.')



