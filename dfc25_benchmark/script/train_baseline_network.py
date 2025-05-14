import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from datetime import datetime

import util_func.lovasz_loss as L
from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.MambaHSI import MambaHSI
from model.SiamCRNN import SiamCRNN
from model.UNet import UNet
from util_func.metrics import Evaluator


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """
    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, holdout_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.args = args
        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=4)
        # Create the deep learning model
        if args.model_type == 'UNet':
            self.deep_model = UNet(in_channels=6, out_channels=4)
        elif args.model_type == 'SiamCRNN':
            self.deep_model = SiamCRNN()
        elif args.model_type == 'MUHSI':
            self.deep_model = MambaHSI(in_channels=6, num_classes=4, hidden_dim=128)
        else:
            raise NotImplementedError(f'Sorry, <{args.model_type}> function is not implemented!')
        
        self.deep_model = self.deep_model.cuda()

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.model_type + '_' + now_str)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()

        train_dataset = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, 
                                                         self.args.train_data_name_list, 
                                                         crop_size=self.args.crop_size, 
                                                         max_iters=self.args.max_iters, 
                                                         type='train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, 
                                       shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        
        tqdm_object = tqdm(train_data_loader, total=len(train_data_loader), leave=False, desc='Train Iter'.rjust(10), colour="green")
        for iteration, batch in enumerate(tqdm_object):
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = batch

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long() # shape: BHW=(self.args.train_batch_size, self.args.crop_size, self.args.crop_size)

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue

            if self.args.model_type == 'UNet':
                # For UNet based architecture the concatenation is required
                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # B C*2 HW
                output_clf = self.deep_model(input_data) # B num_class(4) HW
            elif self.args.model_type == 'SiamCRNN':
                # For Siamse based architecture requires two inputs
                # decoupling the task into two subtasks: building localization and damage classification.
                outout_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
            elif self.args.model_type == 'MUHSI':
                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_clf = self.deep_model(input_data)
                # resize the feature to the raw image shape
                output_clf = F.interpolate(output_clf, self.args.crop_size, None, 'bilinear', align_corners=True)
            elif self.args.model_type == 'ChangeMamba':
                outout_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
            else:
                raise NotImplementedError(f'Sorry, <{self.args.model_type}> function is not implemented!')

            self.optim.zero_grad()   

            if self.args.model_type == 'SiamCRNN':
                ce_loss_loc = F.cross_entropy(outout_loc, labels_loc, ignore_index=255)
                lovasz_loss_loc = L.lovasz_softmax(F.softmax(outout_loc, dim=1), labels_loc, ignore=255)

            ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
            # to avoid the inbanlance of label
            # print("output_clf.shape: ", output_clf.shape)
            predict = F.softmax(output_clf, dim=1) # shape: same as output_clf.shape
            lovasz_loss_clf = L.lovasz_softmax(predict, labels_clf, ignore=255)      
            if self.args.model_type == 'UNet' or 'MUHSI':
                final_loss = ce_loss_clf + 0.75 * lovasz_loss_clf
            elif self.args.model_type == 'SiamCRNN':
                final_loss = ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf  + 0.5 * lovasz_loss_loc
            else:
                raise NotImplementedError(f'Sorry, <{self.args.model_type}> function is not implemented!')

            final_loss.backward()
            self.optim.step()
            tqdm_object.set_postfix(train_loss=final_loss.item())

            if (iteration + 1) % self.args.val_internal == 0:
                self.deep_model.eval()
                val_mIoU, final_OA, IoU_of_each_class = self.validation()

                if val_mIoU > best_mIoU:
                    torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model.pth'))
                    best_mIoU = val_mIoU
                    best_round = {
                        'best iter': iteration + 1,
                        'best mIoU': val_mIoU * 100,
                        'best OA': final_OA * 100,
                        'best sub class IoU': IoU_of_each_class * 100
                    }
                self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting validation-----------')
        self.evaluator.reset()
        # 1024 is the raw full resolution of input image
        dataset = MultimodalDamageAssessmentDatset(self.args.holdout_dataset_path, self.args.holdout_data_name_list, 1024, None, 'test')
        holdout_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=0, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(holdout_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()
                if self.args.model_type == 'UNet':
                    input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                    output_clf = self.deep_model(input_data)
                elif self.args.model_type == 'SiamCRNN':
                    _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)
                elif self.args.model_type == 'MUHSI':
                    input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                    output_clf = self.deep_model(input_data)
                    # resize the feature to the raw image shape
                    output_clf = F.interpolate(output_clf, 1024, None, 'bilinear', align_corners=True)
                else:
                    raise NotImplementedError(f'Sorry, <{self.args.model_type}> function is not implemented!')

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()
                self.evaluator.add_batch(labels_clf, output_clf)

        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return mIoU, final_OA, IoU_of_each_class
    

def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")

    parser.add_argument('--dataset', type=str, default='BRIGHT')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--holdout_dataset_path', type=str)
    parser.add_argument('--holdout_data_list_path', type=str)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--val_internal', type=int, default=500)
    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--holdout_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, choices=['UNet', 'SiamCRNN', 'MUHSI'], default='MUHSI')
    parser.add_argument('--model_param_path', type=str)

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)

    args = parser.parse_args()
    print(args)

    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = train_data_name_list

    with open(args.holdout_data_list_path, "r") as f:
        holdout_data_name_list = [data_name.strip() for data_name in f]
    args.holdout_data_name_list = holdout_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
