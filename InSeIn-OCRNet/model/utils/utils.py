# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

invalid_pairs_cityscape = [(6, 11), (15, 6), (6, 0), (6, 12), (1, 14), (0, 10), (16, 1), (10, 4), (10, 18), (7, 9), (15, 5), (1, 10), (15, 11), (15, 0), (3, 8), (5, 17), (1, 15), (15, 12), (12, 9), (10, 11), (10, 0), (14, 15), (9, 14), (11, 6), (10, 12), (3, 17), (12, 7), (15, 1), (9, 10), (4, 17), (0, 7), (0, 3), (9, 15), (4, 16), (14, 13), (8, 16), (7, 15), (9, 13), (18, 6), (6, 17), (3, 1), (13, 16), (1, 4), (16, 9), (18, 17), (7, 13), (1, 18), (12, 15), (16, 7), (14, 4), (15, 17), (12, 10), (14, 18), (0, 15), (0, 12), (3, 14), (1, 12), (10, 17), (4, 14), (6, 1), (3, 10), (9, 18), (0, 14), (14, 5), (16, 14), (14, 11), (16, 3), (4, 10), (7, 4), (14, 12), (17, 14), (7, 18), (17, 3), (1, 13), (16, 10), (3, 16), (8, 15), (16, 15), (17, 10), (6, 14), (10, 2), (14, 2), (9, 12), (10, 1), (7, 11), (0, 8), (12, 6), (7, 12), (3, 9), (17, 16), (2, 14), (15, 14), (0, 4), (2, 3), (15, 3), (0, 18), (0, 6), (12, 5), (6, 16), (5, 14), (15, 10), (10, 14), (10, 3), (14, 8), (1, 6), (18, 16), (2, 15), (17, 9), (1, 17), (17, 7), (15, 16), (16, 4), (14, 17), (16, 18), (6, 7), (0, 2), (10, 16), (11, 10), (3, 15), (9, 6), (15, 9), (13, 4), (9, 17), (4, 15), (15, 7), (4, 12), (8, 12), (16, 0), (10, 9), (16, 12), (7, 17), (17, 15), (18, 14), (14, 1), (16, 13), (6, 15), (18, 10), (12, 17), (18, 15), (12, 16), (0, 17), (2, 0), (14, 3), (6, 13), (2, 12), (7, 1), (0, 16), (14, 10), (3, 6), (1, 16), (5, 12), (10, 15), (15, 13), (0, 9), (16, 6), (14, 16), (17, 4), (3, 11), (7, 3), (3, 12), (10, 13), (16, 17), (17, 6), (11, 12), (16, 5), (6, 4), (6, 18), (1, 7), (16, 11), (9, 16), (13, 6), (12, 14), (12, 3), (14, 9), (17, 11), (17, 12), (7, 16), (15, 4), (15, 18), (16, 2)]
invalid_pairs_acdc = [(16, 4), (14, 18), (4, 9), (6, 11), (9, 1), (13, 17), (17, 7), (2, 0), (4, 16), (12, 10), (3, 10), (14, 11), (11, 10), (11, 6), (6, 13), (15, 9), (8, 12), (13, 12), (4, 12), (10, 15), (0, 5), (6, 17), (9, 14), (9, 3), (16, 9), (15, 7), (9, 2), (14, 13), (17, 10), (10, 18), (16, 7), (17, 6), (7, 10), (13, 4), (14, 17), (9, 16), (1, 11), (12, 5), (12, 15), (3, 15), (10, 11), (1, 14), (11, 15), (6, 1), (15, 6), (14, 12), (12, 18), (3, 18), (9, 12), (0, 14), (0, 3), (1, 13), (11, 18), (18, 10), (18, 6), (0, 2), (5, 14), (13, 9), (14, 1), (1, 16), (10, 13), (17, 15), (7, 5), (14, 4), (13, 16), (7, 15), (10, 17), (12, 11), (13, 7), (0, 17), (0, 16), (6, 3), (6, 14), (17, 18), (5, 17), (7, 18), (5, 16), (12, 14), (1, 12), (3, 14), (6, 9), (0, 8), (10, 12), (3, 13), (0, 12), (6, 16), (12, 17), (3, 17), (5, 12), (16, 15), (17, 11), (1, 4), (7, 11), (14, 9), (15, 18), (13, 6), (4, 10), (18, 15), (10, 1), (4, 6), (14, 16), (10, 4), (14, 7), (0, 4), (6, 12), (7, 14), (9, 7), (10, 0), (12, 8), (3, 12), (11, 5), (7, 13), (15, 11), (7, 17), (1, 2), (6, 4), (10, 3), (12, 1), (3, 1), (6, 0), (18, 11), (12, 4), (12, 0), (17, 5), (14, 6), (10, 9), (8, 15), (18, 14), (9, 6), (7, 8), (4, 15), (9, 10), (7, 12), (18, 13), (8, 18), (12, 3), (17, 1), (18, 17), (7, 1), (11, 14), (7, 4), (15, 5), (12, 9), (3, 9), (6, 7), (16, 5), (1, 6), (8, 11), (12, 16), (3, 16), (12, 7), (4, 11), (13, 10), (11, 17), (15, 1), (0, 10), (17, 3), (17, 14), (0, 6), (16, 18), (9, 15), (7, 3), (17, 9), (9, 18), (2, 1), (4, 13), (7, 9), (11, 12), (8, 17), (17, 16), (4, 17), (7, 16), (16, 11), (15, 14), (15, 3), (12, 6), (3, 6), (1, 15), (16, 14), (16, 3), (9, 11), (13, 5), (15, 13), (13, 15), (18, 3), (17, 12), (2, 14), (2, 3), (15, 17), (0, 15), (16, 13), (15, 16), (1, 18), (13, 18), (16, 17), (18, 9), (2, 9), (0, 18), (4, 1), (9, 13), (18, 16), (2, 17), (15, 8), (18, 7), (2, 16), (9, 17), (8, 0), (6, 15), (15, 12), (5, 18), (4, 0), (1, 10), (13, 11), (16, 12), (6, 18), (14, 5), (14, 15), (2, 12), (11, 7), (8, 14), (15, 4), (4, 14), (5, 11), (15, 0)]

def _get_relu(name: str) -> nn.Sequential:
    container = nn.Sequential()
    relu = nn.ReLU()
    container.add_module(f'{name}_relu', relu)

    return container


def _max_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


def _avg_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        loss = self.loss(outputs, labels)
        return torch.unsqueeze(loss, 0), outputs


class MaxPoolMatMulLayer(nn.Module):

      def __init__(self, kernel_size=3, stride=1, padding=1):

          super(MaxPoolMatMulLayer, self).__init__()
          self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
          #torch.set_default_dtype(torch.float16)

      def forward(self, x, initial_input):

          # Apply MaxPool2D
          pooled_output = self.pool(x)
          matmul_result = pooled_output*initial_input

          # To combine with spatial dimensions, we might need to reshape or recombine
          return matmul_result

class MaxPoolMatMulStack(nn.Module):

      def __init__(self, num_layers=128, kernel_size=3, stride=1, padding=1):

          super(MaxPoolMatMulStack, self).__init__()
          self.num_layers = num_layers
          module_list = [MaxPoolMatMulLayer(kernel_size, stride, padding) for _ in range(num_layers)]
          self.layers = nn.ModuleList(module_list)
          #torch.set_default_dtype(torch.float16)


      def forward(self, x):

          initial_input = x.clone()
          current_tensor = -x

          current_tensor[:, :, [0, -1], :] = 1
          current_tensor[:, :, :, [0, -1]] = 1

          # Apply the sequence of MaxPool and MatMul layers
          for layer in self.layers:
              current_tensor = layer(current_tensor, initial_input)

          current_tensor0 = F.normalize(current_tensor)
          current_tensor1 = current_tensor0 * initial_input

          return current_tensor1


class PhysicsFormer(nn.Module):

      def __init__(self,invalid_pair_list=None):

          super(PhysicsFormer,self).__init__()
          self.T=128
          self.invalid_pair_list = invalid_pair_list
          self.maxpooling = MaxPoolMatMulStack(num_layers=self.T, kernel_size=3, stride=1, padding=1)
          self.relu = nn.ReLU(inplace=False)
          self.empty_norm = torch.tensor([0,],dtype=torch.float16)

      def final_operation(self, original,mode='opening'):

          if mode == 'opening':

             opened = self.maxpooling(original)

             #     torch.save(opened,f'/cluster/work/cvl/shbasu/opened_{idx}.pt')
             subtracted = original- opened
             #torch.save(subtracted,f'/cluster/work/cvl/shbasu/subtracted_{idx}.pt')

          l1_norm = torch.norm(subtracted, p=1)
          return l1_norm

      def forward(self, input):

          final_norm = 0
          pairs = []

          logits_upscaled = f.interpolate(input,size=(256,256),mode='bilinear',align_corners=False)

          for idx, pair in enumerate(self.invalid_pair_list):

              concatenated_tensor = torch.cat((logits_upscaled[:,pair[0]:pair[0]+1,::], logits_upscaled[:,pair[1]:pair[1]+1,::]), dim=1)
              softmax = torch.softmax(concatenated_tensor,dim=1)
              #torch.save(softmax,f'/cluster/work/cvl/shbasu/softmax_{idx}.pt')
              difference = softmax[:, 0:1, :, :] - softmax[:, 1:2, :, :]
              relu = self.relu(difference)
              #torch.save(relu,f'/cluster/work/cvl/shbasu/relu_{idx}.pt')
              pairs.append(relu)

          final_concatenated = torch.cat(pairs, dim=1)
          del pairs
          norm_opened_1 = self.final_operation(final_concatenated)
          if torch.any(norm_opened_1.isnan()) or norm_opened_1.numel()==0:
              final_norm = self.empty_norm.cuda()
          else:
              norm_opened_1 = torch.clamp(norm_opened_1,max=65504)
              final_norm = norm_opened_1.half()

          return final_norm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    #root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not os.path.exists(cfg.OUTPUT_DIR):
        #print('=> creating {}'.format(cfg.OUTPUT_DIR))
        os.mkdir(cfg.OUTPUT_DIR)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+ cfg_name


    #print('=> creating {}'.format(final_output_dir))
    try:

        if not os.path.exists(cfg.OUTPUT_DIR+'/'+'/'+dataset):
            os.mkdir(cfg.OUTPUT_DIR+'/'+'/'+dataset)
        if not os.path.exists(cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+cfg_name):
            os.mkdir(cfg.OUTPUT_DIR+'/'+'/'+dataset+'/'+cfg_name)

    except FileExistsError:

        logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_out = logging.getLogger(__name__)
        logger_out.error('final_output direcotry exists')


    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir+'/'+log_file
    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = cfg.LOG_DIR+dataset+'/'+model+'/'+cfg_name+'_'+time_str
    #print('=> creating {}'.format(tensorboard_log_dir))
    try:

        if not os.path.exists(cfg.LOG_DIR+'/'+dataset):
            os.mkdir(cfg.LOG_DIR+'/'+dataset)
        if not os.path.exists(cfg.LOG_DIR+'/'+dataset+'/'+model):
            os.mkdir(cfg.LOG_DIR+'/'+dataset+'/'+model)
        if not os.path.exists(cfg.LOG_DIR+'/'+dataset+'/'+model+'/'+cfg_name+'_'+time_str):
            os.mkdir(cfg.LOG_DIR+'/'+dataset+'/'+model+'/'+cfg_name+'_'+time_str)

    except FileExistsError:
        logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_log = logging.getLogger(__name__)
        logger_log.error('final_log direcotry exists')

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
