import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
import random
import pdb

from helper import *
from utility_func import *
from prepare_data import *
import config

# sbatch sample_mnist.sh Banzhaf_GT Logistic 5 5000 0.11 8 1

# python check_weighted_acc.py --dataset covertype --value_type Shapley_Perm --model_type Logistic --n_data 200 --n_val 2000 --n_sample 10000 --flip_ratio 0.1 --random_state 1 --sigma 0

# python applications.py --task mislabel_detect --dataset covertype --value_type Shapley_Perm --model_type Logistic --n_data 200 --n_val 2000 --n_sample 10000 --flip_ratio 0.1 --random_state 1 --sigma 0

# python applications.py --task mislabel_detect --dataset CIFAR10 --value_type Banzhaf_GT --model_type SmallCNN --n_data 1000 --n_val 5000 --n_repeat 5 --n_sample 5000 --batch_size 128 --flip_ratio 0.1 --random_state 1 --sigma 0

# python applications.py --task weighted_acc --dataset Dog_vs_CatFeature --value_type Shapley_Perm --model_type MLP --n_data 2000 --n_val 2000 --n_repeat 5 --n_sample 100000 --batch_size 128 --flip_ratio 0.1 --random_state 1

# python applications.py --task mislabel_detect --dataset fraud --value_type Shapley_Perm --model_type MLP --n_data 200 --n_val 2000 --n_repeat 5 --n_sample 10000 --batch_size 32 --lr 1e-2 --flip_ratio 0.1 --random_state 1


import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--n_sample', type=int)
parser.add_argument('--random_state', type=int)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--debug', action='store_true')


args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
model_type = args.model_type
n_data = args.n_data
n_val = args.n_val
n_repeat = args.n_repeat
n_sample = args.n_sample
random_state = args.random_state
flip_ratio = float(args.flip_ratio) * 1.0
batch_size = args.batch_size
lr = args.lr
a, b = args.alpha, args.beta
task = args.task


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

verbose = 0
if args.debug:
  verbose = 1

u_func = get_weighted_ufunc(dataset, model_type, batch_size, lr, verbose)

x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)


if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


if value_type != 'Uniform':

  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley'] and dataset in big_dataset:
    args.n_sample *= 10
  value_args = load_value_args(value_type, args)
  value_args['n_data'] = n_data

else:
  value_args = {}
  value_args['n_data'] = n_data


data_lst = []

for i in range(5):

  v_args = copy.deepcopy(value_args)

  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley']:

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature'][:, i]
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature'] + np.random.normal(scale=args.sigma, size=n_sample) , a_min=0, a_max=1)

  elif value_type == 'LOO':

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature'][:, i]
      v_args['u_total'] = value_args['u_total'][i]
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature']+np.random.normal(scale=args.sigma, size=len(value_args['y_feature'])), a_min=0, a_max=1)
      v_args['u_total'] = np.clip( value_args['u_total']+np.random.normal(scale=args.sigma), a_min=0, a_max=1)

  sv = compute_value(value_type, v_args)

  if args.debug: pdb.set_trace()

  if task=='weighted_acc':

    sv = normalize(sv) if value_type!='Uniform' else sv

    if dataset in big_dataset or dataset in OpenML_dataset :
      acc_lst = []
      for j in range(5):
        acc_lst.append( u_func(x_train, y_train, x_val, y_val, sv) )
      acc = np.mean(acc_lst)
    else:
      acc = u_func(x_train, y_train, x_val, y_val, sv)
    print('round {}, acc={}'.format(i, acc))
    data_lst.append( acc )

  elif task=='mislabel_detect':
    # acc1, acc2 = kmeans_aucroc(sv, cluster=False), kmeans_aucroc(sv, cluster=True)
    acc1, acc2 = kmeans_f1score(sv, cluster=False), kmeans_f1score(sv, cluster=True)
    data_lst.append( [acc1, acc2] )


if task=='mislabel_detect':
  
  data_lst = np.array(data_lst)
  acc_nocluster, std_nocluster = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)
  acc_cluster, std_cluster = np.round( np.mean(data_lst[:, 1]), 3), np.round( np.std(data_lst[:, 1]), 3)

  if value_type == 'BetaShapley':
    print('*** {}_{}_{} {} ({}) {} ({}) ***'.format(value_type, b, a, acc_nocluster, std_nocluster, acc_cluster, std_cluster ))
  elif value_type in ['FixedCard_MC', 'FixedCard_MSR', 'FixedCard_MSRPerm']:
    print('*** {} card={} {} ({}) {} ({}) ***'.format(value_type, args.card, acc_nocluster, std_nocluster, acc_cluster, std_cluster ))
  else:
    print('*** {} {} ({}) {} ({}) ***'.format(value_type, acc_nocluster, std_nocluster, acc_cluster, std_cluster ))
    



