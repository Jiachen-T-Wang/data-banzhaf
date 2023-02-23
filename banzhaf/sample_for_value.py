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

from helper import *
from utility_func import *
from prepare_data import *
import config

import pdb

# sbatch sample_mnist.sh Banzhaf_GT Logistic 5 5000 0.11 8 1


import argparse

parser = argparse.ArgumentParser('')


# python sample_for_value.py --dataset CIFAR10 --value_type Shapley_Perm --model_type SmallCNN --n_data 5 --n_val 5000 --n_sample 5 --n_repeat 5 --batch_size 32 --random_state 1 --flip_ratio 0

# python sample_for_value.py --dataset $1 --value_type $2 --model_type $3 --n_data $4 --n_val $5 --n_sample $6 --n_repeat $7 --batch_size $8 --random_state $9 --flip_ratio ${10}

# sbatch sample_for_value.sh CIFAR10 Shapley_Perm SmallCNN 2000 5000 4000 5 32 1 0

# python sample_for_value.py --dataset Dog_vs_CatFeature --value_type Banzhaf_GT --model_type MLP --n_data 2000 --n_val 2000 --n_sample 2000 --n_repeat 5 --batch_size 32 --random_state 1 --flip_ratio 0

# sbatch sample_for_value.sh Dog_vs_CatFeature Shapley_Perm Logistic 200 2000 5000 5 32 1 0


# python sample_for_value.py --dataset covertype --value_type Banzhaf_GT --model_type Logistic --n_data 200 --n_val 2000 --n_sample 10000 --n_repeat 1 --flip_ratio 0.1 --random_state 1

# python sample_for_value.py --dataset covertype --value_type LOO --model_type Logistic --n_data 200 --n_val 2000 --n_sample 10000 --n_repeat 1 --flip_ratio 0.1 --random_state 1

# python sample_for_value.py --dataset covertype --value_type KNN --model_type Logistic --n_data 200 --n_val 2000 --n_sample 10000 --n_repeat 1 --flip_ratio 0.1 --random_state 1


# python sample_for_value.py --dataset  --value_type $2 --model_type $3 --n_data $4 --n_val $5 --n_sample $6 --n_repeat $7 --batch_size $8 --lr $9 --random_state ${10} --flip_ratio ${11}

# sbatch sample_for_value.sh covertype LOO 


# python sample_for_value.py --dataset Dog_vs_CatFeature --value_type Banzhaf_GT --model_type MLP --n_data 2000 --n_val 2000 --n_sample 3 --n_repeat 5 --batch_size 128 --random_state 1 --flip_ratio 0.1 --debug


# python sample_for_value.py --dataset MNIST --value_type Banzhaf_GT --model_type SmallCNN --n_data 500 --n_val 2000 --n_repeat 5 --n_sample 3 --batch_size 128 --flip_ratio 0.1 --random_state 1 --lr 0.001 --debug


# python sample_for_value.py --dataset FMNIST --value_type Banzhaf_GT --model_type SmallCNN --n_data 500 --n_val 2000 --n_repeat 5 --n_sample 3 --batch_size 128 --flip_ratio 0.1 --random_state 1 --lr 0.001 --debug


# python sample_for_value.py --dataset fraud --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 2000 --n_repeat 5 --n_sample 3 --batch_size 32 --flip_ratio 0.1 --random_state 1 --lr 0.01 --debug


# python sample_for_value.py --dataset CIFAR10 --value_type Banzhaf_GT --model_type SmallCNN --n_data 2000 --n_val 2000 --n_repeat 5 --n_sample 3 --batch_size 128 --flip_ratio 0.1 --random_state 1 --lr 0.001 --debug


# python sample_for_value.py --dataset CIFAR10 --value_type Banzhaf_GT --model_type SmallCNN --n_data 5000 --n_val 2000 --n_repeat 5 --n_sample 3 --batch_size 128 --flip_ratio 0.1 --random_state 1 --lr 0.001 --debug



parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_repeat', type=int, default=1)
parser.add_argument('--n_sample', type=int)
parser.add_argument('--random_state', type=int, default=1)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
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
flip_ratio = args.flip_ratio
batch_size = args.batch_size
lr = args.lr


verbose = 0
if args.debug:
    verbose = 1


save_dir = 'result/'
big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset


if dataset in big_dataset+OpenML_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed{}.data'.format(
        value_type, dataset, model_type, n_data, n_val, n_sample, batch_size, lr, n_repeat, flip_ratio, random_state)
else:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data'.format(
        value_type, dataset, model_type, n_data, n_val, n_sample, flip_ratio)


u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose)
utility_func_mult = lambda a, b, c, d: sample_utility_multiple(a, b, c, d, u_func, n_repeat)

x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)


if args.debug: 
    print(y_train[:20], y_val[:20])
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))


utility_func_args = (x_train, y_train, x_val, y_val)

n_class = len(np.unique(y_val))
sv_baseline = 1.0/n_class


if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def process_yfeature(y_feature):
    y_feature = np.array(y_feature)
    if n_repeat==1:
        y_feature = y_feature.reshape(-1)
    return y_feature


if value_type == 'Banzhaf_MC':
    n_sample_per_data = int(n_sample / n_data)
    save_arg = {}
    for target_ind in range(n_data):
        utility_set_tgt = sample_utility_banzhaf_mc(n_sample_per_data, utility_func_mult, utility_func_args, target_ind)
        save_arg[target_ind] = utility_set_tgt

elif value_type == 'Shapley_Perm':
    n_perm = int(n_sample / n_data)
    X_feature_test, y_feature_test = sample_utility_shapley_perm(n_perm, utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'Banzhaf_GT':
    X_feature_test, y_feature_test = sample_utility_banzhaf_gt(n_sample, utility_func_mult, utility_func_args, dummy=True)
    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'Shapley_GT':
    X_feature_test, y_feature_test = sample_utility_shapley_gt(n_sample, utility_func_mult, utility_func_args)


    y_feature_test = process_yfeature(y_feature_test)
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test}

elif value_type == 'LOO':
    X_feature_test, y_feature_test, u_total = sample_utility_loo(utility_func_mult, utility_func_args)
    y_feature_test = process_yfeature(y_feature_test)
    u_total = np.array(u_total)
    if n_repeat==1:
        u_total = u_total[0]
    save_arg = {'X_feature': X_feature_test, 'y_feature': y_feature_test, 'u_total': u_total}

elif value_type == 'KNN':
    sv = knn_shapley(x_train, y_train, x_val, y_val, K=10)
    # sv = knn_shapley(x_train, y_train, x_val[:200], y_val[:200], K=10)
    save_arg = {'knn': sv}

save_arg['sv_baseline'] = sv_baseline
save_arg['n_data'] = n_data

pickle.dump(save_arg, open(save_name, 'wb'))



