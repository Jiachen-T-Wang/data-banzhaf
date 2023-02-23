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

import config




def get_ufunc(dataset, model_type, batch_size, lr, verbose):
    if dataset in ['MNIST', 'FMNIST']:
        u_func = lambda a, b, c, d: torch_mnist_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'CIFAR10':
        u_func = lambda a, b, c, d: torch_cifar_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_Cat':
        u_func = lambda a, b, c, d: torch_dogcat_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_CatFeature':
        u_func = lambda a, b, c, d: torch_dogcatFeature_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'FMNIST':
        sys.exit(1)
    elif dataset in ['covertype']:
        u_func = lambda a, b, c, d: binary_data_to_acc(model_type, a, b, c, d)
    elif dataset in config.OpenML_dataset:
        u_func = lambda a, b, c, d: torch_binary_data_to_acc(model_type, a, b, c, d, batch_size=batch_size, lr=lr, verbose=verbose)
    return u_func



def get_weighted_ufunc(dataset, model_type, batch_size, lr, verbose):
    if dataset in ['MNIST', 'FMNIST']:
        u_func = lambda a, b, c, d, w: torch_mnist_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'CIFAR10':
        u_func = lambda a, b, c, d, w: torch_cifar_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_Cat':
        u_func = lambda a, b, c, d, w: torch_dogcat_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'Dog_vs_CatFeature':
        u_func = lambda a, b, c, d, w: torch_dogcatFeature_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    elif dataset == 'FMNIST':
        sys.exit(1)
    elif dataset in ['covertype']:
        u_func = lambda a, b, c, d, w: binary_data_to_acc(model_type, a, b, c, d, w=w)
    elif dataset in config.OpenML_dataset:
        u_func = lambda a, b, c, d, w: torch_binary_data_to_acc(model_type, a, b, c, d, weights=w, batch_size=batch_size, lr=lr, verbose=verbose)
    return u_func



def make_balance_sample_multiclass(data, target, n_data):

    n_class = len(np.unique(target))

    n_data_per_class = int(n_data / n_class)

    selected_ind = np.array([])

    for i in range(n_class):

        index_class = np.where(target == i)[0]

        ind = np.random.choice(index_class, size=n_data_per_class, replace=False)

        selected_ind = np.concatenate([selected_ind, ind])

    selected_ind = selected_ind.astype(int)

    data, target = data[selected_ind], target[selected_ind]

    assert n_data == len(target)

    idxs=np.random.permutation(n_data)
    data, target=data[idxs], target[idxs]

    return data, target



def get_processed_data(dataset, n_data, n_val, flip_ratio):
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    if dataset in config.OpenML_dataset:
        X, y, _, _ = get_data(dataset)
        x_train, y_train = X[:n_data], y[:n_data]
        x_val, y_val = X[n_data:n_data+n_val], y[n_data:n_data+n_val]

        X_mean, X_std= np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

    else:
        x_train, y_train, x_test, y_test = get_data(dataset)
        x_val, y_val = x_test, y_test

        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_data)

    np.random.seed(999)
    n_flip = int(n_data*flip_ratio)

    assert len(y_train.shape)==1
    n_class = len(np.unique(y_train))
    print('# of classes = {}'.format(n_class))
    print('-------')

    if n_class == 2:
        y_train[:n_flip] = 1 - y_train[:n_flip]
    else:
        y_train[:n_flip] = np.array( [ np.random.choice( np.setdiff1d(np.arange(n_class), [y_train[i]]) ) for i in range(n_flip) ] )

    return x_train, y_train, x_val, y_val



def get_data(dataset):

    if dataset in ['covertype']+config.OpenML_dataset:
        x_train, y_train, x_test, y_test = get_minidata(dataset)
    elif dataset == 'MNIST':
        x_train, y_train, x_test, y_test = get_mnist()
    elif dataset == 'CIFAR10':
        x_train, y_train, x_test, y_test = get_cifar()
    elif dataset == 'Dog_vs_Cat':
        x_train, y_train, x_test, y_test = get_dogcat()
    elif dataset == 'Dog_vs_CatFeature':
        x_train, y_train, x_test, y_test = get_dogcatFeature()
    elif dataset == 'FMNIST':
        x_train, y_train, x_test, y_test = get_fmnist()
    else:
        sys.exit(1)

    return x_train, y_train, x_test, y_test


def make_balance_sample(data, target):
    p = np.mean(target)
    if p < 0.5:
        minor_class=1
    else:
        minor_class=0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target



def get_minidata(dataset):

    open_ml_path = 'OpenML_datasets/'

    np.random.seed(999)

    if dataset == 'covertype':
        x_train, y_train, x_test, y_test = pickle.load( open('covertype_200.dataset', 'rb') )

    elif dataset == 'fraud':
        data_dict=pickle.load(open(open_ml_path+'CreditCardFraudDetection_42397.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'apsfail':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'APSFailure_41138.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'click':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'Click_prediction_small_1218.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'phoneme':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'phoneme_1489.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'wind':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'wind_847.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'pol':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'creditcard':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'cpu':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'cpu_act_761.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'vehicle':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == '2dplanes':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    else:
        print('No such dataset!')
        sys.exit(1)


    if dataset not in ['covertype']:
        idxs=np.random.permutation(len(data))
        data, target=data[idxs], target[idxs]
        return data, target, None, None
    else:
        return x_train, y_train, x_test, y_test



def get_mnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.MNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.MNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test



def get_fmnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.FashionMNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.FashionMNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test




def get_cifar():
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train) 
    testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def get_dogcat():

    x_train, y_train, x_test, y_test = get_cifar()

    dogcat_ind = np.where(np.logical_or(y_train==3, y_train==5))[0]
    x_train, y_train = x_train[dogcat_ind], y_train[dogcat_ind]
    y_train[y_train==3] = 0
    y_train[y_train==5] = 1

    dogcat_ind = np.where(np.logical_or(y_test==3, y_test==5))[0]
    x_test, y_test = x_test[dogcat_ind], y_test[dogcat_ind]
    y_test[y_test==3] = 0
    y_test[y_test==5] = 1

    return x_train, y_train, x_test, y_test


def get_dogcatFeature():

    # x_train, y_train, x_test, y_test = pickle.load( open('dogvscat_feature.dataset', 'rb') )
    x_train, y_train, x_test, y_test = pickle.load( open('result/DogCatImageNetPretrain.data', 'rb') )

    return x_train, y_train, x_test, y_test







