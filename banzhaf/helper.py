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

import scipy
from scipy.special import beta, comb
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score



from utility_func import *

import config


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'



def kmeans_f1score(value_array, cluster=True):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)



def kmeans_aucroc(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return roc_auc_score(true, pred)


def kmeans_aupr(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return average_precision_score(true, pred)




"""
def kmeans_f1score(value_array):

  n_data = len(value_array)

  X = value_array.reshape(-1, 1)
  kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
  min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  pred = np.zeros(n_data)
  pred[value_array < min_cluster] = 1
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)
"""


"""
def load_value_args(value_type, args):

  if args.dataset == 'Dog_vs_CatFeature':

    if args.value_type == 'LeastCore':
      save_name = save_dir + 'Banzhaf_GT_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'
      value_arg = pickle.load( open(save_name, 'rb') )

      for i, x in enumerate(value_arg['X_feature']):
        value_arg['X_feature'][i] = x[x<args.n_data]
    else:
      save_name = save_dir + '{}_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'.format( value_type )
      value_arg = pickle.load( open(save_name, 'rb') )

    value_arg['y_feature'] = np.mean( np.array(value_arg['y_feature']), axis=1 )

  else:

    save_name = save_dir + '{}_{}_Logistic_Ndata200_Nval2000_Nsample2000_FR0.1_Seed{}.data'.format(value_type, args.dataset, args.random_state)
    value_arg = pickle.load( open(save_name, 'rb') )
    if 'y_feature' in value_arg.keys():
      value_arg['y_feature'] = np.array(value_arg['y_feature'])

  return value_arg
"""


def load_value_args(value_type, args):
  if value_type == 'BetaShapley':
    base_value_type = 'Shapley_Perm'
  else:
    base_value_type = value_type

  if args.dataset in big_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio)
  elif args.dataset in OpenML_dataset:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_Seed0.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio)
  else:
    save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data'.format(
        base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.flip_ratio)

  value_arg = pickle.load( open(save_name, 'rb') )

  if 'y_feature' in value_arg.keys():
    value_arg['y_feature'] = np.array(value_arg['y_feature'])

  if value_type == 'BetaShapley':
    value_arg['alpha'] = args.alpha
    value_arg['beta'] = args.beta

  return value_arg


# args: a dictionary
def compute_value(value_type, args):
  if value_type == 'Shapley_Perm':
    sv = shapley_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], v0=args['sv_baseline'])
  elif value_type == 'BetaShapley':
    sv = betasv_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], args['alpha'], args['beta'], v0=args['sv_baseline'])
  elif value_type == 'Banzhaf_GT':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=True)
  elif value_type == 'LOO':
    sv = compute_loo(args['y_feature'], args['u_total'])
  elif value_type == 'KNN':
    sv = args['knn']
  elif value_type == 'Uniform':
    sv = np.ones(args['n_data'])
  elif value_type == 'Shapley_GT':
    sv = shapley_grouptest_from_data(args['X_feature'], args['y_feature'], args['n_data'])
  elif value_type == 'LeastCore':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=False)
  return sv



def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val


def shapley_permsampling_from_data(X_feature, y_feature, n_data, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += (with_score-without_score)
  
  return sv_vector / n_perm


def beta_constant(a, b):
    '''
    the second argument (b; beta) should be integer in this function
    '''
    beta_fct_value=1/a
    for i in range(1,b):
        beta_fct_value=beta_fct_value*(i/(a+i))
    return beta_fct_value


def compute_weight_list(m, alpha=1, beta=1):
    '''
    Given a prior distribution (beta distribution (alpha,beta))
    beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.
    # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
    '''
    weight_list=np.zeros(m)
    normalizing_constant=1/beta_constant(alpha, beta)
    for j in np.arange(m):
        # when the cardinality of random sets is j
        weight_list[j]=beta_constant(j+alpha, m-j+beta-1)/beta_constant(j+1, m-j)
        weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
    return weight_list


def betasv_permsampling_from_data(X_feature, y_feature, n_data, a, b, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  """
  weight_vector = np.zeros(n_data)
  for j in range(1, n_data+1):
    w = n_data * beta(j+b-1, n_data-j+a) / beta(a, b) * comb(n_data-1, j-1)
    weight_vector[j-1] = w
  """
  weight_vector = compute_weight_list(n_data, alpha=a, beta=b)
  #print(weight_vector[:1000])

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += weight_vector[j]*(with_score-without_score)

  return sv_vector / n_perm



def banzhaf_grouptest_bias_from_data(X_feature, y_feature, n_data, dummy=True):

  n_sample = len(y_feature)
  if dummy:
    N = n_data+1
  else:
    N = n_data

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  sv_approx = np.zeros(n_data)

  for i in range(n_data):
    if np.sum(A[:, i]) == n_sample:
      sv_approx[i] = np.dot( A[:, i], B ) / n_sample
    elif np.sum(A[:, i]) == 0:
      sv_approx[i] = - np.dot( (1-A[:, i]), B ) / n_sample
    else:
      sv_approx[i] = np.dot(A[:, i], B)/np.sum(A[:, i]) - np.dot(1-A[:, i], B)/np.sum(1-A[:, i])

  return sv_approx


def sample_utility_multiple(x_train, y_train, x_test, y_test, utility_func, n_repeat):

  acc_lst = []

  for _ in range(n_repeat):
    acc = utility_func(x_train, y_train, x_test, y_test)
    acc_lst.append(acc)

  return acc_lst


def sample_utility_shapley_perm(n_perm, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)
      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))

  return X_feature_test, y_feature_test


def sample_L_utility_shapley_perm(n_perm, du_model, n_data):

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)

      subset_bin = np.zeros((1, 200))
      subset_bin[0, subset_index] = 1

      y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
      y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)




def uniformly_subset_sample(dataset):

  sampled_set = []

  for data in dataset:
    if randint(0, 1) == 1:
      sampled_set.append(data)

  return sampled_set


def sample_utility_banzhaf_mc(n_sample, utility_func, utility_func_args, target_ind):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  n_sample_per_data = int( n_sample / 2 )

  # utility set will store tuples (with/without target index)
  utility_set = []

  dataset = np.arange(n_data)
  leave_one_out_set = np.delete(dataset, target_ind)

  for _ in range(n_sample_per_data):
    sampled_idx_without = np.array(uniformly_subset_sample(leave_one_out_set))
    utility_without = utility_func(x_train[sampled_idx_without], y_train[sampled_idx_without], x_val, y_val)
    sampled_idx_with = np.array( list(sampled_idx_without) + [target_ind] )
    utility_with = utility_func(x_train[sampled_idx_with], y_train[sampled_idx_with], x_val, y_val)

    to_be_store = { 'ind': sampled_idx_without, 'u_without': utility_without, 'u_with': utility_with }

    utility_set.append(to_be_store)

  return utility_set


# Implement Dummy Data Point Idea
def sample_utility_banzhaf_gt(n_sample, utility_func, utility_func_args, dummy=False):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test




# Leave-one-out
def sample_utility_loo(utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data

  X_feature_test = []
  y_feature_test = []

  u_total = utility_func(x_train, y_train, x_val, y_val)

  for i in range(N):

    loo_index = np.ones(N)
    loo_index[i] = 0
    loo_index = loo_index.nonzero()[0]

    X_feature_test.append( loo_index )
    y_feature_test.append( utility_func(x_train[loo_index], y_train[loo_index], x_val, y_val) )

  return X_feature_test, y_feature_test, u_total


# y_feature is 1-dim array, u_total is scalar
def compute_loo(y_feature, u_total):
  score = np.zeros(len(y_feature))
  for i in range(len(y_feature)):
    score[i] = u_total - y_feature[i]
  return score



def rank_neighbor(x_test, x_train):
  distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

  return sv


def knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K)

  return sv




def uniformly_subset_givensize(dataset, size):

  sampled_set = np.random.permutation(dataset)

  return sampled_set[:int(size)]


def sample_utility_givensize(n_sample_lst, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []

  for size in n_sample_lst:

    subset_ind = np.array(uniformly_subset_givensize( np.arange(n_data), size )).astype(int)

    X_feature_test.append(subset_ind)

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_utility_shapley_gt(n_sample, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]

    if size == 0:
      y_feature_test.append( [0.1] )
    else:
      y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_L_utility_shapley_gt(n_sample, du_model, n_data):

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]
    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)


# Implement Dummy Data Point Idea
def sample_L_utility_banzhaf_gt(n_sample, du_model, n_data, dummy=False):

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)



def shapley_grouptest_from_data(X_feature, y_feature, n_data):

  n_sample = len(y_feature)
  N = n_data+1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  C = {}
  for i in range(N):
    for j in [n_data]:
      C[(i,j)] = Z*(B.dot(A[:,i] - A[:,j]))/n_sample

  sv_last = 0
  sv_approx = np.zeros(n_data)

  for i in range(n_data): 
    sv_approx[i] = C[(i, N-1)] + sv_last
  
  return sv_approx








