# Data Banzhaf: A Robust Data Valuation Framework for Machine Learning

By [Jiachen T. Wang](https://tianhaowang.netlify.app/) and [Ruoxi Jia](https://ruoxijia.info/). 

This repository provides an implementation of the paper [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning
](https://arxiv.org/abs/2205.15466) accepted at [AISTATS 2023](http://aistats.org/aistats2023/) as oral presentation. We propose a robust data valuation method, Data Banzhaf, which is powerful at capturing the importance of data points **at the presence of stochasticity in the learning algorithm**.


## Requirements 

The code is tested with Python 3.8 and PyTorch 1.12.1. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`. The code should be compatible with other versions of packages.

## Quick Start

### Sampling Data Subset Utilities

To estimate the data value score of training data points, we need to first sample the performance scores of a learning algorithm trained on different data subsets (where the sampling distribution depends on the specific data values). The following command trains 10,000 MLPs on different subsets of a size-200 Pol dataset from OpenML. `n_repeat' means for each data subset, we train 5 different models on it (with different random seeds for SGD). 

`
python sample_for_value.py --dataset pol --value_type Banzhaf_GT --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 10000 --batch_size 32 --flip_ratio 0.1 --random_state 42 --lr 0.01
`
(this takes around 5 CPU hours)

We provide the utility samples for Shapley value (estimated by permutation sampling) and Banzhaf value (estimated by MSR) here. 

### Computing and Evaluating Data Values

We evaluate the quality of data value scores with two canonical applications: value-weighted training and mislabel data detection. See the following commands which take the Shapley value for example. 

`python applications.py --task weighted_acc --dataset pol --value_type Shapley_Perm --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 10000 --batch_size 32 --lr 1e-2 --flip_ratio 0.1 --random_state 1`

`python applications.py --task mislabel_detect --dataset pol --value_type Shapley_Perm --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 10000 --batch_size 32 --lr 1e-2 --flip_ratio 0.1 --random_state 1`



## Related Repositories

[BetaShapley](https://github.com/ykwon0407/beta_shapley) by Yongchan Kwon.
