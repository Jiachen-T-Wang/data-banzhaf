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

from random import randint

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from helper import *
from models import *


class MnistLogistic(torch.nn.Module):
    def __init__(self):
        super(MnistLogistic, self).__init__()

        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x, last=False):

        x = torch.flatten(x, 1)
        logits = self.linear(x)
        # outputs = F.softmax(logits, dim=1)

        return logits


class DogCatLogistic(torch.nn.Module):
    def __init__(self):
        super(DogCatLogistic, self).__init__()

        self.linear = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):

        x = torch.flatten(x, 1)
        logits = self.linear(x)

        return logits


class DogCatMLP(torch.nn.Module):
    def __init__(self):
        super(DogCatMLP, self).__init__()
        self.linear1 = nn.Linear(in_features=512, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits




class MnistLeNet(torch.nn.Module):
    def __init__(self):
        super(MnistLeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding_mode='replicate')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding_mode='replicate')

        self.linear1 = nn.Linear(in_features=640, out_features=500)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x, last=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        # outputs = F.softmax(logits, dim=1)

        return logits

    def getFeature(self, x, numpy=True):
        if x.shape[1]==28:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x

class MnistLargeCNN(torch.nn.Module):
    def __init__(self):
        super(MnistLargeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding_mode='replicate')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding_mode='replicate')

        self.linear1 = nn.Linear(in_features=640, out_features=500)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=500, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x, last=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        logits = self.linear3(x)

        return logits

    def getFeature(self, x, numpy=True):
        if x.shape[1]==28:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x


class SmallCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SmallCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, last=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if last:
          return output, x
        else:
          return output

    def getFeature(self, x, numpy=True):
        if x.shape[1]==32:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x

    def get_embedding_dim(self):
        return 84



def torch_mnist_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  if len(y_train) == 0: return 0.1

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'Logistic':
    net = MnistLogistic().cuda()
  elif model_type == 'SmallCNN':
    net = MnistLeNet().cuda()
  elif model_type == 'LargeCNN':
    net = MnistLargeCNN().cuda()

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc




def torch_cifar_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type[:3] == 'VGG':
    net = VGG(model_type).cuda()
  elif model_type == 'ResNet18':
    net = ResNet18().cuda()
  elif model_type == 'ResNet50':
    net = ResNet50().cuda()
  elif model_type == 'DenseNet':
    net = densenet_cifar().cuda()
  elif model_type == 'SmallCNN':
    net = SmallCNN_CIFAR().cuda()
  else:
    print('not supported')

  n_epoch = 50

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc



def torch_dogcat_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'ResNet18':
    net = ResNet18(num_classes=2).cuda()
  else:
    print('not supported')

  n_epoch = 50

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      if accuracy.item() > max_acc:
        net_best = net

      max_acc = max(max_acc, accuracy.item())

  if return_net:
    return max_acc, net_best
  else:
    return max_acc



def torch_dogcatFeature_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False):

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'Logistic':
    net = DogCatLogistic().cuda()
  elif model_type == 'MLP':
    net = DogCatMLP().cuda()
  else:
    print('not supported')

  n_epoch = 15

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      if accuracy.item() > max_acc:
        net_best = net

      max_acc = max(max_acc, accuracy.item())

  if return_net:
    return max_acc, net_best
  else:
    return max_acc



def binary_data_to_acc(model_type, x_train, y_train, x_test, y_test, w=None):
  if model_type == 'Logistic':
    model = LogisticRegression(max_iter=5000, solver='liblinear')
  elif model_type == 'SVM':
    model = SVC(kernel='rbf', max_iter=5000, C=1)
  if len(y_train)==0:
    return 0.5, 0.5
  try:
    model.fit(x_train, y_train, sample_weight=w)
  except:
    return 0.5
  acc = model.score(x_test, y_test)
  return acc



class BinaryMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(BinaryMLP, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=2)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits


def torch_binary_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if model_type == 'MLP':
    net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).to(device)
  else:
    print('not supported')

  n_epoch = 15
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)
  criterion = torch.nn.CrossEntropyLoss()

  tensor_x, tensor_y = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device)
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      if accuracy.item() > max_acc:
        net_best = net

      max_acc = max(max_acc, accuracy.item())

  if return_net:
    return max_acc, net_best
  else:
    return max_acc









"""
def torch_mnist_Logistic_data_to_acc_weighted(x_train, y_train, x_test, y_test, weights, verbose=0, batch_size=8, lr=0.001):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLogistic().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc


def torch_mnist_Logistic_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=8, lr=0.001):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLogistic().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc, net


def torch_mnist_smallCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=32):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLeNet().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())

  return max_acc, net


def torch_mnist_largeCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=32):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLargeCNN().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc, net




# utility_func_args = [x_train, y_train, x_val, y_val]
def sample_utility_and_cpt(n, size_min, size_max, utility_func, utility_func_args, random_state, save_dir, ub_prob=0, verbose=False):

  x_train, y_train, x_val, y_val = utility_func_args
  x_train, y_train = np.array(x_train), np.array(y_train)

  N = len(y_train)

  np.random.seed(random_state)
  
  for i in range(n):
    if verbose: print('Sample {} / {}'.format(i, n))

    n_select = np.random.choice(range(size_min, size_max))

    subset_index = []

    toss = np.random.uniform()

    # With probability ub_prob, sample a class-imbalanced subset
    if toss > 1-ub_prob:
      n_per_class = int(N / 10)
      alpha = np.ones(10)*30
      alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
      p = np.random.dirichlet(alpha=alpha)
      occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
      counts = np.array([np.sum(occur==i) for i in range(10)])
      for i in range(10):
        # ind_i = np.where(np.argmax(y_train, 1)==i)[0]
        ind_i = np.where(y_train==i)[0]
        if len(ind_i) > counts[i]:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
        else:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
        subset_index = subset_index + list(selected_ind_i)
      subset_index = np.array(subset_index)

    else:
      subset_index = np.random.choice(range(N), n_select, replace=False)

    subset_index = np.array(subset_index)
    acc, net = utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val)

    PATH = save_dir + '_{}.cpt'.format(i)

    torch.save({'model_state_dict': net.state_dict(), 'subset_index': subset_index, 'accuracy': acc}, 
               PATH)
"""





