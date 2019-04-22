import torch
import torch.nn.functional as F
from torch import autograd, optim, nn
from scipy.io import loadmat
import torch.utils.data
import matplotlib.pyplot as plt
from logistic import Logistic
import numpy as np
import pandas as pd
from merge_data import merge
from sklearn.preprocessing import StandardScaler
from confusion_matrix import confusionMatrix

l = [i for i in range(1415)]
data = pd.read_csv('x_train.csv', names = l)
x_train = data.values
data = pd.read_csv('x_test.csv', names = l)
x_test = data.values

a=[5]
data = pd.read_csv('y_train.csv', names = a)
y_train = data.values.flatten()
data = pd.read_csv('y_test.csv', names = a)
y_test = data.values.flatten()

#x_train, x_test, y_train, y_test = merge()

logistic = Logistic(x_train, y_train)
print(logistic.theta)
y_pred = logistic.predict(x_train)
print("\nTraining Classification accuracy: ")
print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
y_pred = logistic.predict(x_test)
print("\nTesting Classification accuracy: ")
print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])


#table = pd.read_csv('breast_cancer_data.csv')
#train_range = int(table.values.shape[0] * 0.8)
#x_train = StandardScaler().fit_transform(table.values[0:train_range, :-1])
#y_train = table.values[0:train_range, -1]
#x_test = StandardScaler().fit_transform(table.values[train_range:table.values.shape[0], :-1] )
#y_test = table.values[train_range:table.values.shape[0], -1]
#logistic = Logistic(x_train, y_train)
#print(logistic.theta)
#y_pred = logistic.predict(x_train)
#print("\nTraining Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
#y_pred = logistic.predict(x_test)
#print("\nTesting Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])

confusionMatrix(y_test, y_pred)