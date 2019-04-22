import torch
import torch.nn.functional as F
from torch import autograd, optim, nn
from scipy.io import loadmat
import torch.utils.data
import matplotlib.pyplot as plt
from logistic import Logistic
import numpy as np
import pandas as pd
from merge_data import merge_skin, get_data_skin, create_breast_data
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from confusion_matrix import confusionMatrix
from svm import SVM

#l = [i for i in range(1415)]
#data = pd.read_csv('x_train.csv', names = l)
#x_train = data.values
#data = pd.read_csv('x_test.csv', names = l)
#x_test = data.values
#
#a=[5]
#data = pd.read_csv('y_train.csv', names = a)
#y_train = data.values.flatten()
#data = pd.read_csv('y_test.csv', names = a)
#y_test = data.values.flatten()

#x_train, x_test, y_train, y_test = merge_skin(10)
#x_train, x_test, y_train, y_test = get_data_skin()

#logistic = Logistic(x_train, y_train)
#print(logistic.theta)
#y_pred = logistic.predict(x_train)
#print("\nTraining Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
#confusionMatrix(y_train, y_pred)
#y_pred = logistic.predict(x_test)
#print("\nTesting Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])


x_train, x_test, y_train, y_test = create_breast_data()
logistic = Logistic(x_train, y_train)
print(logistic.theta)
y_pred = logistic.predict(x_train)
print("\nTraining Classification accuracy: ")
print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
confusionMatrix(y_train, y_pred)
y_pred = logistic.predict(x_test)
print("\nTesting Classification accuracy: ")
print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])

confusionMatrix(y_test, y_pred)


#x_train, x_test, y_train, y_test = get_data_skin()
##Create a svm Classifier
#clf = svm.SVC(kernel='linear') # Linear Kernel
#
##Train the model using the training sets
#clf.fit(x_train, y_train)
#
##Predict the response for test dataset
#y_pred = clf.predict(x_train)
##svm = SVM(x_train, np.where(y_train==0, -1, y_train), 100, 0.01)
##svm.main_routine(5)
##print('b is {}'.format(svm.b))
##print('w is {}'.format(svm.w))
#print("\nTraining Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_train))/y_pred.shape[0])
#confusionMatrix(y_train, y_pred)
#y_pred = clf.predict(x_test)
#print("\nTesting Classification accuracy: ")
#print(100 - 100*np.sum(np.abs(y_pred - y_test))/y_pred.shape[0])
#confusionMatrix(y_test, y_pred)