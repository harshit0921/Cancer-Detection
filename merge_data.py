#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:05:25 2019

@author: shivamodeka
"""

#from TransformData import transform
import pandas as pd
import time
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

script_location = os.path.dirname(__file__)

def merge():

    
    l = [i for i in range(7500)]
#    l.extend(('Age','Male','Female','Label'))
    l.append('Label')
    features = [i for i in range(7500)]
#    features.extend(('Age','Male','Female'))
    
    data = pd.read_csv(os.path.join(script_location, 'batches_cnn/batch' + str(0) + '.csv'), names =l)
    x = data.loc[:, features].values
    y = data.loc[:,['Label']].values
    t0 = time.time()

    for i in range(1,10):
        data = pd.read_csv(os.path.join(script_location, 'batches_cnn/batch' + str(i) + '.csv'), names =l)
        x1 = data.loc[:, features].values
        y1 = data.loc[:,['Label']].values
        x= np.append(x,x1, axis =0)
        y= np.append(y,y1, axis =0)
     
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5.0, random_state=0)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
#    
    scale = StandardScaler()
    print('X_Train before transformation {}'.format(x_train.shape))
    print('X_Test before transformation {}'.format(x_test.shape))
    scale.fit(x_train)
    
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)
    
    pca = PCA(.99)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    print('X_Train after transformation {}'.format(x_train.shape))
    print('X_Test after transformation {}'.format(x_test.shape))
    
    np.savetxt('x_train.csv',x_train, delimiter = ",")
    np.savetxt('x_test.csv',x_test, delimiter = ",")
    np.savetxt('y_train.csv',y_train, delimiter = ",")
    np.savetxt('y_test.csv',y_test, delimiter = ",")
    
    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype = torch.float32) 
    y_train = torch.tensor(y_train, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32) 
    
    
    t1 = time.time()

    total = t1-t0
    print('Time taken for merge = {} mins'.format(total/60))
    return x_train, x_test, y_train, y_test

def get_data():
    
    l = [i for i in range(320)]
    x_train = pd.read_csv(os.path.join(script_location, 'CNN_Data/x_train.csv'), names =l)
    x_train = x_train.loc[:,:].values
    x_test = pd.read_csv(os.path.join(script_location, 'CNN_Data/x_test.csv'), names =l)
    x_test = x_test.loc[:,:].values
    y_train = pd.read_csv(os.path.join(script_location, 'CNN_Data/y_train.csv'), names =['Label'])
    y_train = y_train.loc[:,:].values
    y_test = pd.read_csv(os.path.join(script_location, 'CNN_Data/y_test.csv'), names =['Label'])
    y_test = y_test.loc[:,:].values
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    x_train = torch.tensor(x_train, dtype = torch.float32)
    x_test = torch.tensor(x_test, dtype = torch.float32) 
    y_train = torch.tensor(y_train, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32) 
    
#    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    return x_train, x_test, y_train, y_test
    


if __name__ == '__main__':
    get_data()
