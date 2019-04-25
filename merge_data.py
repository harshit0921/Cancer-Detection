import pandas as pd
import time
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

script_location = os.path.dirname(__file__)
skin_data_directory = os.path.join(script_location, 'Skin_Data')
cnn_data_direcory = os.path.join(script_location, 'Skin_Data_CNN')


def merge_skin_data(batches=10, model = None):
    
    print("Merge begins")
    batch_file_directory = os.path.join(script_location, 'batches')
    
    data = pd.read_csv(os.path.join(batch_file_directory, 'batch' + str(0) + '.csv'))
    d = data.values.shape[1]
    l = [i for i in range(d - 1)]
    l.append('Label')
    features = [i for i in range(d - 1)]
    
    data = pd.read_csv(os.path.join(batch_file_directory, 'batch' + str(0) + '.csv'), names =l)
    x = data.loc[:, features].values
    y = data.loc[:,['Label']].values
    t0 = time.time()

    for i in range(1,batches):
        data = pd.read_csv(os.path.join(batch_file_directory, 'batch' + str(i) + '.csv'), names =l)
        x1 = data.loc[:, features].values
        y1 = data.loc[:,['Label']].values
        x= np.append(x,x1, axis =0)
        y= np.append(y,y1, axis =0)
     
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5.0, random_state=0)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    print('X_Train dimension {}'.format(x_train.shape))
    print('X_Test dimension {}'.format(x_test.shape))
    
    if (model is None):
        
        directory = skin_data_directory
   
        scale = StandardScaler()
        
        scale.fit(x_train)
        
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)
        
        pca = PCA(.99)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        
        print('X_Train after transformation {}'.format(x_train.shape))
        print('X_Test after transformation {}'.format(x_test.shape))
    else:
        
        directory = cnn_data_direcory
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(os.path.join(directory, 'x_train.csv'), x_train, delimiter = ",")
    np.savetxt(os.path.join(directory, 'x_test.csv'), x_test, delimiter = ",")
    np.savetxt(os.path.join(directory, 'y_train.csv'), y_train, delimiter = ",")
    np.savetxt(os.path.join(directory, 'y_test.csv'), y_test, delimiter = ",")    
    
    t1 = time.time()

    total = t1-t0
    print('Time taken for merge = {} mins'.format(total/60))
    return x_train, x_test, y_train, y_test


def get_data_skin(model = None):
    if model is None:
        directory = skin_data_directory
    else:
        directory = cnn_data_direcory
    data = pd.read_csv(os.path.join(directory, 'x_train.csv'))
    l = [i for i in range(data.values.shape[1])]
    x_train = pd.read_csv(os.path.join(directory, 'x_train.csv'), names =l)
    x_train = x_train.loc[:,:].values
    x_test = pd.read_csv(os.path.join(directory, 'x_test.csv'), names =l)
    x_test = x_test.loc[:,:].values
    y_train = pd.read_csv(os.path.join(directory, 'y_train.csv'), names =['Label'])
    y_train = y_train.loc[:,:].values
    y_test = pd.read_csv(os.path.join(directory, 'y_test.csv'), names =['Label'])
    y_test = y_test.loc[:,:].values
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return x_train, x_test, y_train, y_test
    
def create_breast_data():

    data = pd.read_csv(os.path.join(script_location, 'breast_cancer_data.csv'))
    x = data.values[:, :-1]
    y = data.values[:, -1]
     
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5.0, random_state=100)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
   
    scale = StandardScaler()
    scale.fit(x_train)
    
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)
    
    return x_train, x_test, y_train, y_test

