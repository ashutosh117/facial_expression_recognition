from __future__ import division,print_function
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


def normalize_data(data):
    m = np.mean(data,axis=0,keepdims = True)
    sd = np.std(data,axis=0,keepdims = True)
    norm_data = (data-m)/(sd + 1e-8)
    return norm_data


def create_data_pickle():
    data = pd.read_csv('fer2013/fer2013.csv')
    X_train,X_valid,X_test,Y_train,Y_valid,Y_test = [],[],[],[],[],[]
    n = data.shape[0]
    for i in range(n):
        temp = np.array(data.loc[i,'pixels'].split(' '),dtype = np.float32)
        if data.loc[i,'Usage'] == 'Training':
            Y_train.append(data.loc[i,'emotion'])
            X_train.append(temp)
        elif data.loc[i,'Usage'] == 'PublicTest':
            Y_valid.append(data.loc[i,'emotion'])
            X_valid.append(temp)
        elif data.loc[i,'Usage'] == 'PrivateTest':
            Y_test.append(data.loc[i,'emotion'])
            X_test.append(temp)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    #dumping the files into a pickle
    f = open('fer_data.pickle','w')
    pickle.dump((X_train,Y_train,X_valid,Y_valid,X_test,Y_test),f)
    f.close()
    
def get_normalized_data():
    curr_dir = os.getcwd()
    pickle_file = os.path.join(curr_dir,'fer_data.pickle')
    if not os.path.isfile(pickle_file):
        create_data_pickle()
    f = open('fer_data.pickle','r')
    X_train,Y_train,X_valid,Y_valid,X_test,Y_test = pickle.load(f)
    f.close()
    X_train = normalize_data(X_train)
    X_valid = normalize_data(X_valid)
    X_test = normalize_data(X_test)
    X_train,Y_train = shuffle(X_train,Y_train)
    X_valid,Y_valid,X_test,Y_test = shuffle(X_valid,Y_valid,X_test,Y_test)
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

def get_image_data():
    curr_dir = os.getcwd()
    pickle_file = os.path.join(curr_dir,'fer_data.pickle')
    if not os.path.isfile(pickle_file):
        create_data_pickle()
    f = open('fer_data.pickle','r')
    X_train,Y_train,X_valid,Y_valid,X_test,Y_test = pickle.load(f)
    f.close()
    X_train = np.append(X_train,X_valid,axis=0)
    Y_train = np.append(Y_train,Y_valid,axis=0)
    X_train,X_test = X_train/255.0,X_test/255.0
    n,d = X_train.shape
    d = int(np.sqrt(d))
    X_train = X_train.reshape(n,d,d,1)
    nt = X_test.shape[0]
    X_test = X_test.reshape(nt,d,d,1)
    X_train,Y_train = shuffle(X_train,Y_train)
    X_test,Y_test = shuffle(X_test,Y_test)
    return X_train,Y_train,X_test,Y_test
    

def y2ind(y):
    n = len(y)
    p = len(np.unique(y))
    temp = np.zeros((n,p))
    temp[range(n),y] = 1.0
    return temp
    
    
def get_iris_data():
    data = pd.read_csv('~/machine_learning/datasets/iris/iris.data.txt')
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    X = normalize_data(X)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X,Y = shuffle(X,Y)
    return X,Y


def init_weights(m1,m2):
    w = np.random.randn(m1,m2)*(2.0/np.sqrt(m1))
    b = np.zeros(m2,dtype = np.float32)
    return w.astype(np.float32),b

def init_filter(shape,poolsz):
    w = np.random.randn(*shape)*(2.0/np.sqrt(np.prod(shape[1:]) + shape[0]*(np.prod(shape[2:])/poolsz[0]*poolsz[1])))
    return w.astype(np.float32)