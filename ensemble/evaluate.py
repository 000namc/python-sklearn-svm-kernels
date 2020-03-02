import os
import sys
import json
import imp
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from kernels import *


data_name = sys.argv[1]
kernel_name = sys.argv[2]
alpha1 = float(sys.argv[3])
alpha2 = float(sys.argv[4])
lamda = float(sys.argv[5])
epsilon = float(sys.argv[6])

data = pd.read_csv('../data/processed/' + data_name + '.csv')
target = ['target']

with open('../names.json', mode = 'r') as io:
    names = json.loads(io.read())
cols = names[data_name]['cols']

with open('fold.json', mode = 'r') as io:
    fold = json.loads(io.read())

    
for i in range(10):
    
    train_x = data.loc[fold[data_name]['fold' + str(i+1)]['train'],cols].copy()
    valid_x = data.loc[fold[data_name]['fold' + str(i+1)]['valid'],cols].copy()
    train_y = data.loc[fold[data_name]['fold' + str(i+1)]['train'],'target'].copy()
    valid_y = data.loc[fold[data_name]['fold' + str(i+1)]['valid'],'target'].copy()

    kernel_ = getattr(sys.modules[__name__], "%s_kernel" % kernel_name)
        


    if names[data_name]['type'] == 'classification':
        model = SVC(kernel = 'precomputed', cache_size = 10000,
                    C = lamda)
    if names[data_name]['type'] == 'regression':
        model = SVR(kernel = 'precomputed', cache_size = 10000,
                    C = lamda, epsilon = epsilon, max_iter = 10000)
        
    X = np.array(train_x[cols])
    gram_matrix = kernel_(X,X,pars = [alpha1,alpha2])
    model.fit(gram_matrix,train_y)
    X_valid = np.array(valid_x[cols])
    gram_matrix_valid = kernel_(X_valid,X,[alpha1, alpha2])
    preds_valid = model.predict(gram_matrix_valid)
    
    

    result = pd.DataFrame(columns = ['dataname','kernel','alpha1','alpha2','lambda','epsilon','preds'])
    result.loc[0,:] = [data_name, kernel_name, alpha1, alpha2, lamda, epsilon, preds_valid]

    result.to_csv('fold' + str(i+1) + '/' + data_name + '_' + kernel_name + '.csv' ,index = False)
