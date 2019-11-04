import os
import sys
import json
import imp
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from kernels import *
data_name = sys.argv[1]
data = pd.readdata = pd.read_csv('data/processed/' + data_name + '.csv')

with open('names.json', mode = 'r') as io:
    names = json.loads(io.read())
cols = names[data_name]['cols']
target = ['target']

kernel_names = ['linear'
               ,'polynomial'
               ,'gaussian'
               ,'exponential'
               ,'hyperbolic_tangent'
               ,'rational_quadratic'
               ,'inverse_multiquadratic'
               ,'log'
               ,'power']

score_train = {}
score_valid = {}

for kernel_name in kernel_names:

    score_train[kernel_name] = [0] * 10
    score_valid[kernel_name] = [0] * 10

    kf = KFold(n_splits=10)
    count = 0

    for train_index, valid_index in kf.split(data):

        train_x = data.loc[train_index,cols].copy()
        valid_x = data.loc[valid_index,cols].copy()
        train_y = data.loc[train_index,'target'].copy()
        valid_y = data.loc[valid_index,'target'].copy()

        kernel_ = getattr(sys.modules[__name__], "%s_kernel" % kernel_name)
        
        if names[data_name]['type'] == 'classification':
            model = SVC(kernel = 'precomputed', cache_size = 30000)
        if names[data_name]['type'] == 'regression':
            model = SVR(kernel = 'precomputed', cache_size = 30000)
        
        X = np.array(train_x[cols])
        gram_matrix = kernel_(X,X)
        model.fit(gram_matrix,train_y)
        
        preds_train = model.predict(gram_matrix)
        
        X_valid = np.array(valid_x[cols])
        gram_matrix_valid = kernel_(X_valid,X)
        preds_valid = model.predict(gram_matrix_valid)
        
        if names[data_name]['type'] == 'classification':
            score_train[kernel_name][count] = accuracy_score(preds_train,train_y)
            score_valid[kernel_name][count] = accuracy_score(preds_valid,valid_y)
        if names[data_name]['type'] == 'regression':
            score_train[kernel_name][count] = r2_score(preds_train,train_y)
            score_valid[kernel_name][count] = r2_score(preds_valid,valid_y)

#         print(count, '번째 fold의 결과')
#         print(score_train[kernel_name][count])
#         print(score_valid[kernel_name][count])
#         print(confusion_matrix(preds_valid,valid_y))

#         print('______________________________________________________')

        count += 1
    print('______________________________________________________')
    print(kernel_name)
    print('10fold평균 결과, train set : ',np.array(score_train[kernel_name]).mean())
    print('10fold평균 결과, valid set : ',np.array(score_valid[kernel_name]).mean())


result = pd.DataFrame()
result['kernel'] = kernel_names
result['train_score'] = [np.array(score_train[kernel_name]).mean() for kernel_name in kernel_names]
result['valid_score'] = [np.array(score_valid[kernel_name]).mean() for kernel_name in kernel_names]
result['valid_std'] = [np.array(score_valid[kernel_name]).std() for kernel_name in kernel_names]

result.to_csv('result/' + data_name + '.csv', index = False)
