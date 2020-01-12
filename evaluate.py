import os
import sys
import json
import imp
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from kernels import *
data_name = sys.argv[1]
data = pd.read_csv('data/processed/' + data_name + '.csv')

with open('names.json', mode = 'r') as io:
    names = json.loads(io.read())
cols = names[data_name]['cols']

with open('pars.json', mode = 'r') as io:
    pars = json.loads(io.read())

target = ['target']

kernel_names = ['linear'
               ,'polynomial'
               ,'hyperbolic_tangent'
               ,'vovks_real_polynomial'
               ,'vovks_infinite_polynomial'
               ,'gaussian'
               ,'laplacian'
               ,'rational_quadratic'
               ,'multiquadratic'
               ,'inverse_multiquadratic'
               ,'circular'
               ,'spherical'
               ,'wave'  
               ,'power'
               ,'log'
               ,'generalized_tstudent'
               ,'anova'
               ,'spline'
               ,'chi_square'
               ,'histogram_intersection'
               ,'hellingers']


def evaluate(kernel_name,par):
    
    score_train = [0] * 10
    score_valid = [0] * 10
    
    kf = KFold(n_splits=10, shuffle = True)
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
            model = SVR(kernel = 'precomputed', cache_size = 30000,
                        epsilon = 0.5,max_iter = 10000)
        
        try:
            X = np.array(train_x[cols])
            gram_matrix = kernel_(X,X,[par])
            model.fit(gram_matrix,train_y)

            preds_train = model.predict(gram_matrix)

            X_valid = np.array(valid_x[cols])
            gram_matrix_valid = kernel_(X_valid,X,[par])
            preds_valid = model.predict(gram_matrix_valid)
        except:
            print('잘 정의되지 않는 kernel :',kernel_name)
            score_train = [-1] * 10
            score_valid = [-1] * 10
            break

        try:
            
            if names[data_name]['type'] == 'classification':
                score_train[count] = accuracy_score(preds_train,train_y)
                score_valid[count] = accuracy_score(preds_valid,valid_y)
            if names[data_name]['type'] == 'regression':
                score_train[count] = mean_squared_error(preds_train,train_y)
                score_valid[count] = mean_squared_error(preds_valid,valid_y)
        
        except:
            print('점수 계산상 발산함 kernel :', kernel_name)
            score_train = [-1] * 10
            score_valid = [-1] * 10
            break
#         print(count, '번째 fold의 결과')
#         print(score_train[kernel_name][count])
#         print(score_valid[kernel_name][count])
#         print(confusion_matrix(preds_valid,valid_y))

#         print('______________________________________________________')

        count += 1
    
        
    return(score_train,score_valid)

for kernel_name in kernel_names:
    
    par_list = pars[kernel_name]
    result = pd.DataFrame()
    result['kernel'] = [kernel_name] * len(par_list) 
    result['par'] = par_list

    for i, par in enumerate(par_list):
        
        score_train, score_valid = evaluate(kernel_name, par)
    
        print('______________________________________________________')
        print(kernel_name, par)
        print('10fold평균 결과, train set : ',np.array(score_train).mean())
        print('10fold평균 결과, valid set : ',np.array(score_valid).mean())
    
        result.loc[i,'train_score'] = np.array(score_train).mean()
        result.loc[i,'valid_score'] = np.array(score_valid).mean()
        result.loc[i,'valid_std'] = np.array(score_valid).std()

    result.to_csv('result/' + data_name + '_' + kernel_name + '.csv', index = False)

    
    
    
    
    
    
# for kernel_name in kernel_names:

    
#     par = 0
    
#     score_train[kernel_name] = [0] * 10
#     score_valid[kernel_name] = [0] * 10

#     kf = KFold(n_splits=10, shuffle = True)
#     count = 0
    
#     for train_index, valid_index in kf.split(data):

#         train_x = data.loc[train_index,cols].copy()
#         valid_x = data.loc[valid_index,cols].copy()
#         train_y = data.loc[train_index,'target'].copy()
#         valid_y = data.loc[valid_index,'target'].copy()

#         kernel_ = getattr(sys.modules[__name__], "%s_kernel" % kernel_name)
        
#         if names[data_name]['type'] == 'classification':
#             model = SVC(kernel = 'precomputed', cache_size = 30000)
#         if names[data_name]['type'] == 'regression':
#             model = SVR(kernel = 'precomputed', cache_size = 30000,
#                         epsilon = 0.5,max_iter = 10000)
        
#         try:
#             X = np.array(train_x[cols])
#             gram_matrix = kernel_(X,X,par)
#             model.fit(gram_matrix,train_y)

#             preds_train = model.predict(gram_matrix)

#             X_valid = np.array(valid_x[cols])
#             gram_matrix_valid = kernel_(X_valid,X,par)
#             preds_valid = model.predict(gram_matrix_valid)
#         except:
#             print('잘 정의되지 않는 kernel :',kernel_name)
#             score_train[kernel_name] = [-1] * 10
#             score_valid[kernel_name] = [-1] * 10
#             break

#         if names[data_name]['type'] == 'classification':
#             score_train[kernel_name][count] = accuracy_score(preds_train,train_y)
#             score_valid[kernel_name][count] = accuracy_score(preds_valid,valid_y)
#         if names[data_name]['type'] == 'regression':
#             score_train[kernel_name][count] = mean_squared_error(preds_train,train_y)
#             score_valid[kernel_name][count] = mean_squared_error(preds_valid,valid_y)

# #         print(count, '번째 fold의 결과')
# #         print(score_train[kernel_name][count])
# #         print(score_valid[kernel_name][count])
# #         print(confusion_matrix(preds_valid,valid_y))

# #         print('______________________________________________________')

#         count += 1
#     print('______________________________________________________')
#     print(kernel_name)
#     print('10fold평균 결과, train set : ',np.array(score_train[kernel_name]).mean())
#     print('10fold평균 결과, valid set : ',np.array(score_valid[kernel_name]).mean())


# result = pd.DataFrame()
# result['kernel'] = kernel_names
# result['train_score'] = [np.array(score_train[kernel_name]).mean() for kernel_name in kernel_names]
# result['valid_score'] = [np.array(score_valid[kernel_name]).mean() for kernel_name in kernel_names]
# result['valid_std'] = [np.array(score_valid[kernel_name]).std() for kernel_name in kernel_names]

# result.to_csv('result/' + data_name + '.csv', index = False)
