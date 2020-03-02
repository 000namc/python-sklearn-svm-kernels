import os
import json
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import KFold



for i in range(10):
    if 'fold' + str(i+1) not in os.listdir():
        os.mkdir('fold' + str(i+1))

        
        
data_names = ["breast_cancer"
             ,"yeast"
             ,"segmentation"
             ,"waveform"
             ,"leaf"
             ,"wine"
             ,"crime"
             ,"airfoil"
             ,"fire"
             ,"fish"]

fold = dict()

for data_name in data_names:
    
    kf = KFold(n_splits=10, shuffle = True)
    data = pd.read_csv('../data/processed/' + data_name + '.csv')
    fold[data_name] = dict()
    
    for i, (train_index, valid_index) in enumerate(kf.split(data)):

        fold[data_name]['fold' + str(i+1)] = dict()
        fold[data_name]['fold' + str(i+1)]['train'] = train_index.tolist()
        fold[data_name]['fold' + str(i+1)]['valid'] = valid_index.tolist()


with open('fold.json', mode = 'w') as io:
    json.dump(fold, io , indent = '\t')
