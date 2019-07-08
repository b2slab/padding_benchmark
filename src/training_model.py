from __future__ import print_function, absolute_import, division

import random
import time
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import h5py
import datetime

import tensorflow as tf
import keras
from keras.models import load_model
from sklearn import metrics
from collections import Counter

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import Target

np.random.seed(8)
random.seed(8)    

def batch_generator(batch_size, file, dataset, indices, labels="labels"):
    """Generates batches of a fiven size from a dataset of the HDF5 file. 
    It needs the indices of the dataset"""
    sample_size = len(indices)
    n_batches = int(sample_size/batch_size)
    h5f = h5py.File(file,'r')
    instarget = Target('AAAAAA')
    aa_to_int = instarget.predefining_dict()
    while True:    
        for i in range(n_batches):
            if i == n_batches:
                batch_samples = h5f[dataset][i*batch_size:sample_size]
                seqs_onehot = instarget.int_to_onehot(list(batch_samples), len(aa_to_int))
                batch_y = h5f[labels][i*batch_size:sample_size]
            else:
                batch_samples = h5f[dataset][i*batch_size:i*batch_size+batch_size]
                seqs_onehot = instarget.int_to_onehot(list(batch_samples), len(aa_to_int))
                batch_y = h5f[labels][i*batch_size:i*batch_size+batch_size]
            yield (seqs_onehot, batch_y)        

def loading_val_data(x_name, y_name, i_val, h5_file):
    """Loading data for validation"""
    h5f = h5py.File(h5_file, 'r')
    x_val = h5f[x_name][sorted(i_val)]
    y_val = h5f[y_name][sorted(i_val)]
    return x_val, y_val

def count_time(start, end, folder, model_type):
    """Count the time the model takes to run and save it to a file"""
    print("It has been ", str(datetime.timedelta(seconds=(end - start)))
    timee = (end - start)/3600
    #if the folder doesn't exist, create it
    if not os.path.exists(''.join(string for string in [absPath, 'data/results', folder, '/', model_type]):
        os.makedirs(''.join(string for string in [absPath, 'data/results', folder, '/', model_type])
    file_time = ''.join(string for string in [absPath, 'data/results', folder, '/', model_type, '/time.pickle'])

    with open(file_time, "wb") as output_file:
        pickle.dump(file_time, output_file)

def saving_results(f1s, model_type, folder):
    """Saving F1-score and history"""
    if not os.path.exists(''.join(string for string in [absPath, 'data/results', folder, '/', model_type]):
        os.makedirs(''.join(string for string in [absPath, 'data/results', folder, '/', model_type])
                    
    file_f1 = ''.join(string for string in [absPath, 'data/results/', folder, '/', model_type, '/f1_score.pickle'])
    with open(file_f1, "wb") as output_file:
        pickle.dump(f1s, output_file)
                    
    file_his = ''.join(string for string in [absPath, 'data/results/',folder, '/', model_type, '/history.pickle'])

    with open(file_his, "wb") as output_file:
        pickle.dump(history.history, output_file)

                    
