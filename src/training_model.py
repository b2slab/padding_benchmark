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
from keras.callbacks import ModelCheckpoint

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import Target
from src.callbacks import *

np.random.seed(8)
random.seed(8)    

def batch_generator(batch_size, file, dataset, indices, labels="labels"):
    """Generates batches of a given size from a dataset of the HDF5 file. 
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
    print("It has been ", str(datetime.timedelta(seconds=(end - start))))
    timee = (end - start)/3600
    #if the folder doesn't exist, create it
    if not os.path.exists(''.join(string for string in [absPath, 'data/results/', folder, model_type, '/'])):
        os.makedirs(''.join(string for string in [absPath, 'data/results/', folder, model_type, '/']))
    file_time = ''.join(string for string in [absPath, 'data/results/', folder, model_type, '/time.pickle'])

    with open(file_time, "wb") as output_file:
        pickle.dump(timee, output_file)

def saving_results(history, model_type, folder, task, idx=None, kfold_bool=False):
    """Saving F1-score and history"""
    if kfold_bool == False:
        if not os.path.exists(''.join(string for string in [absPath, 'data/results/', folder, task, model_type])):
            os.makedirs(''.join(string for string in [absPath, 'data/results/', folder, task, model_type]))
                  
    #file_f1 = ''.join(string for string in [absPath, 'data/results/', folder, '/', model_type, '/f1_score.pickle'])
    #with open(file_f1, "wb") as output_file:
    #    pickle.dump(f1s, output_file)
                    
        file_his = ''.join(string for string in [absPath, 'data/results/',folder, task, model_type, '/history.pickle'])

        with open(file_his, "wb") as output_file:
            pickle.dump(history.history, output_file)
    else:
        if not os.path.exists(''.join(string for string in [absPath, 'data/results/', folder, task, model_type, '/', str(idx), '/'])):
            os.makedirs(''.join(string for string in [absPath, 'data/results/', folder, task, model_type, '/', str(idx), '/']))
        file_his = ''.join(string for string in [absPath, 'data/results/',folder, task, model_type, '/', str(idx), '/history.pickle'])

        with open(file_his, "wb") as output_file:
            pickle.dump(history.history, output_file)

def model_number_layers(model):
    """Print the correspondence between layers and indices"""
    for idx, layer in enumerate(model.layers):
        print(idx, layer.name)
        
def trainval_generators(indices, indices_aug, model_type, folder, batch_size, labels, kfold_bool=False):
    """create training and validation generators depending on if it's kfold or not"""
    #which data to load
    if model_type == "aug_padding":
        file_data = os.path.join(absPath, 'data/', folder, 'aug_data.h5')
        indices = indices_aug
    else:
        file_data = os.path.join(absPath, 'data/', folder, 'data.h5')
    h5f = h5py.File(file_data, 'r')
    #now creating batches
    if kfold_bool == False:
        i_train, i_val, i_test = indices
        train_generator = batch_generator(batch_size, file_data, model_type, i_train, labels)
        val_generator = batch_generator(batch_size, file_data, model_type, i_val, labels)
        generators = (train_generator, val_generator)
    else:
        generators = []
        for k_fold in indices:
            i_train, i_val, i_test = k_fold
            train_generator = batch_generator(batch_size, file_data, model_type, i_train, labels)
            val_generator = batch_generator(batch_size, file_data, model_type, i_val, labels)
            generators.append((train_generator, val_generator))
    return generators
                    
def calling_callbacks(folder_cp, folder_wei, model_type, x_val, y_val, datasets_names, layers_numbers, scores=True, multilabel=False, weights=False, nans=True):
    """Defining callbacks for the models"""
    # define the checkpoint
    cp_path = ''.join(string for string in [absPath, 'data/checkpoint/', folder_cp, 
                                                   '/weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5'])
    checkpoint = ModelCheckpoint(cp_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    if scores == True:
        scores = Scores(x_val, y_val, multilabel)
        callbacks_list.append(scores)
    if nans == True:
        nans = Nans(x_val, y_val)
        callbacks_list.append(nans)
    if weights == True:
        file_weights = ''.join(string for string in [absPath, 'data/weights/',folder_wei, model_type])
        weights = Weights(file_weights, datasets_names, layers_numbers)
        callbacks_list.append(weights)
    return callbacks_list

