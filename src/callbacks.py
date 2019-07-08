from __future__ import division, absolute_import

import sys
import os
import time

import numpy as np
import pandas as pd
import h5py
import math
import bisect
import pylab as pl

import numpy.ma as ma
from keras import Model
from keras.callbacks import EarlyStopping, Callback 
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import Target

def saving_initial_weights(datasets_names, layers_numbers, folder, model_type, layers_numbers):
    """Saving initial weights of a model into a pre-specified folder"""
    if len(datasets_names) != len(layers_numbers):
        print("datasets_names should be the same length as layers_numbers")
    #create directories if they don't exist
    if not os.path.exists(''.join(string for string in [absPath, 'data/weights/', model_type]):
        os.makedirs(''.join(string for string in [absPath, 'data/weights/', model_type])
    
    filee = ''.join(string for string in [absPath, 'data/weights/', folder, model_type, '/model_weights.h5'])
    file_weights = h5py.File(filee, 'w')
    
    for idx,i in enumerate(datasets_names):
        if not os.path.exists(''.join(string for string in [absPath, 'data/weights/', model_type, '/', i]):
            os.makedirs(''.join(string for string in [absPath, 'data/weights/', model_type, '/', i])
        name_dataset = ''.join(string for string in ['/', i, '/weights_0'])
        file_weights.create_dataset(name_dataset, shape=model.layers[layers_numbers[idx]].get_weights()[0].shape, data=model.layers[layers_numbers[idx]].get_weights()[0])
    file_weights.close()


class Weights(Callback):
    """Callback to save weights"""
    def __init__(self, filee, datasets_names, layers_numbers):
        self.filee = filee 
        self.datasets_names = datasets_names
        self.layers_numbers = layers_numbers
        
    def on_train_begin(self, logs={}):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.file = h5py.File('%s/model_weights.h5' % self.filee,'r+')

        self.times_file = open('%s/time_data.txt' % self.filee,'w')
        self.times_file.write(self.timestr + '\n')

    def on_epoch_end(self, epoch, logs={}):
        for idx, i in enumerate(datasets_names):
            dataset_name = ''.join(string for string in ['/', i, '/weights_%s'])
            self.file.create_dataset(dataset_name % str(epoch).zfill(3),
                                     shape=self.model.layers[layers_numbers[idx]]get_weights()[0].shape, 
                                    data=self.model.layers[layers_numbers[idx]].get_weights()[0])
            self.times_file.write(self.timestr + '\n')
    
    def on_train_end(self, logs={}):
        self.file.close()
        self.times_file.close()
                        
def saving_initial_biases(datasets_names, layers_numbers, folder, model_type, layers_numbers):
    """Saving initial weights of a model into a pre-specified folder"""
    if len(datasets_names) != len(layers_numbers):
        print("datasets_names should be the same length as layers_numbers")
    #create directories if they don't exist
    if not os.path.exists(''.join(string for string in [absPath, 'data/biases/', model_type]):
        os.makedirs(''.join(string for string in [absPath, 'data/biases/', model_type])
    
    filee = ''.join(string for string in [absPath, 'data/biases/', folder, model_type, '/model_weights.h5'])
    file_weights = h5py.File(filee, 'w')
    
    for idx,i in enumerate(datasets_names):
        if not os.path.exists(''.join(string for string in [absPath, 'data/biases/', model_type, '/', i]):
            os.makedirs(''.join(string for string in [absPath, 'data/biases/', model_type, '/', i])
        name_dataset = ''.join(string for string in ['/', i, '/bias_0'])
        file_weights.create_dataset(name_dataset, shape=model.layers[layers_numbers[idx]].get_weights()[1].shape, data=model.layers[layers_numbers[idx]].get_weights()[1])
    file_weights.close()
        
class Biases(Callback):
    """Callback to save bias"""
    def __init__(self, filee, datasets_names, layers_numbers):
        self.filee = filee 
        self.datasets_names = datasets_names
        self.layers_numbers = layers_numbers
        
    def on_train_begin(self, logs={}):
        #self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.file = h5py.File('%s/model_biases.h5' % self.filee,'r+')

    def on_epoch_end(self, epoch, logs={}):
        for idx, i in enumerate(datasets_names):
            dataset_name = ''.join(string for string in ['/', i, '/bias_%s'])
            self.file.create_dataset(dataset_name % str(epoch).zfill(3),
                                     shape=self.model.layers[layers_numbers[idx]]get_weights()[1].shape, 
                                    data=self.model.layers[layers_numbers[idx]].get_weights()[1])
    
    def on_train_end(self, logs={}):
        self.file.close()
        #self.times_file.close()

        
class Scores(Callback):
    """Defininf callback for computing F1-score, precision and recall"""
# Source: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
        # https://github.com/keras-team/keras/issues/3358#issuecomment-312531958
    def __init__(self, x_val, y_val, multilabel = False):
        self.x_val = x_val 
        self.y_val = y_val
        self.multilabel = multilabel
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        
    def on_epoch_end(self, epoch, logs={}):
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_predprob = self.model.predict(self.x_val)
        val_predict = y_predprob.argmax(axis=-1)
        
        val_targ = self.y_val
        val_targ = val_targ.argmax(axis=-1)
        if multilabel == False:
            _val_f1 = f1_score(val_targ, val_predict)
            _val_recall = recall_score(val_targ, val_predict)
            _val_precision = precision_score(val_targ, val_predict)
        else:
            _val_f1 = f1_score(val_targ, val_predict, average='macro')
            _val_recall = recall_score(val_targ, val_predict, average='macro')
            _val_precision = precision_score(val_targ, val_predict, average='macro')    
        
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return
  
            
class Nans(Callback):
    """Callback for checking if there are Nans in the predicted"""
    def __init__(self, x_val, y_val):
        self.x_val = x_val 
        self.y_val = y_val
    
    def on_train_begin(self, logs={}):
        self.number_nans = []
 
    def on_epoch_end(self, epoch, logs={}):
        #val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_predprob = self.model.predict(self.x_val)[:,1]
        _number_nans = sum(math.isnan(x) for x in y_predprob)
        self.number_nans.append(_number_nans)
        
        print (" — number_nans: %f" %(_number_nans))
        return