from __future__ import print_function, absolute_import, division

import re
import math
import random
import time
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import h5py

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session 

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import preprocessing

np.random.seed(8)
random.seed(8)

def building_3dense_model_task1(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, final_act, folder):
    """"Builds a model with three Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    flatten_seq = Flatten()(dropout_seq)
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(flatten_seq)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    dense_seq3 = Dense(number_neurons[2], activation='relu')(dropout_seq2)
    dropout_seq3 = Dropout(drop_hid)(dense_seq3)
    main_dense = Dense(n_class, activation=final_act)(dropout_seq3)
    print(model.summary())
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

#### sin terminar : meter convolucional!!! y todos los parametros de las convolucionales
def building_convdense_model_task1(max_len, dict_size, nnumber_neurons, n_class, drop_per, drop_hid, final_act, folder):
    """"Builds a model with three Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    flatten_seq = Flatten()(dropout_seq)
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(flatten_seq)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    dense_seq3 = Dense(number_neurons[2], activation='relu')(dropout_seq2)
    dropout_seq3 = Dropout(drop_hid)(dense_seq3)
    main_dense = Dense(n_class, activation=final_act)(dropout_seq3)
    print(model.summary())
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model