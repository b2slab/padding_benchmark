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
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D, MaxPooling1D, concatenate, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session 

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import Target

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
    model = Model(inputs=[input_seq], outputs=[main_dense])
    print(model.summary())
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def building_1convdense_model_task1(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, n_filt, kernel_size, final_act, folder):
    """"Builds a model with a convolutional layer and three Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    
    c1 = Conv1D(filters=n_filt, kernel_size=kernel_size, padding='same', strides=1, activation='relu')(dropout_seq)
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(c1)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    dense_seq3 = Dense(number_neurons[2], activation='relu')(dropout_seq2)
    dropout_seq3 = Dropout(drop_hid)(dense_seq3)
    flatten = Flatten()(dropout_seq3)
    main_dense = Dense(n_class, activation=final_act)(flatten)
    model = Model(inputs=[input_seq], outputs=[main_dense])
    print(model.summary())
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def building_stackconv_model_task1(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, n_filt, kernel_size, pool_size, final_act, folder):
    """"Builds a model with a stack of convolutional layers and three Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    
    c1 = Conv1D(filters=n_filt, kernel_size=kernel_size[0], padding='same', strides=1, activation='relu')(dropout_seq)
    c2 = Conv1D(filters=n_filt, kernel_size=kernel_size[1], padding='same', strides=1, activation='relu')(dropout_seq)
    c3 = Conv1D(filters=n_filt, kernel_size=kernel_size[2], padding='same', strides=1, activation='relu')(dropout_seq)
    c4 = Conv1D(filters=n_filt, kernel_size=kernel_size[3], padding='same', strides=1, activation='relu')(dropout_seq)
    c5 = Conv1D(filters=n_filt, kernel_size=kernel_size[4], padding='same', strides=1, activation='relu')(dropout_seq)

    merged_conv = concatenate([c1, c2, c3, c4, c5], axis=1)
    #######################################################################################

    #Pooling to reduce dimensions
    pooled = MaxPooling1D(pool_size=pool_size, strides=None, padding='same')(merged_conv)    
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(pooled)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    dense_seq3 = Dense(number_neurons[2], activation='relu')(dropout_seq2)
    dropout_seq3 = Dropout(drop_hid)(dense_seq3)
    main_dense = Dense(n_class, activation=final_act)(dropout_seq3)
    model = Model(inputs=[input_seq], outputs=[main_dense])
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])

    print(model.summary())
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def building_2dense_model_task2(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, final_act, folder):
    """"Builds a model with two Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    flatten_seq = Flatten()(dropout_seq)
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(flatten_seq)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    main_dense = Dense(n_class, activation=final_act)(dropout_seq2)
    model = Model(inputs=[input_seq], outputs=[main_dense])
    print(model.summary())
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def building_1convdense_model_task2(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, n_filt, kernel_size,
                                    final_act, folder):
    """"Builds a model with a convolutional layer and two Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    
    c1 = Conv1D(filters=n_filt, kernel_size=kernel_size, padding='same', strides=1, activation='relu')(dropout_seq)
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(c1)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    flattenn = Flatten()(dropout_seq2)
    main_dense = Dense(n_class, activation=final_act)(flattenn)
    print(model.summary())
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def building_stackconv_model_task2(max_len, dict_size, number_neurons, n_class, drop_per, drop_hid, n_filt, kernel_size,
                                    pool_size, final_act, folder):
    """"Builds a model with a stack of convolutional layers and three Dense layers whose number of neurons are specified in decreasing order in number_neurons"""
    input_seq = Input(shape=(max_len, dict_size), dtype='float32')
    dropout_seq = Dropout(drop_per)(input_seq)
    
    c1 = Conv1D(filters=n_filt, kernel_size=kernel_size[0], padding='same', strides=1, activation='relu')(dropout_seq)
    c2 = Conv1D(filters=n_filt, kernel_size=kernel_size[1], padding='same', strides=1, activation='relu')(dropout_seq)
    c3 = Conv1D(filters=n_filt, kernel_size=kernel_size[2], padding='same', strides=1, activation='relu')(dropout_seq)
    c4 = Conv1D(filters=n_filt, kernel_size=kernel_size[3], padding='same', strides=1, activation='relu')(dropout_seq)
    c5 = Conv1D(filters=n_filt, kernel_size=kernel_size[4], padding='same', strides=1, activation='relu')(dropout_seq)

    merged_conv = concatenate([c1, c2, c3, c4, c5], axis=1)
    #######################################################################################

    #Pooling to reduce dimensions
    pooled = MaxPooling1D(pool_size=pool_size, strides=None, padding='same')(merged_conv)    
    #Denses
    dense_seq1 = Dense(number_neurons[0], activation='relu')(c1)
    dropout_seq1 = Dropout(drop_hid)(dense_seq1)
    dense_seq2 = Dense(number_neurons[1], activation='relu')(dropout_seq1)
    dropout_seq2 = Dropout(drop_hid)(dense_seq2)
    main_dense = Dense(n_class, activation=final_act)(dropout_seq2)
    model = Model(inputs=[input_seq], outputs=[main_dense])
    print(model.summary())
    
    adamm = Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adamm, metrics=['accuracy'])
    
    # saving the model
    file_model = os.path.join(absPath, 'data/', folder, 'model.h5')

    model.save(file_model)
    return model

def model_choice(architecture, task, folder, max_len, dict_size, n_neur, n_class, drop_per, drop_hid, final_act, 
                 n_filt=None, kernel_size=None, pool_size=None):
    """Choosing model architecture and defining model"""
    if architecture == "only_denses":
        if task == "task1/":
            model = building_2dense_model_task2(max_len, dict_size, n_neur, n_class, drop_per,
                                                drop_hid, final_act, folder)
        else:
            model = building_2dense_model_task2(max_len, dict_size, n_neur, n_class, 
                                                drop_per, drop_hid, final_act, folder)                   
    elif architecture == "conv_dense":
        if task == "task1/":
            model = building_1convdense_model_task1(max_len, dict_size, n_neur, n_class, 
                                                    drop_per, drop_hid, n_filt, kernel_size, final_act, folder)            
        else:
            model = building_1convdense_model_task2(max_len, dict_size, n_neur, n_class, drop_per, 
                                                    drop_hid, n_filt, kernel_size, final_act, folder)
    elif architecture == "stack_conv":
        if task == "task1/":
            building_stackconv_model_task1(max_len, dict_size, n_neur, n_class, drop_per, drop_hid, 
                                            n_filt, kernel_size, pool_size, final_act, folder)
        else:
            model = building_stackconv_model_task2(max_len, dict_size, n_neur, n_class, drop_per, 
                                                    drop_hid, n_filt, kernel_size, pool_size, final_act, folder)
    return model