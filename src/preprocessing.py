from __future__ import division, absolute_import

import re
import math
import random
import time
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import collections

from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.model_selection import StratifiedKFold

#root
absPath = '/home/angela/padding_uniprot/'
sys.path.insert(0, absPath)

from src.Target import Target

np.random.seed(8)
random.seed(8)

def loading_data(file_in, column, to_string=True):
    """ Loads data from file_in path, converts column to string for further pre-processing"""
    df = pd.read_table(file_in,compression='gzip',sep='\t')
    #converting column of interest to string
    if to_string == True:
        df[column] = df[column].apply(str)
    return df

def looking_max_len(df, quan=0.9):
    """ Get the maximum length of the sequences present in df, compute the histogram of the lengths, compute 
    the length at which the coverage is quan"""
    max_len = df.Sequence.map(lambda x: len(x)).max()
    #Plots sequence length histogram
    seq_len = df['Sequence'].str.len()
    seq_len.hist(bins=40)
    #we compute the sequence length which covers quan percentage of the proteins
    quan_len = seq_len.quantile(quan)
    return max_len, quan_len

def filtering_over_maxlen(df, max_len):
    """"Filter sequences over the established max_len"""
    mask = (df['Sequence'].str.len() < max_len)
    df_filt = df.loc[mask,:].reset_index().drop(["index"], axis=1)
    return df_filt

def creating_dict():
    """ Create dictionary of amino acids"""
    instarget = Target('AAAAA')
    aa_to_int = instarget.predefining_dict()
    return aa_to_int

#vamos a dejar el onehot para el batch generator pot si hay muchos datos
def processing_sequences(df, type_padding):
    """ Processing amino acid sequences to an array of padded integers"""
    #Converting sequence to Target
    df['target']= df['Sequence'].apply(lambda x: Target(x))
    #Creating one target for getting the dictionary
    instarget = Target('AAAAA')
    #1st: we pad the sequences to the chosen max_len, with the strategy defined in type_padding
    seqs_padded = df['target'].apply(lambda x: x.padding_seq_position(max_len, type_padding))
    # 2nd: we convert amino acid sequences to integer sequences
    seqs_int = [instarget.string_to_int(x, aa_to_int) for x in seqs_padded]
    #this is simply to convert it to an array, I am NOT padding again
    seqs_int_array = sequence.pad_sequences(sequences=seqs_int_post, maxlen=max_len)
    return seqs_int_array

def splitting_sets(training_split, val_split, data, labels, folder, kfold_bool, n_splits=None):
    """"Defining indices for splitting between training, validation and test sets
    It can be chosen if splitting with (stratified) KFold or not. val_split is respect 
    to 1-training_split"""
    indices = list(np.arange(len(data)))
    if kfold_bool == False:
        #this has to be done twice in order to separate in training/validation and test
        x_train, x_valtest, y_train, y_valtest, idx_train, idx_valtest = train_test_split(data, 
                                                                                      labels, indices, 
                                                                                      test_size=(1-training_split))
        x_val, x_test, y_val, y_test, idx_val, idx_test = train_test_split(x_valtest, y_valtest, idx_valtest,
                                                                           test_size=val_split)
        file_idcs = os.path.join(absPath, 'data/', folder, 'idcs_split.pickle')
        with open(file_idcs, "wb") as output_file:
            pickle.dump((idx_train, idx_val, idx_test), output_file)
    else:
        k_indices = []
        #first I split between training and test+validation sets
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=(1-training_split), random_state=8)
        for train_index, valtest_index in sss.split(data, labels):
            X_valtest, y_valtest = data[valtest_index], labels[valtest_index]
            #then I define a new splitter for dividing test+validation (n_splits=1)
            ooo = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=0)
            val_index, test_index = ooo.split(X_valtest, y_valtest)
            real_val_idx = [valtest_index[i] for i in val_index]
            real_test_idx = [valtest_index[i] for i in test_index]
            k_indices.append((train_index, real_val_idx, real_test_idx))
            file_idcs = os.path.join(absPath, 'data/', folder, 'idcs_split.pickle')
            with open(file_idcs, "wb") as output_file:
                pickle.dump(k_indices, output_file)

def creating_augmented_data(vars_padding, labels_task1, indices, labels_task2 = None):
    """creating augmented data from all the types of padding from a specific set of indices"""
    #creating empty list for sequences and labels
    seqs, lbl_task1, lbl_task2 = [],[],[]
    #joining the paddings
    for idx in indices:
        for padding_type in vars_padding:
            seqs.append(padding_type[idx]), lbl_task1.append(labels_task1[idx])
            if labels_task2 != None:
                lbl_task2.append(labels_task2[idx])
    #now list must be shuffled
    index_shuf = list(range(len(indices)))
    random.shuffle(index_shuf)
    seqs_shuf = np.array([seqs[i] for i in index_shuf])
    task1_shuf = np.array([lbl_task1[i] for i in index_shuf])
    if labels_task2 != None:
        task2_shuf = np.array([lbl_task2[i] for i in index_shuf])
        return seqs_shuf, task1_shuf, task2_shuf
    else:
        return seqs_shuf, task1_shuf, task2_shuf        

def data_to_hdf5(saving_path, list_x, list_vars, labels_task1=None, labels_task2=None):
    """ Saving encoded data to HDF5"""
    if len(list_x) != len(list_vars):
        print("list_x and list_vars should have the same length")
    else:
        file_h5 = os.path.join(absPath, 'data/', saving_path)
        h5_bin = h5py.File(file_h5, 'w')
        for i in range(len(list_x)):
            h5_bin.create_dataset(list_x[i], data=list_vars[i])
        if labels_task1 != None:
            h5_bin.create_dataset('labels_task1', data=labels_task1)
        if labels_task2 != None:
            h5_bin.create_dataset('labels_task2', data=labels_task2)
        h5_bin.close()
            
        
    
################# EC number functions

def binarizing_EC(df):
    """ From the EC number, creates an enzyme label: 0.0 if no enzyme (no EC number), else 1.0"""
    df.loc[df['EC number'] == "nan", 'enzyme'] = 0.0
    df.loc[df['EC number'] != "nan", 'enzyme'] = 1.0
    print(df["enzyme"].value_counts())
    return df

#this one is not finished (looking at lapsus 41)
def first_digit_EC(df):
    """ From the EC number, creates a digit1 label with only the first digit of the EC number"""
    df['digit1'] = [row['EC number'].split('; ') for _, row in df.iterrows()]
    df['digit1'] = [[i.split('.')[0] for i in row['digit1']] for _, row in df.iterrows()]
    #if it is always the same number: keep it only once. 
    df['digit1'] = [list(set(row['digit1'])) for _, row in df.iterrows()]

    
def counting_multilabel(df):
    """ We count and plot histogram of the first digit of the EC number"""
    #How many instances have more than one label?
    numlabels = [len(row['digit1']) for _, row in df.iterrows()] 
    print("There are ", sum(float(num) >1 for num in numlabels), "samples with more than one label")
    # we count how many instances of each EC number
    labels_separated = df.digit1.apply(pd.Series)
    #first label
    first_label = dict(Counter(labels_separated.loc[:,0]))
    #second label (most of instances don't have a second label so we have to filter nans)
    second_label = dict(Counter(labels_separated.loc[:,1]))
    second_label.pop(np.nan, None)
    #Joining all the labels to plot an histogram
    new_dict = { k: first_label.get(k, 0) + second_label.get(k, 0) for k in set(first_label) | set(second_label) }
    unique_ecs = list(labels_separated[0].unique())
    print("The unique labels are ", unique_ecs)
    plt.bar(list(new_dict.keys()), new_dict.values(), color='g')
    plt.title("Histogram of firsts digits of EC number (nan are not enzymes)")
    plt.show()

def encoding_as_multilabel(df, folder):
    """ Encoding first digit of EC number in one hot encoding"""
    #keeping indices of the enzymes 
    idcs_enz = [idx for idx,i in enumerate(list(df.digit1)) if i[0]!='nan']
    #filtering proteins that do not have EC number
    ec_filtered = pd.Series([i for i in list(df.digit1) if i[0]!='nan'])
    #encoding labels 
    mlb = MultiLabelBinarizer()
    ec_multi = mlb.fit_transform(ec_filtered)
    print(mlb.classes_.shape)    
    # Saving max_len to a pickle
    file_mlb = os.path.join(absPath, 'data/', folder, 'mlb.pickle')
    with open(file_mlb, "wb") as output_file:
        pickle.dump(mlb, output_file)
    #mixing nans with encoded labels for task2
    new_list = []
    counter = 0
    for idx,i in enumerate(list(range(len(archaea_filt.digit1)))):
        if idx not in idcs_enz:
            new_list.append(np.array([0, 0, 0, 0, 0, 0, 0]))
        else:
            new_list.append(list(ec_multi)[counter])
            counter = counter+1
    ec_multilabeled =  np.array(new_list).astype("float64")
    print("Shape of the resulting encoding", ec_multilabeled.shape)
    return ec_multilabeled


# faltan las funciones para cuando solo hay enzimas
def keeping_indices_enzymes(df, idx_train, idx_val, idx_test, folder, kfold=False):
    """Re-creates again training, validation and test sets only with enzymes"""
    #keeping only indices from enzymes
    indices_enzyme = [idx for idx, x in enumerate(list(df["enzyme"])) if x==1.0]
    #keeping only enzyme indices for each set
    enz_idx_train = [i for i in idx_train if i in indices_enzyme]
    enz_idx_val = [i for i in idx_val if i in indices_enzyme]
    enz_idx_test = [i for i in idx_test if i in indices_enzyme]
    
    file_idcs_new = os.path.join(absPath, 'data/', folder, 'idcs_split_enzyme.pickle')
    if kfold == False:
        with open(file_idcs_new, "wb") as output_file:
            pickle.dump((enz_idx_train, enz_idx_val, enz_idx_test), output_file)
    else:
        return (enz_idx_train, enz_idx_val, enz_idx_test)
    
def keeping_indices_enzymes_recursively(list_kfolds, df):
    """If there are k sets of indices, goes iteratively over them to keep only the corresponding to enzymes"""
    enzymes_indices = []
    for i in list_kfolds:
        idx_train, idx_val, idx_test = i
        enzymes_indices.append(keeping_indices_enzymes(df, idx_train, idx_val, idx_test, kfold=True))
        file_idcs_new = os.path.join(absPath, 'data/', folder, 'idcs_split_enzyme.pickle')
        with open(file_idcs_new, "wb") as output_file:
            pickle.dump(enzymes_indices, output_file)
            
def keeping_indices_enzymes_augmented(data, labels_task1, labels_task2):
    """Keeping only enzymes indices for a certain set of augmented data"""
    idx_enz = [idx for idx,x in enumerate(list(labels_task1.argmax(-1))) if x==1.0]
    return idx_enz