from __future__ import print_function, absolute_import, division

import random
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import h5py
import pylab as pl
import glob
from plotnine import *

from sklearn import metrics
from collections import Counter

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

np.random.seed(8)
random.seed(8)

def collecting_metrics_folds(metrics, list_paddings, folder, n_fold):
    """Collecting all the metrics from the different k_folds into one"""
    metrics_dict = {}
    for i in list_paddings:
    #It doesn't make much sense to plot history of all the folds. We choose one and plot it
        if metrics == 'history':
            k = random.randint(0, n_fold-1) 
            file_metrics = ''.join(string for string in [absPath, 'data/results/', folder, i, '/', 
                                                         str(k), '/', metrics, '.pickle'])
            with open(file_metrics, "rb") as input_file:
                metrics_dict[i] = pickle.load(input_file)
        else:
            list_results = []
            for k in range(n_fold):
                file_metrics = ''.join(string for string in [absPath, 'data/results/', folder, i, '/', 
                                                         str(k), '/', metrics, '.pickle'])
                with open(file_metrics, "rb") as input_file:
                    #metrics_dict[i] = pickle.load(input_file)
                    list_results.append(pickle.load(input_file))
            metrics_dict[i] = list_results
    metrics_df = pd.DataFrame(metrics_dict)
    return metrics_df, k


def plotting_history(df, task, folder, k):
    """It doesn't make much sense to plot history of all the folds. We choose one and plot it"""
    history_df = df.transpose().reset_index(level=0)
    history_df.columns = ["model_type", 'acc', 'loss', 'val_acc', 'val_loss']
    
    fig = plt.figure()
    for i in history_df["model_type"]:
        plt.plot(history_df.loc[history_df.model_type==i, "acc"].values[0], label=i)
    plt.title('%s- models training accuracy (k=%i)' %(task, k))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, 'train_acc_comparison.png'])
    plt.savefig(file_fig)
    plt.show()
    
    # summarize history for val acc
    fig = plt.figure()
    for i in history_df["model_type"]:
        plt.plot(history_df.loc[history_df.model_type==i, "val_acc"].values[0], label=i)
    plt.title('%s- models validation accuracy (k=%i)' %(task, k))
    plt.ylabel('val accuracy')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, 'val_acc_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

    # summarize history for loss
    fig = plt.figure()
    for i in history_df["model_type"]:
        plt.plot(history_df.loc[history_df.model_type==i, "loss"].values[0], label=i)
    plt.title('%s- models training loss (k=%i)' %(task, k))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, 'train_loss_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

    # summarize history for validation loss
    fig = plt.figure()
    for i in history_df["model_type"]:
        plt.plot(history_df.loc[history_df.model_type==i, "val_loss"].values[0], label=i)
    plt.title('%s- models validation loss (k=%i)' %(task, k))
    plt.ylabel('val loss')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, 'val_loss_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

def processing_roc_auc(df, metrics):
    """Processing ROC curves and AUC to be plotted"""
    df= df.reset_index(0)
    df = df.melt(id_vars='index')
    if metrics == "roc":
        new_col_list = ['fpr','tpr','_']
        for n,col in enumerate(new_col_list):
            df[col] = df['value'].apply(lambda value: value[n])
        df = df.drop('value',axis=1)
    return df

