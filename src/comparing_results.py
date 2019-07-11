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
    plt.title('%s- models training accuracy (holdout=%i)' %(task, k))
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
    plt.title('%s- models validation accuracy (holdout=%i)' %(task, k))
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
    plt.title('%s- models training loss (holdout=%i)' %(task, k))
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
    plt.title('%s- models validation loss (holdout=%i)' %(task, k))
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

def plotting_auc_acc_boxplots(df, folder, metrics, nfolds, task):
    """Plotting AUC/accuracy/scores on test values in boxplots"""
    if metrics == "auc":
        titlee = "%s - AUC (%i holdouts)" %(task, nfolds)
        filename = "aucs_comparison.pdf"
        x="variable"
    elif metrics == "accuracy":
        titlee = "%s - Test accuracy (%i holdouts)" %(task, nfolds)
        filename = "test_accuracy_comparison.pdf"
        x="variable"
    
    p = (ggplot(df, aes(x=x, y="value", fill=x))
         +geom_boxplot()
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
         +theme(figure_size=(12,16), aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=10),
                legend_text=element_text(size=10), strip_text_x = element_text(size=10), axis_text_x = element_blank())
         + ggtitle(titlee)
    )
    file_auc = ''.join(string for string in [absPath,'data/results/', folder])
    p.save(path = file_auc, format = 'pdf', dpi=300, filename=filename)
    return p

def plotting_ROC_curves(df, folder, nfolds, task):
    """Plotting ROC curves"""
    k = random.randint(0, nfolds-1)
    df = df.loc[df.index == k]
    fig = plt.figure(figsize=(12,9))
    lw = 3
    for i in df["variable"]:
        plt.plot(df.loc[df.variable==i, "fpr"].values[0], df.loc[df.variable==i, "tpr"].values[0], label=i)
    #plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.legend(prop={'size': 12})
    plt.title("%s - ROC curves (holdout=%i)" %(task,k), size=18)
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, 'ROC_curves.png'])
    plt.savefig(file_fig)
    plt.show()
    
def processing_metrics_results(df, list_paddings, folder, nfolds):
    """It process the saved metrics from the models and returns a dataframe with F1-Score, precision and recall and another dataframe with accuracy on test"""
    metrics, k = collecting_metrics_folds("resulting_metrics", list_paddings, folder, 3)
    accu = metrics.apply(lambda x: [y[0] for y in x])
    scores = metrics.apply(lambda x: [y[2] for y in x])
    
    #processing scores
    list_dfs = []
    for i,row in scores.iterrows():
        for pad in list_paddings:
            formatted = pd.DataFrame(scores.loc[0, pad]).transpose().reset_index()
            formatted.columns = ['class', 'f1-score', 'precision', 'recall', 'support']
            formatted['index'] = row.name
            formatted['type_padding'] = pad
            list_dfs.append(formatted)
    scores_final = pd.concat(list_dfs)
    scores_final = scores_final.drop("support", 1)
    scores_final = scores_final.melt(id_vars=["class", "index", "type_padding"])
    #processing test accuracy
    accu = accu.reset_index().melt(id_vars='index')
    return scores_final, accu

def plotting_scores_boxplots(df, folder, nfolds, task):
    """Plotting F1-score/precision/recall on test values in boxplots"""
    p = (ggplot(df, aes(x='model_type', y="value", fill='model_type'))
         +geom_boxplot()
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
         +theme(figure_size=(12,16), aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=10),
                legend_text=element_text(size=10), strip_text_x = element_text(size=10), axis_text_x = element_blank())
         + facet_grid("class~variable")
         + ggtitle("%s - performance metrics (%i holdouts)" %(task, nfolds))
    )
    file_met = ''.join(string for string in [absPath,'data/results/', folder])
    p.save(path = file_met, format = 'pdf', dpi=300, filename="scores.pdf")
    return p