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

def collecting_metrics_folds(metrics, list_paddings, folder, task, n_fold):
    """Collecting all the metrics (history, auc, roc) from the different k_folds into one"""
    metrics_dict = {}
    for i in list_paddings:
    #It doesn't make much sense to plot history of all the folds. We choose one and plot it
        if metrics == 'history':
            k = random.randint(0, n_fold-1) 
            file_metrics = ''.join(string for string in [absPath, 'data/results/', folder, task, i, '/', 
                                                         str(k), '/', metrics, '.pickle'])
            with open(file_metrics, "rb") as input_file:
                metrics_dict[i] = pickle.load(input_file)
        else:
            list_results = []
            for k in range(n_fold):
                file_metrics = ''.join(string for string in [absPath, 'data/results/', folder, task, i, '/', 
                                                         str(k), '/', metrics, '.pickle'])
                with open(file_metrics, "rb") as input_file:
                    #metrics_dict[i] = pickle.load(input_file)
                    list_results.append(pickle.load(input_file))
            metrics_dict[i] = list_results
    metrics_df = pd.DataFrame(metrics_dict)
    return metrics_df, k


def plotting_history(df, task_string, folder, task, k):
    """It doesn't make much sense to plot history of all the folds. We choose one and plot it"""
    color = ["#8DD3C7", "#FFFFB3","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69",
         "#FCCDE5","#D9D9D9","#BC80BD","#CCEBC5","#FFED6F"]
    history_df = df.transpose().reset_index(level=0)
    history_df.columns = ["model_type", 'acc', 'loss', 'val_acc', 'val_loss']
    
    fig = plt.figure()
    for idx,i in enumerate(sorted(history_df["model_type"])):
        plt.plot(history_df.loc[history_df.model_type==i, "acc"].values[0], label=i, c=color[idx])
    plt.title('%s- models training accuracy (holdout=%i)' %(task_string, k))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'train_acc_comparison.png'])
    plt.savefig(file_fig)
    plt.show()
    
    # summarize history for val acc
    fig = plt.figure()
    for idx,i in enumerate(sorted(history_df["model_type"])):
        plt.plot(history_df.loc[history_df.model_type==i, "val_acc"].values[0], label=i, c=color[idx])
    plt.title('%s- models validation accuracy (holdout=%i)' %(task_string, k))
    plt.ylabel('val accuracy')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'val_acc_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

    # summarize history for loss
    fig = plt.figure()
    for idx,i in enumerate(sorted(history_df["model_type"])):
        plt.plot(history_df.loc[history_df.model_type==i, "loss"].values[0], label=i, c=color[idx])
    plt.title('%s- models training loss (holdout=%i)' %(task_string, k))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'train_loss_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

    # summarize history for validation loss
    fig = plt.figure()
    for idx,i in enumerate(sorted(history_df["model_type"])):
        plt.plot(history_df.loc[history_df.model_type==i, "val_loss"].values[0], label=i, c=color[idx])
    plt.title('%s- models validation loss (holdout=%i)' %(task_string, k))
    plt.ylabel('val loss')
    plt.xlabel('epoch')
    plt.legend()
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'val_loss_comparison.png'])
    plt.savefig(file_fig)
    plt.show()

def processing_roc_auc(df, metrics, list_paddings):
    """Processing ROC curves and AUC to be plotted"""
    df.columns = list_paddings
    df= df.reset_index(0)
    df = df.melt(id_vars='index')
    if metrics == "roc":
        new_col_list = ['fpr','tpr','_']
        for n,col in enumerate(new_col_list):
            df[col] = df['value'].apply(lambda value: value[n])
        df = df.drop('value',axis=1)
    return df

def plotting_auc_acc_boxplots(df, folder, metrics, nfolds, task_string, task):
    """Plotting AUC/accuracy/scores on test values in boxplots"""
    if metrics == "auc":
        titlee = "%s - AUC (%i holdouts)" %(task_string, nfolds)
        filename = "aucs_comparison.pdf"
        x = "variable"
    elif metrics == "accuracy":
        titlee = "%s - Accuracy on test (%i holdouts)" %(task_string, nfolds)
        filename = "test_accuracy_comparison.pdf"
        x = "variable"
    
    p = (ggplot(df, aes(x=x, y="value", fill=x))
         +geom_boxplot()
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
        +theme(aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=12),
                legend_text=element_text(size=14), strip_text_x = element_text(size=10), axis_text_x = element_blank(),
          legend_key_size = 14, axis_title_y=element_blank(), axis_title_x=element_blank(), 
           #legend_position="bottom", legend_box="horizontal", 
           plot_title = element_text(size=20))
         + ggtitle(titlee)
    )
    p
    file_auc = ''.join(string for string in [absPath,'data/results/', folder, task])
    p.save(path = file_auc, format = 'pdf', dpi=300, filename=filename)
    return p

def plotting_ROC_curves(df, folder, nfolds, task_string, list_paddings, task):
    """Plotting ROC curves"""
    k = random.randint(0, nfolds-1)
    df = df.loc[df["index"] == k]
    fig = plt.figure(figsize=(12,9))
    lw = 3
    color = ["#8DD3C7", "#FFFFB3","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69",
         "#FCCDE5","#D9D9D9","#BC80BD","#CCEBC5","#FFED6F"]
    for idx,i in enumerate(sorted(list_paddings)):
        plt.plot(df.loc[df.variable==i, "fpr"].values[0], df.loc[df.variable==i, "tpr"].values[0], label=i, lw=3,  c=color[idx])
    #plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.legend(prop={'size': 18})
    plt.title("%s - ROC curves (holdout=%i)" %(task_string,k), size=24)
    file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'ROC_curves.png'])
    plt.savefig(file_fig)
    plt.show()
    
def processing_metrics_results(list_paddings, folder, task, nfolds):
    """It process the saved metrics from the models and returns a dataframe with F1-Score, precision and recall and another dataframe with accuracy on test"""
    metrics, k = collecting_metrics_folds("resulting_metrics", list_paddings, folder, task, nfolds)
    accu = metrics.apply(lambda x: [y[0] for y in x])
    scores = metrics.apply(lambda x: [y[2] for y in x])
    
    #processing scores
    list_dfs = []
    for i,row in scores.iterrows():
        for pad in list_paddings:
            formatted = pd.DataFrame(scores.loc[i, pad]).transpose().reset_index()
            formatted.columns = ['enz_type', 'f1-score', 'precision', 'recall', 'support']
            formatted['index'] = row.name
            formatted['type_padding'] = pad
            list_dfs.append(formatted)
    scores_final = pd.concat(list_dfs)
    scores_final = scores_final.drop("support", 1)
    scores_final = scores_final.melt(id_vars=["enz_type", "index", "type_padding"])
    
    if task == "task2/":
        dicti_enz = {"0":"1", "1":"2", "2":"3", "3":"4", "4":"5", "5":"6", "6":"7", "macro avg": "macro avg", 
             "micro avg": "micro avg", "weighted avg": "weighted avg"}
        scores_final["enz_type"] = scores_final["enz_type"].apply(lambda x: dicti_enz[x])
    #processing test accuracy
    accu = accu.reset_index().melt(id_vars='index')
    return scores_final, accu
#revisar1!!
def plotting_scores_boxplots(df, folder, nfolds, task_string, task):
    """Plotting F1-score/precision/recall on test values in boxplots"""
    p = (ggplot(df, aes(x='type_padding', y="value", fill='type_padding'))
         +geom_boxplot(outlier_size=1)
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
         +theme(figure_size=(24,16), aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=9),
                legend_text=element_text(size=12), strip_text_x = element_text(size=9), axis_text_x = element_blank(),
               strip_text_y = element_text(size=9), plot_title = element_text(size=15), axis_title_y = element_blank(),
               axis_title_x = element_text(size = 9), legend_key_size = 12, legend_position="bottom", 
                legend_box="horizontal")
         + facet_grid("variable~enz_type")
         + ggtitle("%s - performance metrics (%i holdouts)" %(task_string, nfolds))
         + guides(fill=guide_legend(nrow=1))
    )
    file_met = ''.join(string for string in [absPath,'data/results/', folder, task])
    p.save(path = file_met, format = 'pdf', dpi=300, filename="scores.pdf")
    return p



########## comparing different architectures
def processing_metrics_dodge(list_paddings, folders, names_folders,task, nfolds):
    """It process the saved metrics from the models and returns a dataframe with F1-Score, precision and recall and another dataframe with accuracy on test"""
    scores_list = []
    accu_list = []
    for idx, folder in enumerate(folders):
        metrics, k = collecting_metrics_folds("resulting_metrics", list_paddings, folder, task, nfolds)
        accu = metrics.apply(lambda x: [y[0] for y in x])
        scores = metrics.apply(lambda x: [y[2] for y in x])
    
        #processing scores
        list_dfs = []
        for i,row in scores.iterrows():
            for pad in list_paddings:
                formatted = pd.DataFrame(scores.loc[i, pad]).transpose().reset_index()
                formatted.columns = ['enz_type', 'f1-score', 'precision', 'recall', 'support']
                formatted['index'] = row.name
                formatted['type_padding'] = pad
                list_dfs.append(formatted)
        scores_final = pd.concat(list_dfs)
        scores_final["architecture"] = names_folders[idx]
        scores_final = scores_final.drop("support", 1)
        scores_final = scores_final.melt(id_vars=["enz_type", "index", "type_padding", "architecture"])
    
        if task == "task2/":
            dicti_enz = {"0":"1", "1":"2", "2":"3", "3":"4", "4":"5", "5":"6", "6":"7", "macro avg": "macro avg", 
                 "micro avg": "micro avg", "weighted avg": "weighted avg"}
            scores_final["enz_type"] = scores_final["enz_type"].apply(lambda x: dicti_enz[x])
        #processing test accuracy
        accu = accu.reset_index().melt(id_vars='index')
        accu["architecture"] = names_folders[idx]
        scores_list.append(scores_final)
        accu_list.append(accu)
    return scores_list, accu_list


def plotting_acc_dodge_boxplots(df, nfolds, task_string, task):
    """Plotting AUC/accuracy/scores on test values in boxplots"""
    titlee = "%s - Accuracy on test (%i holdouts)" %(task_string, nfolds)
    filename = "test_accuracy_comparison_architectures.pdf"
    x = "variable"
    
    p = (ggplot(df, aes(x=x, y="value", fill="architecture"))
         +geom_boxplot(position="dodge")
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
        +theme(aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=12),
                legend_text=element_text(size=14), strip_text_x = element_text(size=10), 
               axis_text_x = element_text(angle = 90, hjust = 1),
          legend_key_size = 14, axis_title_y=element_blank(), 
               #axis_title_x=element_blank(), 
           #legend_position="bottom", legend_box="horizontal", 
           plot_title = element_text(size=20))
         + ggtitle(titlee)
    )
    p
    file_auc = ''.join(string for string in [absPath,'data/results/', task])
    if not os.path.exists(file_auc):
        os.makedirs(file_auc)
    p.save(path = file_auc, format = 'pdf', dpi=300, filename=filename)
    return p

def plotting_scores_arch(df, nfolds, task_string, task):
    """Plotting F1-score/precision/recall on test values in boxplots"""
    df = df[df.enz_type.isin(["micro avg", "macro avg", "weighted avg"])]
    p = (ggplot(df, aes(x='type_padding', y="value", fill='architecture'))
         +geom_boxplot(outlier_size=1, position="dodge")
         + scale_fill_brewer(palette="Set3", type='qual')
         +theme_bw()
         +theme(figure_size=(24,16), aspect_ratio=1, legend_title=element_blank(), axis_text_y =element_text(size=13),
                legend_text=element_text(size=13), strip_text_x = element_text(size=13), 
                axis_text_x = element_text(angle=90, hjust=1),
               strip_text_y = element_text(size=13), plot_title = element_text(size=20), axis_title_y = element_blank(),
               axis_title_x = element_text(size = 13), legend_key_size = 13, #legend_position="bottom", 
                legend_box="horizontal")
         + facet_grid("variable~enz_type")
         + ggtitle("%s - performance metrics (%i holdouts)" %(task_string, nfolds))
         + guides(fill=guide_legend(nrow=1))
    )
    
    file_met = ''.join(string for string in [absPath,'data/results/', task])
    if not os.path.exists(file_met):
        os.makedirs(file_met)
    p.save(path = file_met, format = 'pdf', dpi=300, filename="scores_architecture.pdf")
    return p