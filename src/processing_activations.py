from __future__ import print_function, absolute_import, division

import random
import os
import pickle
import sys

import h5py
import pickle
import numpy as np
import keras
import pandas as pd
import random
import time

from glob import glob
from keras.models import load_model
from keras import models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime
from plotnine import *

#root
absPath = '/home/angela/padding_benchmark/'
sys.path.insert(0, absPath)

from src.Target import Target

#para no imprimir warnings chungos
import warnings 
warnings.simplefilter("ignore")

np.random.seed(1)
random.seed(1)

def search_model (folder, fold, task, padding):
    model_path = "".join((absPath, "data/checkpoint/", folder, task, padding, "/", str(fold), "/"))
    model_file = glob(os.path.join(model_path, "weights-*.hdf5"))[0]
    return model_file

def random_prot_idx(folder, task):
    if task == "task1":
        file_idcs = os.path.join(absPath, 'data/', folder, 'idcs_split.pickle')
    else:
        file_idcs = os.path.join(absPath, 'data/', folder, 'idcs_data_enz.pickle')
    with open(file_idcs, "rb") as input_file:
        splitting_sets = pickle.load(input_file)[0]
    i_train, i_val, i_test = splitting_sets
    len_test = len(i_test)
    rnd_index = random.randint(0, len_test-1)
    protein_index = i_test[rnd_index]
    return protein_index

def load_prot(folder, protein_index, padding, labels_label, newshape):
    data_path = os.path.join(absPath, "data/", folder, "data.h5")
    f1 = h5py.File(data_path, 'r')
    prot = f1[padding][protein_index]
    label_prot = f1[labels_label][protein_index]
    instarget = Target('AAAAAA')
    aa_to_int = instarget.predefining_dict()
    seq_onehot = instarget.int_to_onehot(list(prot), len(aa_to_int))
    seq_reshaped = np.reshape(seq_onehot, newshape=newshape)
    return seq_reshaped, label_prot

def obtain_activations(model_path, n_layer, seq_reshaped):
    model = load_model(model_path)
    layer_output = model.layers[n_layer].output
    activation_model = models.Model(inputs=model.input, outputs=layer_output) 
    activations = activation_model.predict(seq_reshaped, batch_size=1)
    shape_act = activations.shape
    activations_reshaped = np.reshape(activations, newshape=(shape_act[2], shape_act[1]))
    #print(activations_reshaped.shape)
    return activations_reshaped

def activations_to_df(activations, padding, fold):
    activations_df = pd.DataFrame(activations)
    activations_df["padding"] = padding
    activations_df["order"] = list(range(0,activations.shape[0]))
    activations_df["fold"] = fold
    #label_prot_noonehot = np.argmax(label_prot, -1)
    #activations_df["label"] = label_prot_noonehot
    return activations_df

def train_pca(list_dfs, ncomps):
    df_joint = pd.concat(list_dfs, ignore_index=True)
    x = df_joint.drop(['order', 'padding', 'fold', "enzyme_type",
                       #, 'label'
                      ], axis=1).values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=ncomps)
    pca.fit(x)
    principalComponents = pca.fit_transform(x)
    principal_df = pd.DataFrame(principalComponents, columns=["PC1", "PC2"])
    final_df = pd.concat([principal_df, df_joint[['padding']], df_joint[['order']],df_joint[['fold']], 
                          df_joint[["enzyme_type"]]], 
                          axis=1)
    return final_df, pca

def plot_pca(df, pca, type_plot, list_paddings=None):
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(1,1,1) 
    label_x = " ".join(("Principal Component 1 (", str(round(pca.explained_variance_ratio_[0]*100, 2)), "%)"))
    label_y = " ".join(("Principal Component 2 (", str(round(pca.explained_variance_ratio_[1]*100, 2)), "%)"))
    ax.set_xlabel(label_x, fontsize = 11)
    ax.set_ylabel(label_y, fontsize = 11)
    ax.set_title('PCA of Conv1D (size=5)', fontsize = 14)
    
    if type_plot == "padding":
        colors = ["#FFED6F","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69",
             "#FCCDE5"]
        for padding, color in zip(list_paddings,colors):
            indicesToKeep = df['padding'] == padding
            print(padding, df.loc[indicesToKeep, 'PC1'].shape)
            ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , c = color
                   , s = 25,
                      alpha=1)
        ax.legend(list_paddings, title = "Padding type")
        ax.grid()
    elif type_plot == "enzyme_type":
        types_enzymes = list(range(1,8))
        colors = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494"]
        for enz_type, color in zip(types_enzymes,colors):
            indicesToKeep = df['enzyme_type'] == enz_type
            #print(enz_type, df.loc[indicesToKeep, 'PC1'].shape)
            ax.scatter(df.loc[indicesToKeep, 'PC1']
               , df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 25,
                  alpha=0.8)
        ax.legend(types_enzymes, title = "Enzyme type")
        ax.grid()
        
    elif type_plot == "enz_type1":
        types_enzymes = [0,1]
        colors = ["#6CB89E", "#F38E73"]
        for enz_type, color in zip(types_enzymes, colors):
            indicesToKeep = df['enzyme_type'] == enz_type
            #print(enz_type, df.loc[indicesToKeep, 'PC1'].shape)
            ax.scatter(df.loc[indicesToKeep, 'PC1']
               , df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 25,
                  alpha=0.8)
        ax.legend(types_enzymes, title = "Enzyme type")
        ax.grid()
    else:
        ax.scatter(df.loc[:, 'PC1'], df.loc[:, 'PC2'], 
                   c=df.loc[:, 'sequence_length'], edgecolor='none', alpha=0.7)
        im = plt.gca().get_children()[0]
        cax = fig.add_axes([1,0.1,0.03,0.8]) 
        clb=fig.colorbar(im, cax=cax)  
        clb.ax.set_title("Protein sequence length")
        ax.grid()
    #file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'pcaActivations_', typle_plot.png'])
    #plt.savefig(file_fig)
    
#function to select proteins of the test set with an specific enzimatic label
def find_prot_labeled(folder, enz_label, quant=2, task = "task2", labels_label = "labels_task2"):
    data_path = os.path.join(absPath, "data/", folder, "data.h5")
    f1 = h5py.File(data_path, 'r')
    indices_enzymes = []
    while len(indices_enzymes)<quant:
        idx = random_prot_idx(folder, task)
        if task == "task2":
            label_prot = np.argmax(f1[labels_label][idx]) + 1
        else:
            label_prot = np.argmax(f1[labels_label][idx])
        if label_prot == enz_label:
            indices_enzymes.append(idx)
    return indices_enzymes

def load_prot_labeled(folder, padding, newshape, enz_label, idx):
    data_path = os.path.join(absPath, "data/", folder, "data.h5")
    f1 = h5py.File(data_path, 'r')
    instarget = Target('AAAAAA')
    aa_to_int = instarget.predefining_dict()
    prot = f1[padding][idx]
    seq_onehot = instarget.int_to_onehot(list(prot), len(aa_to_int))
    seq_reshaped = np.reshape(seq_onehot, newshape=newshape)
    return seq_reshaped

def plot_pca_per_padding(final_df, pca, plot_by = "factor(enzyme_type)"):
    p = (ggplot(final_df, aes(x="PC1", y="PC2", color=plot_by))
    + geom_jitter(alpha=0.8, size=1)
     #+ scale_color_brewer(palette="Set2", type='qual', name="enzyme type")
    + theme_bw()
    + facet_wrap("~padding", nrow=4, ncol=2)
    +theme(figure_size=(8,8), axis_text_y =element_text(size=9),
                legend_text=element_text(size=12), strip_text_x = element_text(size=9), 
               strip_text_y = element_text(size=9), plot_title = element_text(size=14), 
           #axis_title_y = element_blank(),
               axis_title_x = element_text(size = 11), legend_key_size = 12,
           axis_title_y = element_text(size = 11),
                legend_box="horizontal")
    +ggtitle('PCA of Conv1D (size=5)')
    + xlab(" ".join(("Principal Component 1 (", str(round(pca.explained_variance_ratio_[0]*100, 2)), "%)")))
    +ylab(" ".join(("Principal Component 2 (", str(round(pca.explained_variance_ratio_[1]*100, 2)), "%)"))))
    
    if plot_by == "factor(enzyme_type)":
        q = p + scale_color_brewer(palette="Set2", type='qual', name="enzyme type")
    else:
        q = p
    return q

def density_facet(final_df, pca):
    p = (ggplot(final_df, aes(x="PC1", y="PC2", fill="factor(enzyme_type)"))
    #+ geom_density_2d(aes(fill="..level..", levels=10, alpha=0.8))
    # +geom_point()
     + stat_density_2d(aes(alpha="stat(level)"), color="black", geom="polygon")
     + scale_fill_brewer(palette="Set2", type='qual', name="enzyme type")
     #+ scale_fill_gradient(low = "")
    + theme_bw()
     + facet_grid("enzyme_type~padding")
    #+ facet_grid("~padding", nrow=4, ncol=2)
    +theme(figure_size=(10,10), axis_text_y =element_text(size=9),
                legend_text=element_text(size=12), strip_text_x = element_text(size=9), 
       #axis_text_x = element_blank(),
               strip_text_y = element_text(size=9), plot_title = element_text(size=14), 
           #axis_title_y = element_blank(),
               axis_title_x = element_text(size = 11), legend_key_size = 12,
           axis_title_y = element_text(size = 11),
       #legend_position="bottom", 
                legend_box="horizontal")
    +ggtitle('PCA of Conv1D (size=5)')
    + xlab(" ".join(("Principal Component 1 (", str(round(pca.explained_variance_ratio_[0]*100, 2)), "%)")))
    +ylab(" ".join(("Principal Component 2 (", str(round(pca.explained_variance_ratio_[1]*100, 2)), "%)"))))
    return p

#pintar loadings
def plot_loadings(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.figure(figsize=(10,10))
    #plt.scatter(xs * scalex,ys * scaley, 
    #           # c = y
    #           )
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-0.1,0.1)
    plt.ylim(-0.1,0.1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

def load_prot_aminoacids(folder, protein_index, padding, labels_label):
    data_path = os.path.join(absPath, "data/", folder, "data.h5")
    f1 = h5py.File(data_path, 'r')
    prot = f1[padding][protein_index]
    instarget = Target('AAAAAA')
    seq_int = f1[padding][protein_index]
    aa_to_int = instarget.predefining_dict()
    sequence_aas = instarget.int_to_aa(seq_int, aa_to_int)
    label_prot = f1[labels_label][protein_index]
    return sequence_aas, label_prot

def add_sequences_length(final_df, df_indices):
    len_len_seq = int((len(final_df)/7)/2)
    print(len_len_seq)
    lista_lengths = []
    for idx, row in df_indices.iterrows():
        len_seq = row["len_seq"]
        lista_lengths.append([len_seq]*len_len_seq)
    lista_lengths_unnested = [item for sublist in lista_lengths for item in sublist]
    print(len(lista_lengths_unnested))
    print(final_df.info())
    final_df["sequence_length"] = lista_lengths_unnested
    return final_df

def myplot(score,coeff,labels=None):
    plt.figure(figsize=(10,10))
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    #plt.scatter(xs * scalex,ys * scaley, 
    #            #c = y
    #           )
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-0.1,0.1)
    plt.ylim(-0.1,0.1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    
def saving_activations_plot(packagee, type_plot, folder, task, p=None):
    if packagee == "plt":
        extension = ".png"
        file_fig = ''.join(string for string in [absPath,'data/results/', folder, task, 'pcaActivations_',
                                         type_plot, extension])
        plt.savefig(file_fig)
    else:
        extension = ".pdf"
        path_fig = ''.join(string for string in [absPath,'data/results/', folder, task])
        file_fig = ''.join(string for string in ['pcaActivations_', type_plot, extension])
        p.save(path = path_fig, format = 'pdf', dpi=300, filename=file_fig)