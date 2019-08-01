from __future__ import print_function, absolute_import, division

import re
import math
import random
import time
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
import bisect

import tensorflow as tf
import keras
from keras.models import load_model
from sklearn import metrics
from collections import Counter

#root
absPath = '/home/angela/padding_EBI/'
sys.path.insert(0, absPath)

from src.Target import Target
from src.preprocessing import *

np.random.seed(8)
random.seed(8)

def plot_history(path_history):
    """Plot evolution of accuracy and loss both in training and in validation sets"""
    #file_his = ''.join(string for string in [absPath, 'data/results/', folder, '/', model_type, '/history.pickle'])
    file_his = ''.join(string for string in [path_history, '/history.pickle'])
    with open(file_his, "rb") as input_file:
        history = pickle.load(input_file)

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    file_fig = ''.join(string for string in [path_history, '/history_acc.png'])
    plt.savefig(file_fig)
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()    
    file_fig = ''.join(string for string in [path_history, '/history_loss.png'])
    plt.savefig(file_fig)
    return history
    
def load_best_model(history, path_to_cp):
    """Looks for the best epoch in terms of validation accuracy and load the corresponding model"""
    #which one is the best epoch?
    best_epoch = str(history['val_acc'].index(max(history['val_acc'])) +1).zfill(3)
    print('best epoch: ', best_epoch)
    cp_path = ''.join(string for string in [path_to_cp, '/weights-improvement-', best_epoch, '*.hdf5'])
    cp_path = glob.glob(cp_path)[0]
    model = load_model(cp_path)
    return model, cp_path

def predict_on_test(data_file, model_type, x_name, labels, i_test, model):
    """Load test data and predict on it"""
    #load data
    file_data = os.path.join(absPath, 'data/', data_file)
    
    h5f = h5py.File(file_data, 'r')
    x_test = h5f[model_type][sorted(i_test)]
    y_test = h5f[labels][sorted(i_test)]
    
    #prediction
    y_predprob = model.predict(x_test)
    y_pred = y_predprob.argmax(axis=-1)
    y_test_scalar = y_test.argmax(axis=-1)
    y_prob = y_predprob[:,1]
    return y_predprob, y_pred, y_test_scalar, y_prob

def confusion_matrix(y_test_scalar, y_pred, path_to_confusion):
    """Creating a confusion matrix and saving it"""
    #model report
    print ("\nModel Report")
    print ("Accuracy (test set): %.4g" % metrics.accuracy_score(y_test_scalar, y_pred))
    print("Confusion matrix:")
    print (metrics.confusion_matrix(y_test_scalar, y_pred))
    print("Detailed classification report:")
    print (metrics.classification_report(y_test_scalar, y_pred))

    #Saving metrics 
    #file_out = ''.join(string for string in [absPath, 'data/checkpoint/',folder, '/', model_type, '/resulting_metrics.pickle'])
    file_out = ''.join(string for string in [path_to_confusion, '/resulting_metrics.pickle'])
    d = (metrics.accuracy_score(y_test_scalar, y_pred), metrics.confusion_matrix(y_test_scalar, y_pred), 
     metrics.classification_report(y_test_scalar, y_pred, output_dict=True)) 

    with open(file_out, "wb") as output_file:
        pickle.dump(d, output_file)

def compute_roc(y_test_scalar, y_prob, path_to_auc):
    """Computing ROC curve and plotting it"""
    #Print model report:
    print ("\nModel Report II part")
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(y_test_scalar, y_prob))   

    #Saving metrics 
    #file_auc = ''.join(string for string in [absPath, 'data/checkpoint/', folder, '/', model_type, '/auc.pickle']) 
    file_roc = ''.join(string for string in [path_to_auc, '/roc.pickle'])
    with open(file_roc, "wb") as output_file:
        pickle.dump(metrics.roc_curve(y_test_scalar, y_prob), output_file)
        
    file_auc = ''.join(string for string in [path_to_auc, '/auc.pickle'])
    with open(file_auc, "wb") as output_file:
        pickle.dump(metrics.roc_auc_score(y_test_scalar, y_prob), output_file)
    
    # Computing ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test_scalar, y_prob)
    fig = plt.figure(figsize=(9,7))
    lw = 3
    plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title("ROC Curve w/ AUC=%s" % str(metrics.auc(fpr,tpr)), fontsize = 18)
    plt.show()
    file_fig = ''.join(string for string in [path_to_auc, '/auc.png'])
    plt.savefig(file_fig)


#Functions to compute and plot AUC
def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    """Computing AUC from FPR and TPR"""
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in list(zip(ft[: -1], ft[1: ])):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    """Computing new FPR and TPR specified by threshold"""
    p = bisect.bisect_left(fpr, thresh)
    fpr = fpr.copy()
    fpr[p] = thresh
    return fpr[: p + 1], tpr[: p + 1]

def computing_partial_auc(y_test_scalar, y_prob, folder, thresh=0.05):
    #fpr, tpr, thresh, trapezoid=False):
    fpr, tpr, _ = metrics.roc_curve(y_test_scalar, y_prob)
    """Computing partial AUC at a given threshold"""
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, thresh)
    part_auc_notrapez = auc_from_fpr_tpr(fpr_thresh, tpr_thresh)
    part_auc_trapez = auc_from_fpr_tpr(fpr_thresh, tpr_thresh, True)
    print("Partial AUC:", part_auc_notrapez, part_auc_trapez)
    
    #Saving partial AUC
    #file_pauc = ''.join(string for string in [absPath, 'data/results/', folder, '/', model_type, '/pauc.pickle']) 
    file_pauc = ''.join(string for string in [folder, '/pauc.pickle'])

    with open(file_pauc, "wb") as output_file:
        pickle.dump(part_auc_notrapez, output_file)
    return part_auc_notrapez, part_auc_trapez

#Functions for giffing weights
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
    
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def giffing_weights(folder, layers, model_type):
    """Create gifs from weights on each epoch"""
    #Processing weights
    file_weights = file = ''.join(string for string in [absPath, 'data/weights/', folder, '/', model_type, '/model_weights.h5'])
    f = h5py.File(file_weights, 'r')
    names_epoch = ["epoch_"+str(i).zfill(3) for i in range(0, epochss+1)]
    
    # Iterating over layers in layers and saving the images for gifs (convolutionals)
    for i in layers:
        namee = i.split('_')[0]
        print("W shape : ", f[i][ u'weights_0'].shape)
        for index, j in enumerate(f[i]):
            layer_weights = f[i][j]
            w = np.squeeze(layer_weights)
            if namee == 'Dense':
                w = np.swapaxes(w,0,1)
            else:
                w = np.swapaxes(w,0,2)
            fig = pl.figure(figsize=(15, 15))
            pl.title('%s_%s' % (i, names_epoch[index]))
            if namee == 'Dense':
                ax = pl.gca()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", 1.2, pad=0.1)
        
                im = ax.imshow(w, vmin=-1, vmax=1, interpolation='nearest', cmap=cm.bwr)
                pl.colorbar(im, cax=cax)
                fig
            else:
                nice_imshow(pl.gca(), make_mosaic(w, 5, 2), vmin=-0.5, vmax=0.5, cmap=cm.bwr)
            if not os.path.exists(''.join(string for string in [absPath, 'data/weights/', folder, '/', model_type, '/gifs/', i])):
                os.makedirs(''.join(string for string in [absPath, 'data/weights/', folder, '/', model_type, '/gifs/', i]))
            fig.savefig(os.path.join(''.join(string for string in [absPath, 'data/weights/', folder, '/', model_type, '/gifs/', i, '/', i, '_', names_epoch[index], '.png'])))
            pl.close(fig)
        #pathh = os.path.join(''.join(string for string in [absPath, 'data/weights/', folder, '/', model_type, '/gifs/', i, '/*.png']))
        #pathh_gif = os.path.join(''.join(string for string in [absPath, 'data/weights/',  folder, '/', model_type, '/gifs/', i, '/', i, '.gif']))
        #! convert -delay 40 "{pathh}" "{pathh_gif}"                     

def processing_results(folder, task, model_type, idx, i_test, labels):
    """Processing results (all together)"""
    dicti = creating_dict()
    his_folder = ''.join(string for string in [absPath, 'data/results/', folder, task, model_type, 
                                                   '/', str(idx)])
    history = plot_history(his_folder)
    path_to_cp = ''.join(string for string in [absPath, 'data/checkpoint/', folder, task, model_type, 
                                            '/', str(idx)])
    model, best_path = load_best_model(history, path_to_cp)
    cps_loc = ''.join(string for string in [absPath, 'data/checkpoint/', folder, task, model_type, 
                                            '/', str(idx), '/*.hdf5'])
    #removing the rest of weights
    fileList = glob.glob(cps_loc, recursive=True)
    fileList.remove(best_path)
    if len(fileList) >1:
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")
    #loading test data        
    if model_type == "aug_padding":
        file_data = os.path.join(absPath, 'data/', folder, 'aug_data.h5')
    else:
        file_data = os.path.join(absPath, 'data/', folder, 'data.h5')
    h5f = h5py.File(file_data, 'r')
        
    instarget = Target('AAAAAA')
    x_test = h5f[model_type][sorted(i_test)]
    x_test = instarget.int_to_onehot(list(x_test), len(dicti))
    y_test = h5f[labels][sorted(i_test)]
        
    #predicting 
    y_predprob = model.predict(x_test)
    y_pred = y_predprob.argmax(axis=-1)
    y_test_scalar = y_test.argmax(axis=-1)
    y_prob = y_predprob[:,1]
    
    print(Counter(y_pred))
        
    #confusion matrix
    confusion_matrix(y_test_scalar, y_pred, his_folder)
        
    if task == "task1/":
        #AUC
        file_auc = ''.join(string for string in [his_folder, '/AUC.pickle'])
        compute_roc(y_test_scalar, y_prob, his_folder)
    
        computing_partial_auc(y_test_scalar, y_prob, his_folder)
    