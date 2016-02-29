#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json, pickle
from sklearn.metrics import confusion_matrix
import codecs



def plot_confusion_matrix(raga_name_map_file, result_file, outputname):
    
    raga_name_map = json.load(open(raga_name_map_file,'r'))
    results = pickle.load(open(result_file, 'r'))
    labels = results['var1']['gt_label']
    predictions = results['var1']['pred_label']
    u_labels = np.unique(labels)
    conf_arr = confusion_matrix(labels, predictions, labels = u_labels)
    raga_names = u_labels
    
    y_labels = []
    x_labels = []
    for ii, r in enumerate(raga_names):
        y_labels.append('R'+str(ii+1))
        x_labels.append(y_labels[-1] + '-'+raga_name_map[r])
    
    width = len(conf_arr)
    height = len(conf_arr[0])
 
    fig = plt.figure(figsize=(14,14))
    #fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.grid(which='major')
    cmap_local = plt.get_cmap('binary', np.max(conf_arr)-np.min(conf_arr)+1)
    res = ax.matshow(np.array(conf_arr), #cmap=plt.cm.binary, 
                    interpolation='nearest', aspect='1', cmap=cmap_local,
                    ##Commenting out this line sets labels correctly,
                    ##but the grid is off
                    extent=[0, width, height, 0], vmin =np.min(conf_arr)-.5, vmax = np.max(conf_arr)+0.5,
                    )
    ticks = np.arange(np.min(conf_arr),np.max(conf_arr)+1)
    tickpos = np.linspace(ticks[0] , ticks[-1] , len(ticks));
    #cax = plt.colorbar(mat, ticks=tickpos)
    #cax.set_ticklabels(ticks)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.4)
    cb = fig.colorbar(res, cax=cax, orientation = 'horizontal', ticks=tickpos)

    #Axes
    ax.set_xticks(range(width))
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.xaxis.labelpad = 0.5
    ax.set_yticks(range(height))
    ax.set_yticklabels(y_labels , rotation='horizontal')
    #plt.tight_layout()
    plt.savefig(outputname)
    #plt.show()
