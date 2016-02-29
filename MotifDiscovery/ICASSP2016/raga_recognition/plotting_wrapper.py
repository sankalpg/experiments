#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json, pickle
import codecs

baseClusterCoffFileName = '_ClusteringCoff'
randomizationSuffix = '_RANDOM'
baseNetworkPropFileName = '_NetworkInfo'
    
def readClusterinCoffCurve(root_dir, nFiles):
    """
    This function just reads multiple files and accumulate CC values and returns it
    """
    CC = []
    CC_Rand = []    
    for ii in range(1, nFiles+1):

        try:
            cc = np.loadtxt(os.path.join(root_dir, str(ii)+baseClusterCoffFileName+'.txt'))
            cc_rand = np.loadtxt(os.path.join(root_dir,str(ii)+baseClusterCoffFileName+randomizationSuffix+'.txt'))
        except:
            cc = 0
            cc_rand = 0
         
        CC.append(cc)
        CC_Rand.append(cc_rand)
    
    return CC, CC_Rand

def plotClusteringCoff(root_dir, nFiles, plotName=-1, legData = []):
    """
    This function plots clustering cofficient as a function of threshold using which the network was build. It also plots the CC corresponding to the randomized network.
    
    """
    cc, cc_rand = readClusterinCoffCurve(root_dir, nFiles)
    
    cc = np.array(cc[1:])   #because of 0 bin is 1 file
    cc_rand = np.array(cc_rand[1:])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    fsize = 18
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("$T_s$ (bin index)", fontsize = fsize, fontname=font)
    plt.ylabel("Clustering Coefficient $C$", fontsize = fsize, fontname=font, labelpad=fsize2)
    
    
    pLeg = []
    
    markers = ['.', 'o', 's', '^', '<', '>', 'p']    
    colors = ['r', 'b', 'm', 'c', 'g', 'k']
    colors_dotted = ['r--', 'b--', 'm--', 'c--', 'g--', 'k--']

    p, = plt.plot(cc, 'b', linewidth=2)
    pLeg.append(p)
    p, = plt.plot(cc_rand, 'b--', linewidth=2)
    pLeg.append(p)
    p, = plt.plot(cc-cc_rand, 'r', linewidth=0.5, marker = '.')
    pLeg.append(p)

    ax.set_ylim([0,0.45])
    ax.set_xlim([0,35])
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.legend(pLeg, ['$C(\mathcal{G})$','$C(\mathcal{G}_r)$', '$C(\mathcal{G})-C(\mathcal{G}_r)$'], fontsize = 15, loc=4, bbox_to_anchor=(1,0.1))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')
        
        
#def plot_confusion_matrix(result_file, raga_uuid_name_file):


def plot_confusion_matrix(raga_name_map_file, result_file, outputname):
    
    raga_name_map = json.load(open(raga_name_map_file,'r'))
    results = pickle.load(open(result_file, 'r'))
    conf_arr = results['var2']['cm'][0]
    raga_names = results['var2']['mapping']
    
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
                    extent=[0, width, height, 0],
                    vmin =np.min(conf_arr)-.5, vmax = np.max(conf_arr)+0.5,
                    )
    ticks = np.arange(np.min(conf_arr),np.max(conf_arr)+1)
    tickpos = np.linspace(ticks[0] , ticks[-1] , len(ticks));
    #cax = plt.colorbar(mat, ticks=tickpos)
    #cax.set_ticklabels(ticks)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    cb = fig.colorbar(res, cax=cax, orientation = 'horizontal', ticks=tickpos)

    #Axes
    ax.set_xticks(np.arange(width))
    ax.set_xticklabels(x_labels, rotation='vertical')
    #ax.xaxis.labelpad = 0.1
    ax.set_yticks(np.arange(height))
    ax.set_yticklabels(y_labels , rotation='horizontal')

    for x in xrange(conf_arr.shape[0]):
        for y in xrange(conf_arr.shape[1]):
            textcolor = 'black'
            if conf_arr[x,y] >= 6:
                textcolor = 'white'
            if conf_arr[x,y]==0:
                continue
            ax.annotate(int(conf_arr[x,y]), xy=(y+0.5, x+0.5),  horizontalalignment='center', verticalalignment='center', color=textcolor, fontsize=12)
    plt.tight_layout()
    plt.savefig(outputname)
    #plt.show()
    

def plotPerThresholdAcuracy(cc_dir, thsld_dir, plotName=-1):
    """
    """
    #lets read all the clustering coffs
    cc, cc_rand = readClusterinCoffCurve(cc_dir, 35)
    cc_diff = {}
    for ii in range(30):
        cc_diff[ii] = cc[ii+1]-cc_rand[ii+1]
    
    #lets read accuracies for all the files with different distance threhsolds
    range_thslds = range(6,16)
    cnt=1
    accuracy = {}
    for ii in range_thslds:
        data = pickle.load(open(os.path.join(thsld_dir, 'config_%d'%cnt, 'experiment_results.pkl')))
        cnt+=1
        accuracy[ii] = data['var2']['accuracy']
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.hold(True)
    fsize = 20
    fsize2 = 15
    font="Times New Roman"
    
    ax1.set_xlabel("$T_s$ (bin index)", fontsize = fsize, fontname=font)
    ax1.set_ylabel("Clustering Coefficient $C$", fontsize = fsize, fontname=font,  labelpad=10)
    ax2.set_ylabel("Accuracy (%)", fontsize = fsize, fontname=font,  labelpad=10)
    
    pLeg = []
    markers = ['.', 'o', 's', '^', '<', '>', 'p']    
    colors = ['r', 'b', 'm', 'c', 'g', 'k']
    colors_dotted = ['r--', 'b--', 'm--', 'c--', 'g--', 'k--']

    p, = ax1.plot(range_thslds, [cc_diff[i] for i in range_thslds], 'r--', linewidth=2)
    pLeg.append(p)
    p, = ax2.plot(range_thslds, [accuracy[i]*100 for i in range_thslds], 'b', linewidth=2)
    pLeg.append(p)
    
    ax1.set_ylim([.1,0.28])
    ax1.set_xlim([6,15])
    ax2.set_ylim([30,75])
    
    xLim = ax1.get_xlim()
    yLim = ax1.get_ylim()    
    #ax1.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    
    ax1.set(adjustable='box-forced',
    aspect=((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0]))))
    
    xLim = ax2.get_xlim()
    yLim = ax2.get_ylim()
    #ax2.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    
    ax2.set(adjustable='box-forced',
    aspect=((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0]))))
    
    
    plt.legend(pLeg, ['$C(\mathcal{G})-C(\mathcal{G}_r)$', 'Accuracy of $M$'], fontsize = 14, loc=1)
    ax1.tick_params(axis='both', which='major', labelsize=fsize2)
    ax2.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')
        


#fig, ax1 = plt.subplots()
#t = np.arange(0.01, 10.0, 0.01)
#s1 = np.exp(t)
#ax1.plot(t, s1, 'b-')
#ax1.set_xlabel('time (s)')
## Make the y-axis label and tick labels match the line color.
#ax1.set_ylabel('exp', color='b')
#for tl in ax1.get_yticklabels():
    #tl.set_color('b')


#ax2 = ax1.twinx()
#s2 = np.sin(2*np.pi*t)
#ax2.plot(t, s2, 'r.')
#ax2.set_ylabel('sin', color='r')
#for tl in ax2.get_yticklabels():
    #tl.set_color('r')
#plt.show()

    
def get_confusion_matrix_data_sertan(filename):
    
    data = json.load(open(filename,'r'))
    
    gt_label = []
    pd_label = []
    for d in data:
        gt_label.append(d['annotated_mode'])
        pd_label.append(d['estimated_mode'])
    
    raga_ind_map = {}
    ind_raga_map = {}
    
    u_ragas = np.unique(np.array(gt_label))
    
    for ii, u in enumerate(u_ragas):
        raga_ind_map[u] = ii
        ind_raga_map[ii] = u
    
    cnf_mtx = np.zeros((len(u_ragas), len(u_ragas)))
    
    
    for ii, gt in enumerate(gt_label):
        cnf_mtx[raga_ind_map[gt], raga_ind_map[pd_label[ii]]]+=1
    
    return cnf_mtx, raga_ind_map

def plot_confusion_matrix_sertan(raga_name_map_file, sertan_results_file, outputname):
    
    raga_name_map = json.load(open(raga_name_map_file,'r'))
    conf_arr,raga_names =  get_confusion_matrix_data_sertan(sertan_results_file)
    
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
