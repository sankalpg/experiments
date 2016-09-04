#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import os,sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json, pickle
import codecs
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rc('font',family='Times New Roman')
from sklearn.metrics import confusion_matrix
baseClusterCoffFileName = '_ClusteringCoff'
randomizationSuffix = '_RANDOM'
baseNetworkPropFileName = '_NetworkInfo'
    
def readClusterinCoffCurve(root_dir, nFiles,suffix=''):
    """
    This function just reads multiple files and accumulate CC values and returns it
    """
    CC = []
    CC_Rand = []    
    for ii in range(1, nFiles+1):

        try:
            cc = np.loadtxt(os.path.join(root_dir, str(ii)+baseClusterCoffFileName+suffix+'.txt'))
            cc_rand = np.loadtxt(os.path.join(root_dir,str(ii)+baseClusterCoffFileName+randomizationSuffix+suffix+'.txt'))
        except:
            cc = 0
            cc_rand = 0
         
        CC.append(cc)
        CC_Rand.append(cc_rand)
    
    return CC, CC_Rand

def plotClusteringCoff(root_dir, nFiles, suffix = '', plotName=-1, legData = []):
    """
    This function plots clustering cofficient as a function of threshold using which the network was build. It also plots the CC corresponding to the randomized network.
    Hindustani: plt.plotClusteringCoff('ClusteringCoffs/hindustani/config2/clusteringCoffs_Sep2016/', 17, suffix='_config2', plotName='plots/CC_Evolution_Hindustani_Config2.pdf')
    Carnatic: plt.plotClusteringCoff('ClusteringCoffs/carnatic/config3/clusteringCoffs_Sep2016/', 17, suffix='_config3', plotName='plots/CC_Evolution_Carnatic_Config3.pdf')
    
    """
    cc, cc_rand = readClusterinCoffCurve(root_dir, nFiles, suffix)
    
    cc = np.array(cc)   #because of 0 bin is 1 file
    cc_rand = np.array(cc_rand)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    fsize = 18
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("$\Delta''$ (bin index)", fontsize = fsize, fontname=font)
    plt.ylabel("Clustering Coefficient $C$", fontsize = fsize, fontname=font, labelpad=fsize2)
    
    
    pLeg = []
    
    markers = ['.', 'o', 's', '^', '<', '>', 'p']    
    colors = ['r', 'b', 'm', 'c', 'g', 'k']
    colors_dotted = ['r--', 'b--', 'm--', 'c--', 'g--', 'k--']

    p, = plt.plot(cc, 'b', linewidth=2)
    plt.xticks(np.arange(0, len(cc)+2, 1))
    pLeg.append(p)
    p, = plt.plot(cc_rand, 'b--', linewidth=2)
    pLeg.append(p)
    p, = plt.plot(cc-cc_rand, 'r', linewidth=0.5, marker = '.')
    pLeg.append(p)

    # ax.set_ylim([0,0.45])
    # ax.set_xlim([0,35])
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.legend(pLeg, ['$C(\mathcal{G})$','$C(\mathcal{G}_r)$', '$C(\mathcal{G})-C(\mathcal{G}_r)$'], fontsize = 15, loc=3,  ncol = 3,frameon=True, borderaxespad=0.6,bbox_to_anchor=(0.1, 1.))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')
        
        
#def plot_confusion_matrix(result_file, raga_uuid_name_file):



def plotPerThresholdAcuracyHindustaniConfig2():
    """
    Final thesis plot
    """

    cc_dir = 'ClusteringCoffs/hindustani/config2/clusteringCoffs_Sep2016/'
    thsld_dir =  'PhraseBased/Hindustani/thresholds'
    thslds = range(3,15)
    plotName = str("plots/HindustaniConfig2_Accuracy_vs_CCdiff.pdf")
    suffix = '_config2'
    
    #lets read all the clustering coffs
    cc, cc_rand = readClusterinCoffCurve(cc_dir, 30,suffix)
    cc_diff = {}
    for ii in range(25):
        cc_diff[ii] = cc[ii]-cc_rand[ii]
    
    #lets read accuracies for all the files with different distance threhsolds
    range_thslds = thslds
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
    fsize = 18
    fsize2 = 16
    
    ax1.set_xlabel("$\Delta''$ (bin index)", fontsize = fsize)
    ax1.set_ylabel("Clustering Coefficient $C$", fontsize = fsize, labelpad=10)
    ax2.set_ylabel("Accuracy (%)", fontsize = fsize, labelpad=10)
    
    pLeg = []
    markers = ['.', 'o', 's', '^', '<', '>', 'p']    
    colors = ['r', 'b', 'm', 'c', 'g', 'k']
    colors_dotted = ['r--', 'b--', 'm--', 'c--', 'g--', 'k--']

    p, = ax1.plot(range_thslds, [cc_diff[i] for i in range_thslds], 'r--', linewidth=2)
    plt.xticks(np.arange(min(range_thslds), max(range_thslds)+1, 1))
    pLeg.append(p)
    p, = ax2.plot(range_thslds, [accuracy[i]*100 for i in range_thslds], 'b', linewidth=2)
    pLeg.append(p)
    
    # ax1.set_ylim([.1,0.28])
    # ax1.set_xlim([min(thslds),max(thslds)])
    # ax2.set_ylim([30,75])
    
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
    
    
    plt.legend(pLeg, ['$C(\mathcal{G})-C(\mathcal{G}_r)$', 'Accuracy of $\mathrm{M}_\mathrm{VSM}$'], fontsize = fsize, loc=3,  ncol = 2,frameon=True, borderaxespad=0.6,bbox_to_anchor=(0.05, 1.))
    ax1.tick_params(axis='both', which='major', labelsize=fsize2)
    ax2.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

def plotPerThresholdAcuracyCarnaticConfig3():
    """
    Final thesis plot
    """

    cc_dir = 'ClusteringCoffs/carnatic/config3/clusteringCoffs_Sep2016/'
    thsld_dir =  'PhraseBased/Carnatic/thresholds'
    thslds = range(3,15)
    plotName = str("plots/CarnaticConfig3_Accuracy_vs_CCdiff.pdf")
    suffix = '_config3'
    
    #lets read all the clustering coffs
    cc, cc_rand = readClusterinCoffCurve(cc_dir, 30,suffix)
    cc_diff = {}
    for ii in range(25):
        cc_diff[ii] = cc[ii]-cc_rand[ii]
    
    #lets read accuracies for all the files with different distance threhsolds
    range_thslds = thslds
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
    fsize = 18
    fsize2 = 16
    
    ax1.set_xlabel("$\Delta''$ (bin index)", fontsize = fsize)
    ax1.set_ylabel("Clustering Coefficient $C$", fontsize = fsize, labelpad=10)
    ax2.set_ylabel("Accuracy (%)", fontsize = fsize, labelpad=10)
    
    pLeg = []
    markers = ['.', 'o', 's', '^', '<', '>', 'p']    
    colors = ['r', 'b', 'm', 'c', 'g', 'k']
    colors_dotted = ['r--', 'b--', 'm--', 'c--', 'g--', 'k--']

    p, = ax1.plot(range_thslds, [cc_diff[i] for i in range_thslds], 'r--', linewidth=2)
    plt.xticks(np.arange(min(range_thslds), max(range_thslds)+1, 1))
    pLeg.append(p)
    p, = ax2.plot(range_thslds, [accuracy[i]*100 for i in range_thslds], 'b', linewidth=2)
    pLeg.append(p)
    
    # ax1.set_ylim([.1,0.28])
    # ax1.set_xlim([min(thslds),max(thslds)])
    # ax2.set_ylim([30,75])
    
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
    
    
    plt.legend(pLeg, ['$C(\mathcal{G})-C(\mathcal{G}_r)$', 'Accuracy of $\mathrm{M}_\mathrm{VSM}$'], fontsize = fsize, loc=3,  ncol = 2,frameon=True, borderaxespad=0.6,bbox_to_anchor=(0.05, 1.))
    ax1.tick_params(axis='both', which='major', labelsize=fsize2)
    ax2.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')        


def get_GT_PD_labels_VSM(filename, version):
    """
    To fetch ground truth and predicted labels in output files produced in VSM method
    """
    data = pickle.load(open(filename,'r'))
    mapping = data[version]['mapping']

    gt_labels = []
    pred_labels = []

    for lab in data[version]['gt_label']:
        gt_labels.append(mapping[lab])

    for lab in data[version]['pred_label']:
        pred_labels.append(mapping[lab[0]])

    return gt_labels, pred_labels


def get_GT_PD_labels_TDMS(filename, version):
    """
    To fetch ground truth and predicted labels in output files produced in TDMS method
    """
    data = pickle.load(open(filename,'r'))
    return data[version]['gt_label'], data['var1']['pred_label']


def get_GT_PD_labels_PCD(filename, version):
    """
    To fetch ground truth and predicted labels in output files produced in PCD method (Sertan)
    """
    data = json.load(open(filename,'r'))
    return data['true_raag'], data['estimated_raag']




def plotAllConfusionMatricesThesis():
    """
    This function plots all the confusion matrices to be used in the thesis
    """
    raga_name_map_file  = 'raga_name_mapping.json'

    vsm_cmd = ['vsm', 'PhraseBased/Carnatic/paperTable_ONLYVAR2/config_20/experiment_results.pkl', 'var2', 'plots/CM_vsm_cmd_var2.pdf']
    vsm_hmd = ['vsm', 'PhraseBased/Hindustani/paperTable_ONLYVAR2/config_20/experiment_results.pkl', 'var2', 'plots/CM_vsm_hmd_var2.pdf']

    tdms_cmd = ['tdms', '../ISMIR_2016/ISMIR_2016_Table_Results/carnatic/M_KL/experiment_results.pkl', 'var1', 'plots/CM_tdms_cmd_var1.pdf']
    tdms_hmd = ['tdms', '../ISMIR_2016/ISMIR_2016_Table_Results/hindustani/M_KL/experiment_results.pkl', 'var1', 'plots/CM_tdms_hmd_var1.pdf']

    pcd_cmd = ['pcd', '../ISMIR_2016/ISMIR_2016_Table_Results/carnatic/E_PCD/carnatic_40_classes_results.json', 'xxx', 'plots/CM_pcd_cmd.pdf']
    pcd_hmd = ['pcd', '../ISMIR_2016/ISMIR_2016_Table_Results/hindustani/E_PCD/hindustani_30_classes_results.json', 'xxx', 'plots/CM_pcd_hmd.pdf']



    methods = [vsm_cmd, vsm_hmd, tdms_cmd, tdms_hmd, pcd_cmd, pcd_hmd]

    for m in methods:
        plot_confusion_matrix(raga_name_map_file, m[1], m[3], m[2], m[0])


            
def plot_confusion_matrix(raga_name_map_file, result_file, outputname, version, id_method):
    
    fncs = {'vsm': get_GT_PD_labels_VSM, 'tdms':get_GT_PD_labels_TDMS, 'pcd': get_GT_PD_labels_PCD}
    raga_name_map = json.load(codecs.open(raga_name_map_file,'r', encoding = 'utf-8'))
    
    labels, predictions = fncs[id_method](result_file, version)
    

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
    #res = ax.matshow(np.array(conf_arr), #cmap=plt.cm.binary, 
    #                interpolation='nearest', aspect='1', cmap=cmap_local,
    #                ##Commenting out this line sets labels correctly,
    #                ##but the grid is off
    #                extent=[0, width, height, 0],
    #                vmin =np.min(conf_arr)-.5, vmax = np.max(conf_arr)+0.5,
    #                )
    res = ax.pcolor(np.array(conf_arr), cmap=cmap_local, edgecolor='black', linestyle=':', lw=1)
    
    ticks = np.arange(np.min(conf_arr),np.max(conf_arr)+2)
    tickpos = np.linspace(ticks[0] , ticks[-2], len(ticks));

    #cax = plt.colorbar(mat, ticks=tickpos)
    #cax.set_ticklabels(ticks)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    cb = fig.colorbar(res, cax=cax, orientation = 'horizontal', ticks=tickpos+0.5)
    cb.ax.set_xticklabels(np.arange(len(tickpos)),fontsize = 14)

    #Axes
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(width)+0.5)
    ax.set_xticklabels(x_labels, rotation='vertical', fontsize = 14)
    #ax.xaxis.labelpad = 0.1
    ax.set_yticks(np.arange(height)+0.5)
    ax.set_yticklabels(y_labels , rotation='horizontal', fontsize = 14)

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