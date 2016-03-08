#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import os,sys
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json, pickle
import matplotlib as mpl
from matplotlib import rc, font_manager
rc('font',family='Times New Roman')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import codecs
plt.switch_backend('agg')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/melodyProcessing'))
import phaseSpaceEmbedding as pse



def plot_confusion_matrix(raga_name_map_file, result_file, outputname):
    
    raga_name_map = json.load(codecs.open(raga_name_map_file,'r', encoding = 'utf-8'))
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


def plotSurfacesExample(plotName = -1):

    #yaman
    feature_file = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/pitch_tonic/48b37bed-e847-4882-8a01-5c721e07f07d/Ajoy_Chakrabarty/The_Genius_Of_Pt__Ajoy_Chakraborty/Raga_Yaman_e59642ca-72bc-466b-bf4b-d82bfbc7b4af.phasespace_500'

    #Bilaskhani todi
    #feature_file = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/pitch_tonic/dd59147d-8775-44ff-a36b-0d9f15b31319/Ajoy_Chakrabarty/The_Genius_Of_Pt__Ajoy_Chakraborty/Raga_Bilaskhani_Todi_d7510269-b26c-4735-a491-245f3c732a58.phasespace_500'
    
    compression = 0.1
    smooth_gauss_sigma = 1


    feature =np.loadtxt(feature_file)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout()
    fsize = 18
    fsize2 = 18
    font="FreeSerif"

    
    ax1.set_xlabel("Index", fontsize = fsize, fontname=font)
    ax2.set_xlabel("Index", fontsize = fsize, fontname=font)
    ax1.set_ylabel(" Index", fontsize = fsize, fontname=font)
    ax1.tick_params(axis='both', which='major', labelsize=fsize2)
    ax2.tick_params(axis='both', which='major', labelsize=fsize2)
    
    feature = feature/np.max(feature)
    ax1.imshow(feature**0.75)
    ax1.set_title('(a)', fontname=font, fontsize=fsize2)

    feature = np.power(feature, compression)
    feature = pse.ndimage.gaussian_filter(feature, smooth_gauss_sigma)
    feature = feature/np.max(feature)

    im = ax2.imshow(feature)
    ax2.set_title('(b)', fontname=font, fontsize=fsize2)

    #cbar_ax = fig.add_axes([0.01, 0.01, 1, 0.01])
    cax,kw = mpl.colorbar.make_axes([ax for ax in [ax1, ax2]], orientation='horizontal')
    cbar = plt.colorbar(im, cax=cax, **kw)
    cbar.ax.tick_params(labelsize=fsize2) 
    #cbar = fig.colorbar(cax, ticks=[-1, 0], orientation='horizontal', cax = cbar_ax)
    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

def getInds(array, time_stamps):
    inds = []
    for t in time_stamps:
        inds.append(np.argmin(abs(array-t)))
    return inds


def plotChalanExample(plotName = -1):

    file1 =     '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/063ea5a0-23b1-4bb5-8537-3d924fe8ebb3/Ajoy_Chakrabarty/Raag_Kedar_&_Jog/Raag_Jog_566d6c64-ae66-45e7-9f91-e33cd6c0f18f.pitchSilIntrpPP'
    tonic1 =    '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/063ea5a0-23b1-4bb5-8537-3d924fe8ebb3/Ajoy_Chakrabarty/Raag_Kedar_&_Jog/Raag_Jog_566d6c64-ae66-45e7-9f91-e33cd6c0f18f.tonicFine'
    time_stamp1 = (363.1, 366)
    pitch1 = np.loadtxt(file1)
    tonic1 = np.loadtxt(tonic1)
    pitch1[:,1] = 1200*np.log2(pitch1[:,1]/tonic1)
    inds1 = getInds(pitch1[:,0], time_stamp1)

    file2 =     '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/d9c603fa-875f-4b84-b851-c6a345427898/Anol_Chatterjee/Raag_Abhogi_&_Megh/Raag_Abhogi_e40375a2-01d2-4137-823a-3e62503baf98.pitchSilIntrpPP'
    tonic2 =    '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/d9c603fa-875f-4b84-b851-c6a345427898/Anol_Chatterjee/Raag_Abhogi_&_Megh/Raag_Abhogi_e40375a2-01d2-4137-823a-3e62503baf98.tonicFine'
    time_stamp2 = (55.4, 57.8)
    pitch2 = np.loadtxt(file2)
    tonic2 = np.loadtxt(tonic2)
    pitch2[:,1] = 1200*np.log2(pitch2[:,1]/tonic2)
    inds2 = getInds(pitch2[:,0], time_stamp2)

    # file3 =     '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/7591faad-e68a-4550-b675-8082842c6056/Rashid_Khan/Vocal/Raga_Bageshri_1d378f5b-33ac-47ae-9219-a03a394de1b9.pitchSilIntrpPP'
    # tonic3 =    '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/7591faad-e68a-4550-b675-8082842c6056/Rashid_Khan/Vocal/Raga_Bageshri_1d378f5b-33ac-47ae-9219-a03a394de1b9.tonicFine'
    # time_stamp3 = (164.3, 167)
    # pitch3 = np.loadtxt(file3)
    # tonic3 = np.loadtxt(tonic3)
    # pitch3[:,1]= 1200*np.log2(pitch3[:,1]/tonic3)
    # inds3 = getInds(pitch3[:,0], time_stamp3)

    270
    file4 =     '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/64e5fb9e-5569-4e80-8e6c-f543af9469c7/Ulhas_Kashalkar/Masterpieces/Malkauns_Bada_Khyal_2d207bf5-90cf-4d74-8703-b3b0d52a531f.pitchSilIntrpPP'
    tonic4 =    '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/audio/64e5fb9e-5569-4e80-8e6c-f543af9469c7/Ulhas_Kashalkar/Masterpieces/Malkauns_Bada_Khyal_2d207bf5-90cf-4d74-8703-b3b0d52a531f.tonicFine'
    time_stamp4 = (525.9, 528.2)
    pitch4 = np.loadtxt(file4)
    tonic4 = np.loadtxt(tonic4)
    pitch4[:,1] = 1200*np.log2(pitch4[:,1]/tonic4)
    inds4 = getInds(pitch4[:,0], time_stamp4)

    plt.hold(True)
    plt.plot(pitch1[inds1[0]:inds1[1], 1])
    plt.plot(pitch2[inds2[0]:inds2[1], 1])
    # plt.plot(pitch3[inds3[0]:inds3[1], 1])
    plt.plot(pitch4[inds4[0]:inds4[1], 1])

    plt.show()
    

def plotAccuracyVsParameter(plotName = -1):

    hind_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/results/PhaseSpaceEmbedding/V1/hindustani/gridsearch'
    carn_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/results/PhaseSpaceEmbedding/V1/carnatic/gridsearch'

    configs_tau_h = [376, 916, 1456, 1996, 2536]
    configs_tau_c = [538, 1078, 1618, 2158, 2698]
    vals_tau = ['0.2', '0.3', '0.5', '1', '1.5']
    
    configs_alpha_h = [1402, 1429, 1456, 1483, 1375]
    configs_alpha_c = [997, 1024, 1051, 1078, 970]
    vals_alpha = ['0.1', '0.25', '0.5', '0.75', '1']

    configs_sigma_h = [1186, 1321, 1456, 1591]
    configs_sigma_c = [673, 808, 943, 1078]
    vals_sigma = ['-', '1', '2', '3']

    configs_knn_h = [1456, 1457, 1458]
    configs_knn_c = [1078, 1079, 1080]
    vals_knn = ['1', '3', '5']


    acc_tau_h = []
    acc_tau_c = []

    root_dir = hind_dir
    for c in configs_tau_h:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_tau_h.append(round(100*data['var1']['accuracy'],2))

    root_dir = carn_dir
    for c in configs_tau_c:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_tau_c.append(round(100*data['var1']['accuracy'],2))

    

    acc_alpha_h = []
    acc_alpha_c = []

    root_dir = hind_dir
    for c in configs_alpha_h:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_alpha_h.append(round(100*data['var1']['accuracy'],2))

    root_dir = carn_dir
    for c in configs_alpha_c:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_alpha_c.append(round(100*data['var1']['accuracy'],2))      

    

    acc_sigma_h = []
    acc_sigma_c = []

    root_dir = hind_dir
    for c in configs_sigma_h:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_sigma_h.append(round(100*data['var1']['accuracy'],2))

    root_dir = carn_dir
    for c in configs_sigma_c:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_sigma_c.append(round(100*data['var1']['accuracy'],2))  


    acc_knn_h = []
    acc_knn_c = []

    root_dir = hind_dir
    for c in configs_knn_h:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_knn_h.append(round(100*data['var1']['accuracy'],2))

    root_dir = carn_dir
    for c in configs_knn_c:
        filename = os.path.join(root_dir, 'config_%d'%c, 'experiment_results.pkl')
        data = pickle.load(open(filename, 'r'))
        acc_knn_c.append(round(100*data['var1']['accuracy'],2))  



    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)
    plt.tight_layout()
    fsize = 20
    fsize2 = 24
    font="Times New Roman Bold"
    
    ax1.hold(True)
    ax1.plot(range(len(acc_tau_h)), acc_tau_h, marker = 'o', color = 'r', linestyle = 'dashed', linewidth = 0.5)
    ax1.plot(range(len(acc_tau_c)), acc_tau_c, marker = 's', color = 'b', linestyle = 'dashed', linewidth = 0.5)
    
    ax1.set_ylim([60, 100])
    yLim = ax1.get_ylim() 
    ax1.set_xticks(range(len(acc_tau_h)))
    ax1.set_xticklabels(vals_tau)
    ax1.set_yticks(np.arange(yLim[0], yLim[1]+1, 10))
    ax1.set_ylabel("Accuracy (\%)", fontsize = fsize)
    ax1.set_xlabel(r"$\tau$", fontsize = fsize2, fontname = font)
    ax1.tick_params(axis='both', which='major', labelsize=fsize)

    # xLim = ax1.get_xlim()
    # yLim = ax1.get_ylim()    
    # ax1.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    
    ax2.hold(True)
    ax2.plot(range(len(acc_alpha_h)), acc_alpha_h, marker = 'o', color = 'r', linestyle = 'dashed', linewidth = 0.5)
    ax2.plot(range(len(acc_alpha_c)), acc_alpha_c, marker = 's', color = 'b', linestyle = 'dashed', linewidth = 0.5)

    ax2.set_ylim([60, 100])
    yLim = ax2.get_ylim() 
    ax2.set_xticks(range(len(acc_alpha_h)))
    ax2.set_xticklabels(vals_alpha)
    ax2.set_yticks(np.arange(yLim[0], yLim[1]+1, 10))
    ax2.set_xlabel(r"$\alpha$", fontsize = fsize2, fontname = font)
    ax2.tick_params(axis='both', which='major', labelsize=fsize)

    # xLim = ax2.get_xlim()
    # yLim = ax2.get_ylim()    
    # ax2.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    
    

    ax3.hold(True)
    ax3.plot(range(len(acc_sigma_h)), acc_sigma_h, marker = 'o', color = 'r', linestyle = 'dashed', linewidth = 0.5)
    ax3.plot(range(len(acc_sigma_c)), acc_sigma_c, marker = 's', color = 'b', linestyle = 'dashed', linewidth = 0.5)

    ax3.set_ylim([60, 100])
    yLim = ax3.get_ylim() 
    ax3.set_xticks(range(len(acc_sigma_h)))
    ax3.set_xticklabels(vals_sigma)
    ax3.set_yticks(np.arange(yLim[0], yLim[1]+1, 10))
    ax3.set_xlabel(r"$\sigma$", fontsize = fsize2, fontname = font)
    ax3.set_ylabel("Accuracy (\%)", fontsize = fsize)
    ax3.tick_params(axis='both', which='major', labelsize=fsize) 

    # xLim = ax3.get_xlim()
    # yLim = ax3.get_ylim()    
    # ax3.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    

    ax4.hold(True)
    ax4.plot(range(len(acc_knn_h)), acc_knn_h, marker = 'o', color = 'r', linestyle = 'dashed', linewidth = 0.5)
    ax4.plot(range(len(acc_knn_c)), acc_knn_c, marker = 's', color = 'b', linestyle = 'dashed', linewidth = 0.5)

    ax4.set_ylim([60, 100])
    yLim = ax4.get_ylim() 
    ax4.set_xticks(range(len(acc_knn_h)))
    ax4.set_xticklabels(vals_knn)
    ax4.set_yticks(np.arange(yLim[0], yLim[1]+1, 10))
    ax4.set_xlabel(r"$k$", fontsize = fsize2, fontname = font)
    ax4.tick_params(axis='both', which='major', labelsize=fsize) 

    # xLim = ax4.get_xlim()
    # yLim = ax4.get_ylim()    
    # ax4.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    
    #fig.subplots_adjust(wspace=-0.6)
    fig.tight_layout(pad=0.4)


    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

def getInds(array, time_stamps):
    inds = []
    for t in time_stamps:
        inds.append(np.argmin(abs(array-t)))
    return inds
    







