#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import matplotlib.pyplot as plt


def plotDistanceMeasureOutputs(distanceFile, plotName=-1):
    
    #reading distances
    distInfo = np.loadtxt(distanceFile)
    
    cumulativeValues = [25, 50, 75]
    CategoryNames = ['$d_1$', '$d_2$', '$d_3$', '$d_4$']
    colors = ['g', 'ks', 'ro',  'b--']
#    markers = ['^', 'o', 's', 'D']
    linewidths = [3,0.1 ,0.1 , 3]

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pLeg = []
    
    ind = range(0, distInfo.shape[1], 100)
    for ii in range(distInfo.shape[0]-1):
        
        p, = plt.plot(distInfo[0,ind], distInfo[ii+1,ind], colors[ii], linewidth=linewidths[ii], markersize=4.5)#, marker = markers[ii], markersize=5)
        pLeg.append(p)

    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("$\delta$ (cents)", fontsize = fsize, fontname=font)
    plt.ylabel("$d_i$", fontsize = fsize, fontname=font, labelpad=fsize2)

    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 2, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    
    ax.set_xlim([0,500 ])
    ax.set_ylim([0,500])
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1