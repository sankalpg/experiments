import numpy as np
import os,sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    
    cc = np.array(cc)
    cc_rand = np.array(cc_rand)
    
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

    ax.set_ylim([0,0.6])
    ax.set_xlim([0,40])
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.legend(pLeg, ['$C(\mathcal{G})$','$C(\mathcal{G}_r)$', '$C(\mathcal{G})-C(\mathcal{G}_r)$'], fontsize = 12, loc=2)
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')
        
        
#def plot_confusion_matrix(result_file, raga_uuid_name_file):


def plot_confusion_matrix(alphabet, conf_arr, outputname):
    norm_conf = []
    width = len(conf_arr)
    height = len(conf_arr[0])
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(14,14))
    #fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.grid(which='major')
    cmap_local = plt.get_cmap('OrRd', np.max(conf_arr)-np.min(conf_arr)+1)
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
    ax.set_xticklabels(alphabet, rotation='vertical')
    ax.xaxis.labelpad = 0.5
    ax.set_yticks(range(height))
    ax.set_yticklabels(alphabet, rotation='horizontal')
    #plt.tight_layout()
    plt.savefig(outputname, format='pdf')
    
    