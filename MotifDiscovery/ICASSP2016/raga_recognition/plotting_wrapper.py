import numpy as np
import os,sys
import matplotlib.pyplot as plt


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
    
    