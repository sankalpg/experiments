import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../library_pythonnew/batchProcessing/'))

import batchProcessing as BP

eps = np.finfo(np.float).eps

#this function plots the ROC curve based on flat and non flat annotations for a particular variance length
def estimateFlatnessThreshold(root_dir, annotationExt, var_len):
    
    binsPOctave = 1200
    dFactor = 5
    var_len = var_len/1000.0    #in ms
    
    filenames = BP.GetFileNamesInDir(root_dir, annotationExt)
    
    flat = np.array([])
    nonFlat = np.array([])
    
    for filename in filenames:
        
        fname, ext = os.path.splitext(filename)
        
        #open pitch file file
        pitchData = np.loadtxt(fname+'.pitch')
        
        pHop = (pitchData[1,0]- pitchData[0,0])*dFactor
        var_samples = int(round(var_len/pHop))
        
        #load tonic data
        tonic = 55 # since we subtract mean actual tonic doesn't matter
        
        pitchData[:,1]=(binsPOctave*np.log2((eps+pitchData[:,1])/tonic)).astype(np.int) 
        
        #downsampling
        ind_new = np.arange(0,pitchData.shape[0],dFactor)
        pitchData = pitchData[ind_new,:]
        pCents = pitchData[:,1]
        
        #removing silence
        indSil = np.where(pCents<0)[0]
        pCents = np.delete(pCents, indSil)
        pitchData = np.delete(pitchData,[indSil],axis=0)
        
        running_mean = np.zeros(pCents.shape[0])
        running_std = np.zeros(pCents.shape[0])
        running_mean[var_samples] = np.sum(pCents[0:2*var_samples+1])
        for ii in np.arange(var_samples+1,pCents.size-var_samples-1):
            running_mean[ii] = running_mean[ii-1]+ pCents[ii+var_samples]-pCents[ii-var_samples-1]
        
        running_mean= running_mean/(1+(var_samples*2))
        
        pCents_sq = np.square(pCents-running_mean)
        
        running_std[var_samples] = np.sum(pCents_sq[0:2*var_samples+1])
        for ii in np.arange(var_samples+1,pCents.size-var_samples-1):
            running_std[ii] = running_std[ii-1]+ pCents_sq[ii+var_samples]-pCents_sq[ii-var_samples-1]
        
        running_std = np.sqrt(running_std/(1+(var_samples*2)))
        
        
        """
        plt.figure()
        plt.hold(True)
        plt.plot(pitchData[:,0], pCents, color='r')
        plt.plot(pitchData[:,0],running_std, color='b')
        """
        
        #opening annotation file
        lines = open(fname + annotationExt).readlines()
        for ii,line in enumerate(lines):
            line = line.split()
            startTime = float(line[0])
            dur = float(line[1])
            annot = (line[2])
            
            startInd = np.argmin(abs(pitchData[:,0]-startTime))
            endInd = np.argmin(abs(pitchData[:,0]-(startTime+dur)))
            if annot == 'f':
                flat = np.append(flat, running_std[startInd:endInd+1])
                #plt.plot(pitchData[startInd:endInd+1,0], running_std[startInd:endInd+1], color='r')
            elif annot == 'nf':
                nonFlat = np.append(nonFlat, running_std[startInd:endInd+1])
    
    ind_zero = np.where(nonFlat<0.1)[0]
    nonFlat = np.delete(nonFlat,ind_zero)
    
    bins = np.arange(0,3600,5)
    flatHist = np.histogram(flat, bins=bins)
    nonFlatHist = np.histogram(nonFlat, bins=bins)
    
    TruePost = []
    FalsePost = []
    binRef = []
    NTotal = flat.shape[0] + nonFlat.shape[0]
    for ii,binval in enumerate(bins):
        indFlatTrue = np.where(flat<binval)[0]
        indNonFlatTrue = np.where(nonFlat<binval)[0]
        
        #indFlatFlase = np.where(flat>=binval)[0]
        
        #indNonFlatFlase = np.where(nonFlat>=binval)[0]
        try:
            TruePost.append(indFlatTrue.size/float(flat.shape[0]))
            FalsePost.append(indNonFlatTrue.size/float(nonFlat.shape[0]))
            binRef.append(binval)
        except:
            FalsePost.append(0)
            TruePost.append(0)
            binRef.append(0)
            print "something happened"
        
        
    #print flat.shape[0]*pHop, nonFlat.shape[0]*pHop
    """
    plt.figure()
    plt.hold(True)
    plt.plot(flatHist[0])
    plt.plot(nonFlatHist[0])
    plt.figure()
    plt.plot(FalsePost, TruePost, color='k')   #essentially (1-precision, recall)
    plt.figure()
    plt.plot(FalsePost,binRef, color='r')
    plt.show()
    """
    return FalsePost, TruePost, binRef

def plotDifferentROCs(root_dir, annotationExt, plotName=-1):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pLeg = []
    colors= ['r', 'k', 'b']
    markers = ['o', 's', '^']
    var_lens = [50, 100, 200]
    variants = ['100 ms', '200 ms', '400 ms']
    
    for ii,var_len in enumerate(var_lens):
        FP, TP, bins = estimateFlatnessThreshold(root_dir, annotationExt, var_len)
        p, = plt.plot(FP, TP, colors[ii], linewidth=1.5, marker = markers[ii], markersize=5)
        pLeg.append(p)

    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlim([0, 1])
    plt.xlabel("False Positives", fontsize = fsize, fontname=font)
    plt.ylabel("True Positives", fontsize = fsize, fontname=font, labelpad=fsize2)
    
    plt.legend(pLeg, [variants[x] for x in range(len(var_lens))], loc ='lower right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1
    
    
