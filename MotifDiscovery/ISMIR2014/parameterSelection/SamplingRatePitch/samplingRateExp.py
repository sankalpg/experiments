
import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing/'))

import batchProcessing as BP

#Python wrapper around ACR
def obtainAutoCorr(pitch, nPoints):
    acorr = np.zeros(nPoints)
    
    for ii in np.arange(nPoints):
        acorr[ii] = sum(pitch[:pitch.shape[0]-ii]*pitch[ii:])*(pitch.shape[0]/(pitch.shape[0]-ii))
    
    return acorr

#python code to compute ACR of all the candidates used for discovery [OLD CODE, NOW WE DO IT WITH THE C CODE DIRECTLY WHICH IS VERY FAST]    
def obtainAutoCorrBatch(root_dir, extProc,nPoints):
        
        
        filenames = BP.GetFileNamesInDir(root_dir, extProc)
        
        autoCorr = np.array([[]])
        
        for filename in filenames:
            
            fname, ext = os.path.splitext(filename)
            
            #open the pitch files
            pitchData = np.loadtxt(fname+'.pitchEssentia')
            
            #open tonic dile
            tonic = np.loadtxt(fname+ '.tonic').astype(np.float)
            
            #open annotation file
            motifData = open(fname+'.lab').readlines()
            
            for motif in motifData:
                
                try:
                    #find closes time stamp
                    motif = motif.split()
                    ind_start = np.argmin(abs(pitchData[:,0]-float(motif[0])))
                    ind_end = np.argmin(abs(pitchData[:,0]-(float(motif[0])+float(motif[1]))))
                    
                    pitchSeg = pitchData[:,1][ind_start: ind_end+1]
                    ind_Sil = np.where(pitchSeg<60)[0]
                    pitchSeg = np.delete(pitchSeg, ind_Sil)
                    
                    pitchSeg = 120*np.log2(pitchSeg/tonic)
                    pitchSeg = pitchSeg - np.mean(pitchSeg)
                    pitchSeg = pitchSeg/np.std(pitchSeg)
                    
                    acorr = obtainAutoCorr(pitchSeg, nPoints)
                    if autoCorr.size==0:
                        autoCorr = np.array([acorr/max(acorr)])
                    else:
                        autoCorr = np.append(autoCorr, np.array([acorr/max(acorr)]),axis=0)
                except:
                   print fname
        
        plt.hold(True)
        #for elem in autoCorr:         
        #    plt.plot(elem)
        mean = np.mean(autoCorr,axis=0)
        median = np.median(autoCorr,axis=0)
        std =  np.std(autoCorr,axis=0)
        minVal =  np.min(autoCorr,axis=0)
        maxVal =  np.max(autoCorr,axis=0)
        
        plt.plot(mean)    
        plt.plot(median)
        plt.plot(std)
        plt.plot(minVal)
        plt.plot(maxVal)
        plt.show()
            
def produce2D_ACR_Hist(acrFile, nBins=100):
    
    data = np.loadtxt(acrFile)
    
    bins = np.linspace(0,1, num=nBins)
    hist2D = np.zeros((bins.size-1,data.shape[1]))
    
    for jj in np.arange(data.shape[1]):
        hist = np.histogram(data[:,jj], bins = bins)
        hist2D[:,jj]= hist[0]
        
    """print hist2D
    plt.imshow(np.log(hist2D+1), cmap=plt.cm.hot, aspect='auto')
    plt.show()"""
    
    return hist2D
    
def produceAggregate2D_ACR_Hist(root_dir, fileExt):
    
    filenames = BP.GetFileNamesInDir(root_dir, fileExt)
    
    for ii,filename in enumerate(filenames):
        if ii==0:
            hist2D = produce2D_ACR_Hist(filename)
        else:
            try:
                hist2DTEMP = produce2D_ACR_Hist(filename)
                hist2D = hist2D+hist2DTEMP
            except:
                print "something wend wrong in file " + filename
                
    np.save('aggHist', hist2D)
    plt.imshow(np.log(hist2D+1), cmap=plt.cm.hot, aspect='auto')
    plt.show()    
    
    
def batchRunACRExtraction(root_dir, exePath):
    
    filenames = BP.GetFileNamesInDir(root_dir, '.pitch')
        
    for filename in filenames:
        fname,ext = os.path.splitext(filename)
        
        cmd = exePath+'/'+'ComputeACR_O3 ' + fname+ " '.pitch' '.tonic' '.taniSeg' '.acr' 2.0 30 1"
        os.system(cmd)
        

def plotAgg2dACRHistogramWithContours(histFile):
    """
    This function plots 2 d histogram of ACR values along with contours which represent the % of total popoulation below that ACR value. A contour of 80% will represent a line where 80% of the patterns ACR value is below this line.

    """
    cumulativeValues = [25 50 75]

    data = np.load(histFile)
    contours = np.zeros((len(cumulativeValues),data.shape[1]))
    #lets get ACR VALUE at every lag where the cumulative distribution becomes more than cumulativeValues % of the total number of excerpts
    for ii in range(data.shape[1]):
        cumulateDist = [np.sum(data[0:k,ii] for k in range(data.shape[0])]
        cumValCrossInd = []
        for val in cumulativeValues:
            contours[jj,ii] = min(np.where(cumulateDist>=val)[0])


    plt.imshow(np.log(hist2D+1), cmap=plt.cm.hot, aspect='auto')
    
    plt,show()


