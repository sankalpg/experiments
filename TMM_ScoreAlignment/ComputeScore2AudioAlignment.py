import numpy as np
import sys,os
import copy
import time
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '../../library_pythonnew/batchProcessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../library_pythonnew/similarityMeasures/dtw/'))
from scipy.interpolate import interp1d
import batchProcessing as BP
import dtw


def drawAlignment(x , y , path):
    
    plt.plot(x+53, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    
    plt.figure()
    plt.plot(x+53, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    for ii, row in enumerate(path[0]):
        
        if ii%10==0:
            x1 = path[0][ii]
            y1 = x[path[0][ii]]+53
            
            x2 = path[1][ii]
            y2 = y[path[1][ii]]
            
            plt.plot([x1,x2], [y1,y2], color='r')
        
        
    plt.show()
        
        
        
        
    
    

def ComputeScore2AudioAlignment(queryFile, targetFile):
    
    #reading queryFile
    queryData = np.loadtxt(queryFile)
    #reading targetFile
    targetData = np.loadtxt(targetFile)
    
    #interpolating query to match target length. Making them equal length has lots of benefits. But we have to be very careful at this step
    #calculating factor by which query has to be upsampled (this if less than 1 means downsampling)
    factor = (targetData.shape[0]-1)/(  float(queryData.shape[0])-1)
    #factorArray = np.arange(targetData.shape[0])/factor    
    factorArray = np.linspace(0, queryData.shape[0]-1,targetData.shape[0])    
    pitch = np.transpose(np.array([queryData[(np.round(factorArray)).astype(np.int),1]]))
    #b = queryData[:,0][0] + (queryData[:,0]-queryData[:,0][0])*factor
    intFunc = interp1d(np.arange(queryData.shape[0]), queryData[:,0], kind='linear')
    time = np.transpose(np.array([intFunc(factorArray)]))
    queryData = np.concatenate((time, pitch),axis=1)
    
    
    #removing silence regions or anything undesirable from both the 
    indSil = np.isfinite(queryData[:,1])
    indSil = np.where(indSil==False)[0]
    queryDataDel = np.delete(queryData, [indSil],axis=0)
    
    indSil = np.isfinite(targetData[:,1])
    indSil = np.where(indSil==False)[0]
    targetDataDel = np.delete(targetData, [indSil],axis=0)
    
    dist, pathLen, path, cost = dtw.dtw1dSubLocalBand(queryDataDel[:,1], targetDataDel[:,1], {'Output':4, 'Ldistance':{'type':2}, 'Constraint':{'CVal':int(queryDataDel.shape[0]*0.2)}})
    
    
    #drawAlignment(queryDataDel, targetDataDel, path)
    
    timeStampsQuery = queryDataDel[:,0][path[0]]
    timeStampsTarget = targetDataDel[:,0][path[1]]
    
    return timeStampsQuery, timeStampsTarget
    
def batchProcessAudioScoreAlignment(root_dir):
    
    audioFileNames = BP.GetFileNamesInDir(root_dir, '.query')
    
    for audiofile in audioFileNames:
        print "processing %s\n"%audiofile
        fname, ext = os.path.splitext(audiofile)
        t1, t2 = ComputeScore2AudioAlignment(fname+'.query', fname+'.target')
        t1 = np.transpose(np.array([t1]))
        t2 = np.transpose(np.array([t2]))
        np.savetxt(fname+'.align', np.concatenate((t1, t2),axis=1))



    
    


    
    
