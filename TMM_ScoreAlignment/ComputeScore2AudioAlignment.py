import numpy as np
import sys,os
import copy
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '../../library_pythonnew/batchProcessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../library_pythonnew/similarityMeasures/dtw/'))
from scipy.interpolate import interp1d
import batchProcessing as BP
import dtw


def ComputeScore2AudioAlignment(queryFile, targetFile):
    
    #reading queryFile
    queryData = np.loadtxt(queryFile)
    #reading targetFile
    targetData = np.loadtxt(targetFile)
    
    #interpolating query to match target length. Making them equal length has lots of benefits. But we have to be very careful at this step
    #calculating factor by which query has to be upsampled (this if less than 1 means downsampling)
    factor = (targetData.shape[0]-1)/(  float(queryData.shape[0])-1)
    factorArray = np.arange(targetData.shape[0])/factor    
    pitch = np.transpose(np.array([queryData[(np.round(factorArray)).astype(np.int),1]]))
    b = queryData[:,0][0] + (queryData[:,0]-queryData[:,0][0])*factor
    intFunc = interp1d(np.arange(queryData.shape[0]), b, kind='linear')
    time = np.transpose(np.array([intFunc(factorArray)]))
    queryData = np.concatenate((time, pitch),axis=1)
    
    #removing silence regions or anything undesirable from both the 
    indSil = np.isfinite(queryData[:,1])
    indSil = np.where(indSil==False)[0]
    queryDataDel = np.delete(queryData, [indSil],axis=0)
    
    indSil = np.isfinite(targetData[:,1])
    indSil = np.where(indSil==False)[0]
    targetDataDel = np.delete(targetData, [indSil],axis=0)
    
    dist, pathLen, path, cost = dtw.dtw1d(queryDataDel[:,1], targetDataDel[:,1], {'Output':4, 'Ldistance':{'type':2}, 'Constraint':{'type':''}})
    
    timeStampsQuery = queryDataDel[:,0][path[0]]
    timeStampsTarget = targetDataDel[:,0][path[1]]
    
    return timeStampsQuery, timeStampsTarget
    
    

    