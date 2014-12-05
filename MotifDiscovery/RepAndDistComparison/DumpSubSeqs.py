import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP

eps =  np.finfo('float').eps

"""
This file has functions to mainly dump the subsequences needed for experiments related with supervised anlaysis where we evaluate several similarity methodologies
"""
serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/hindustani/IITB_Dataset_New/'
localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/IITB_Dataset_New/'

def changePrefix(audiofile):
    
    if audiofile.count(serverPrefix):
        #audiofile = localPrefix + audiofile.split(serverPrefix)[1]
        audiofile = audiofile.replace(serverPrefix, localPrefix)
        print audiofile
    return audiofile

def getPatternLengthFile(fname, annotExt='.anot'):
    
    pattInFile = np.loadtxt(fname)
    if pattInFile.shape[0] == pattInFile.size:
        #this is when there is only a single line in the annotatin file
        durs = pattInFile[1]-pattInFile[0]
        return([durs.tolist()])
    else:
        durs = pattInFile[:,1]-pattInFile[:,0]
        return(durs.tolist())


def getPatternLengthsDB(fileList, annotExt='.anot'):
    
    lines = open(fileList,"r").readlines()
    pattLen = []
    for line in lines:
        line = line.strip()
        fname = changePrefix(line + annotExt)
        print fname
        pattLen.extend(getPatternLengthFile(fname))
    return pattLen
  
def computeTotalDuratoin(fileList, pitchExt):
    
    lines = open(fileList,"r").readlines()
    dur =0
    for line in lines:
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        dur+=pitchTime[-1,0]
    return dur

def computeNoiseSamplingFrequency(fileList, pitchExt, anotExt, nNoiseCands):
    
    lines = open(fileList,"r").readlines()
    fileDur = 0
    pattDur = 0
    for line in lines:
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        pattLens = getPatternLengthFile(changePrefix(line + anotExt))
        fileDur+=pitchTime[-1,0]
        pattDur+=np.sum(pattLens)
        
    return fileDur, pattDur, (fileDur-pattDur)/float(nNoiseCands)

def nearestInd(arr, val):
    return np.argmin(abs(arr-val))

def dumpQuerySubsequences(fileList, pitchExt, anotExt, tonicExt, nSamplesSub, infoOutFile, subOutFilename, subOutFilenameTonicNorm):
    
    
    subOutFile = open(subOutFilename, "ab")
    subOutFileTonicNorm = open(subOutFilenameTonicNorm, "ab")
    infoOutFile = open(infoOutFile, "ab")
    
    lines = open(fileList,"r").readlines()
    for jj, line in enumerate(lines):
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        tonic = float(np.loadtxt(changePrefix(line + tonicExt)))
        hopPitch = pitchTime[1,0]-pitchTime[0,0]
        
        annots = np.loadtxt(changePrefix(line + anotExt))
        
        if annots.shape[0] ==annots.size:
            annots = np.array([annots])
        
        for ii in range(annots.shape[0]):
            start = annots[ii,0]
            end = annots[ii,1]
            pattID = annots[ii,2]
            
            infoOutFile.write("%f\t%f\t%d\t%d\n"%(start, end-start, jj, ii))#start, duration, fileID(lineNumber), pattID
            indStart = nearestInd(pitchTime[:,0], start)
            pitchCents = 1200*np.log2(copy.copy(pitchTime[indStart:indStart+nSamplesSub,1])/55.0)
            subOutFile.write(pitchCents)
            subOutFileTonicNorm.write(pitchCents-(1200*np.log2(tonic/55.0)))
            
            
    subOutFile.close()
    subOutFileTonicNorm.close()
    infoOutFile.close()
            

def dumpNoiseCandidatesSubsequences(fileList, pitchExt, anotExt, tonicExt, hopSizeCandidates, nSamplesSub, infoOutFile, subOutFilename, subOutFilenameTonicNorm):
    
    open(subOutFilename, "w").close()
    open(subOutFilenameTonicNorm, "w").close()
    open(infoOutFile, "w").close()
    
    subOutFile = open(subOutFilename, "ab")
    subOutFileTonicNorm = open(subOutFilenameTonicNorm, "ab")
    infoOutFile = open(infoOutFile, "ab")
    
    pattLens = getPatternLengthsDB(fileList)
    
    lines = open(fileList,"r").readlines()
    for jj, line in enumerate(lines):
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        tonic = float(np.loadtxt(changePrefix(line + tonicExt)))
        hopPitch = pitchTime[1,0]-pitchTime[0,0]
        hopSamples = np.ceil(hopSizeCandidates/hopPitch)
        
        annots = np.loadtxt(changePrefix(line + anotExt))
        
        if annots.shape[0] ==annots.size:
            annots = np.array([annots])
        
        for ii in range(annots.shape[0]):
            start = annots[ii,0]
            end = annots[ii,1]
            indStart = nearestInd(pitchTime[:,0], start)
            indend = nearestInd(pitchTime[:,0], end)
            
            pitchTime = np.delete(pitchTime, range(indStart,indend),0)
        
        nCandidates = int(np.floor((pitchTime.shape[0] - nSamplesSub)/hopSamples))
        
        for ss in range(nCandidates):
            indStart = np.floor(ss*hopSamples)
            infoOutFile.write("%f\t%f\t%d\t%d\n"%(pitchTime[indStart,0], pattLens[np.random.randint(len(pattLens))], jj, -1))#start, duration, 
            pitchCents = 1200*np.log2(copy.copy(pitchTime[indStart:indStart+nSamplesSub,1])/55.0)
            subOutFile.write(pitchCents)
            subOutFileTonicNorm.write(pitchCents-(1200*np.log2(tonic/55.0)))
            
            
    subOutFile.close()
    subOutFileTonicNorm.close()
    infoOutFile.close()
    
    
def dumpSubsequencesQueryAndNoise(fileList, pitchExt, anotExt, tonicExt, hopSizeCandidates, nSamplesSub, infoOutFile, subOutFilename, subOutFilenameTonicNorm, filterLengthAnot):
    
    open(subOutFilename, "w").close()
    open(subOutFilenameTonicNorm, "w").close()
    open(infoOutFile, "w").close()
    
    subOutFile = open(subOutFilename, "ab")
    subOutFileTonicNorm = open(subOutFilenameTonicNorm, "ab")
    infoOutFile = open(infoOutFile, "ab")
    
    pattLens = getPatternLengthsDB(fileList)
    pattLens = np.array(pattLens)
    
    if filterLengthAnot>0:
      indDel = np.where(pattLens>filterLengthAnot)[0]
      pattLens = np.delete(pattLens, indDel)
      print "Max patt length is " + str(np.max(pattLens))
    
    lines = open(fileList,"r").readlines()
    
    # This loop dumps the query info and subseqs
    for jj, line in enumerate(lines):
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        tonic = float(np.loadtxt(changePrefix(line + tonicExt)))
        hopPitch = pitchTime[1,0]-pitchTime[0,0]        
        annots = np.loadtxt(changePrefix(line + anotExt))
        
        if annots.shape[0] ==annots.size:
            annots = np.array([annots])
        
        for ii in range(annots.shape[0]):
            start = annots[ii,0]
            end = annots[ii,1]
            pattID = annots[ii,2]
            
            if filterLengthAnot>0:
              if end-start <= filterLengthAnot:
                infoOutFile.write("%f\t%f\t%d\t%d\n"%(start, end-start, jj, ii))#start, duration, fileID(lineNumber), pattID
                indStart = nearestInd(pitchTime[:,0], start)
                pitchCents = 1200*np.log2(copy.copy(pitchTime[indStart:indStart+nSamplesSub,1] + eps)/55.0)
                subOutFile.write(pitchCents)
                subOutFileTonicNorm.write(pitchCents-(1200*np.log2(tonic/55.0)))
            else:
              infoOutFile.write("%f\t%f\t%d\t%d\n"%(start, end-start, jj, ii))#start, duration, fileID(lineNumber), pattID
              indStart = nearestInd(pitchTime[:,0], start)
              pitchCents = 1200*np.log2(copy.copy(pitchTime[indStart:indStart+nSamplesSub,1] + eps)/55.0)
              subOutFile.write(pitchCents)
              subOutFileTonicNorm.write(pitchCents-(1200*np.log2(tonic/55.0)))              
    
    
    #loop to dump noise candidates
    for jj, line in enumerate(lines):
        line = line.strip()
        pitchFile = changePrefix(line + pitchExt)
        pitchTime = np.loadtxt(pitchFile)
        tonic = float(np.loadtxt(changePrefix(line + tonicExt)))
        hopPitch = pitchTime[1,0]-pitchTime[0,0]
        hopSamples = np.ceil(hopSizeCandidates/hopPitch)
        
        annots = np.loadtxt(changePrefix(line + anotExt))
        
        if annots.shape[0] ==annots.size:
            annots = np.array([annots])
        # This loop removes all the query segments from the pitch array
        for ii in range(annots.shape[0]):
            start = annots[ii,0]
            end = annots[ii,1]
            indStart = nearestInd(pitchTime[:,0], start)
            indend = nearestInd(pitchTime[:,0], end)
            
            pitchTime = np.delete(pitchTime, range(indStart,indend),0)
        
        # Total number of noise candidates to be added in this file based on a hop specified by the user
        nCandidates = int(np.floor((pitchTime.shape[0] - nSamplesSub)/hopSamples))
        
        for ss in range(nCandidates):
            indStart = np.floor(ss*hopSamples)
            infoOutFile.write("%f\t%f\t%d\t%d\n"%(pitchTime[indStart,0], pattLens[np.random.randint(len(pattLens))], jj, -1))#start, duration, 
            pitchCents = 1200*np.log2(copy.copy(pitchTime[indStart:indStart+nSamplesSub,1])/55.0)
            subOutFile.write(pitchCents)
            subOutFileTonicNorm.write(pitchCents-(1200*np.log2(tonic/55.0)))
            
    #closing all the files
    subOutFile.close()
    subOutFileTonicNorm.close()
    infoOutFile.close()    

  
    
    
