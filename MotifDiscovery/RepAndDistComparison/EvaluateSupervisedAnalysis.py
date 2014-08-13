import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP


def getAnotsPerCategory(fileListFile, anotExt = '.anot'):
    
    lines = open(fileListFile,"r").readlines()
    anotPC={}
    for ii, line in enumerate(lines):
        filename = line.strip() + anotExt
        annotations = np.loadtxt(filename)
        if annotations.size ==0:
            continue
        if len(annotations.shape)==1:
            annotations = np.array([annotations])
        for jj in np.arange(annotations.shape[0]):
            line = annotations[jj,:]
            id = int(line[2])
            if not anotPC.has_key(id):
                anotPC[id]=[]
            anotPC[id].append([ii, jj, line[0], line[1]])

    
    for key in anotPC.keys():
        anotPC[key] = np.array(anotPC[key])
        
    return anotPC



def evaluateSupSearch(searchPatternFile, fileListFile, anotExt = '.anot'):
    """
    This code assumes that the format of the searchPatternFile is
    <file number of the query> <line number of the query in the file>  <file number of the match> <time stamps of the match>
    """
    
    fileNames = open(fileListFile).readlines()
    
    #to start with lets get all the annotations for all the files
    anots = getAnotsPerCategory(fileListFile, anotExt)
    
    #read the output file containing search results
    matches = np.loadtxt(searchPatternFile)
    
    decArray=np.zeros((matches.shape[0],2))
    #iterate over all the files (containing queries)
    for ii, fileInd in enumerate(np.unique(matches[:,0])):
        indSingleFile = np.where(matches[:,0]==fileInd)[0]
        
        queryInfo = np.loadtxt(fileNames[int(fileInd)].strip()+anotExt)
        
        #for one file iterate over all the lines (queries)
        for jj, queryInd in enumerate(np.unique(matches[indSingleFile,1])):
            indSingleLine = np.where(matches[indSingleFile,1]==queryInd)[0]
            
            #finding pattern id/category of this query
            patternID = queryInfo[int(queryInd), 2]
            
            #iterate over all the searched files !! (optimzed way!!)
            for kk, searchInd in enumerate(np.unique(matches[indSingleFile[indSingleLine],2])):
                indSingleSearchFile = np.where(matches[indSingleFile[indSingleLine],2]==searchInd)[0]
                
                searchedPattensInFile = matches[indSingleFile[indSingleLine[indSingleSearchFile]],3:5]
                ind1 = np.where(anots[patternID][:,0]==searchInd)[0]
                annotationsInFile = anots[patternID][ind1,2:4]
                #print searchedPattensInFile, annotationsInFile
                dec =  checkPatternSearchHits(searchedPattensInFile, annotationsInFile)
                for pp in range(searchedPattensInFile.shape[0]):
                    decArray[indSingleFile[indSingleLine[indSingleSearchFile]][pp],0]= dec[pp,0]
                    decArray[indSingleFile[indSingleLine[indSingleSearchFile]][pp],1]= anots[patternID][ind1[dec[pp,1]],1]
                
    np.savetxt('tempDec.txt', decArray, fmt="%d")

def checkPatternSearchHits(searchedTS, anotTS, criterion=1):
    
    hitArray = []
    #find max time stamp
    maxT = np.max([np.max(searchedTS),np.max(anotTS)])
    resolution = 10 #ms
    fac = 1000.0/resolution
    annotArray = -1*np.ones(maxT*fac)
    
    for ii, a in enumerate(anotTS):
        annotArray[int(a[0]*fac):int(a[1]*fac)]=ii
        
    if criterion ==1:# in this criterion a hit means if the searched time stamp has any portion of the annotated pattern
        
        for ii, s in enumerate(searchedTS):
            ind = np.where(annotArray[int(s[0]*fac):int(s[1]*fac)]!=-1)[0]
            if len(ind) >0:
                hitArray.append([1, annotArray[int(s[0]*fac):int(s[1]*fac)][ind[0]]])
            else:
                hitArray.append([0, -1])
    hitArray = np.array(hitArray)
    return hitArray
            
                
                       
        
        
    
                
                
            