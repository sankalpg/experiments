import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP

serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/hindustani/'
localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/'


def getAnotsPerCategory(queryFileList, anotExt = '.anot'):
    
    lines = open(queryFileList,"r").readlines()
    anotPC={}
    for ii, line in enumerate(lines):
        filename = line.strip() + anotExt
        annotations = np.loadtxt(changePrefix(filename))
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

def changePrefix(audiofile):
    
    if audiofile.count(serverPrefix):
        audiofile = localPrefix + audiofile.split(serverPrefix)[1]
    return audiofile

def evaluateSupSearch(searchPatternFile, queryFileList, anotExt = '.anot', fileListExt = '.flist', TopNResult = 10):
    """
    This code assumes that the format of the searchPatternFile is
    <file number of the query file> <line number of the query in the query file>  <file number of the match (the number is the line number of the file specified in the fileList of query File)> <time stamps of the match>
    
    
    queryFileList has a list of files for which the query is performed. queryFile has queries, fileList to be searched and annotations.
    """
    
    queryWiseRes=[]
    pattIdWiseRes={}
    
    #fetching the list of file names for each which the searching operation is performed for all the queries in that file    
    queryFileNames = open(queryFileList).readlines()
    
    #read the output file containing search results
    searchResults = np.loadtxt(searchPatternFile)
    
    #initializing an array which will contain the decision values, wheather the hit is correct or not and if its correct second and third colum will contain the file index and annotation line number
    decArray=np.zeros((searchResults.shape[0],3))
    
    #iterate over all the files for which query is performed
    qFileIndUnique = np.unique(searchResults[:,0])
    for ii, fileInd in enumerate(qFileIndUnique):
        
        fileInd = int(fileInd)
        
        #Lets fetch all the annotations in the search space of a query file
        anots = getAnotsPerCategory(changePrefix(queryFileNames[fileInd].strip()+fileListExt), anotExt)
        
        #find all the index where the a query is performed from query file with index fileInd
        qFileInd = np.where(searchResults[:,0]==fileInd)[0]
        
        #obtain all the queries of this query File
        queryList = np.loadtxt(changePrefix(queryFileNames[fileInd].strip())+anotExt)
        
        #obntaining all the unique query index
        qIndUnique = np.unique(searchResults[qFileInd,1])
        
        #for one file iterate over all the lines (queries)
        for jj, queryInd in enumerate(qIndUnique):
            
            #finding index of results of a particular query
            indSingleQuery = np.where(searchResults[qFileInd,1]==queryInd)[0]
            
            #finding pattern id/category of this query
            patternID = queryList[int(queryInd), 2]
            if not pattIdWiseRes.has_key(patternID):
                pattIdWiseRes[patternID]={}
            if not pattIdWiseRes[patternID].has_key(fileInd):
                pattIdWiseRes[patternID][fileInd]={}             
            if not pattIdWiseRes[patternID][fileInd].has_key(queryInd):
                pattIdWiseRes[patternID][fileInd][queryInd]=[]
                
            #getting indices of the searched files
            indSearchFilesUnique = np.unique(searchResults[qFileInd[indSingleQuery],2])
            
            #iterate over all the searched files !! (optimzed way!!)
            for kk, searchFileInd in enumerate(indSearchFilesUnique):
                
                #obtaining indices of search results corresponding to a particular search file of a particular query of a particular query file
                indSingleSearchFile = np.where(searchResults[qFileInd[indSingleQuery][:TopNResult],2]==searchFileInd)[0]
                
                if len(indSingleSearchFile) ==0:
                    continue
                
                searchedPattensInFile = searchResults[qFileInd[indSingleQuery[indSingleSearchFile]],3:5]
                
                ind1 = np.where(anots[patternID][:,0]==searchFileInd)[0]
                annotationsInFile = anots[patternID][ind1,2:4]
                
                if len(annotationsInFile)>0:
                    dec =  checkPatternSearchHits(searchedPattensInFile, annotationsInFile)
                else:
                    dec = np.zeros((searchedPattensInFile.shape[0],2))
                
                for pp in range(searchedPattensInFile.shape[0]):
                    
                    if dec[pp,0]==1:
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],0]= 1
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],1]= anots[patternID][ind1[dec[pp,1]],0]
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],2]= anots[patternID][ind1[dec[pp,1]],1]
                    else:
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],0]= 0
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],1]= -1
                        decArray[qFileInd[indSingleQuery[indSingleSearchFile[pp]]],2]= -1
                        
            pattIdWiseRes[patternID][fileInd][queryInd] = decArray[qFileInd[indSingleQuery][:TopNResult],0]
            #print patternID, averagePrecision(pattIdWiseRes[patternID][fileInd][queryInd]), np.sum(pattIdWiseRes[patternID][fileInd][queryInd])
            
    f = []
    for patt in pattIdWiseRes.keys():
        a = []
        for qFiles in pattIdWiseRes[patt].keys():
            for q in pattIdWiseRes[patt][qFiles].keys():
                #print patt, qFiles, q, averagePrecision(pattIdWiseRes[patt][qFiles][q])
                f.append(averagePrecision(pattIdWiseRes[patt][qFiles][q]))
                a.append(averagePrecision(pattIdWiseRes[patt][qFiles][q]))
        #print "Total mean average precision for pattern " + str(patt) + " is "  + str(np.mean(a))
    #print "Total mean average precision"  + str(np.mean(f))
    
    return np.mean(f)
    
    
        
                
        
    #np.savetxt('tempDec.txt', decArray, fmt="%d")
    
def evaluateAllResults(root_dir, outputFile, baseName = 'configFiles_', searchResultExt = '.motifSearch', qFileListExt = '.flist', nFiles = 1260, TopNResult=10):
    
    a = []
    for ii in range(nFiles):
        print ii
        filename1 = os.path.join(root_dir, baseName + str(ii+1)+searchResultExt)
        filename2 = os.path.join(root_dir, baseName + str(ii+1)+qFileListExt)
        
        res = evaluateSupSearch(filename1, filename2, '.anot', '.flist', TopNResult)
        a.append(res)
        
    np.savetxt(outputFile, a)
    
    

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
            
def averagePrecision(relevanceArray):
    """
    relevanceArray is 1d array of 0 and 1 indicating if a searched result is relevant (1) or not (0). 
    The output is the average precision of the retrieved relevant documents.
    Ref: 
    http://en.wikipedia.org/wiki/Information_retrieval#Average_precision
    """

    relPrecision = []
    for ii, val in enumerate(relevanceArray):
        p = val*np.sum(relevanceArray[0:ii+1])/float(ii+1)
        relPrecision.append(p)
    
    nRel = np.sum(relevanceArray)
    if nRel ==0:
        return 0
    else:
        return np.sum(relPrecision)/float(nRel)                
                       
        
        
    
                
                
            