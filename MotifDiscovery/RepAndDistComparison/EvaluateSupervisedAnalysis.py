import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP


#serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/CarnaticAlaps_IITM_edited/'
#localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/CarnaticAlaps_IITM_edited/'


serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/CarnaticAlaps_IITM_edited/'
localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/CarnaticAlaps_IITM_edited/'

#serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/hindustani/IITB_Dataset_New/'
#localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/IITB_Dataset_New/'

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
                #print sum(pattIdWiseRes[patt][qFiles][q])
                f.append(averagePrecision(pattIdWiseRes[patt][qFiles][q], len(anots[patt])-1))
                a.append(averagePrecision(pattIdWiseRes[patt][qFiles][q], len(anots[patt])-1))
        #print "Total mean average precision for pattern " + str(patt) + " is "  + str(np.mean(a))+ '  '+str(np.median(a))
    #print "Total mean average precision"  + str(np.mean(f))+ '   ' + str(np.median(f))
    
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
    
    

def checkPatternSearchHits(searchedTS, anotTS, criterion=2):
    
    hitArray = []
    #find max time stamp
    maxT = np.max([np.max(searchedTS),np.max(anotTS)])
    resolution = 10 #ms
    fac = 1000.0/resolution
    annotArray = -1*np.ones(maxT*fac)
    
    for ii, a in enumerate(anotTS):
        annotArray[int(a[0]*fac):int(a[1]*fac)]=ii
        
    if criterion == 1:# in this criterion a hit means if the searched time stamp has any portion of the annotated pattern
        
        for ii, s in enumerate(searchedTS):
            ind = np.where(annotArray[int(s[0]*fac):int(s[1]*fac)]!=-1)[0]
            if len(ind) >0:
                hitArray.append([1, annotArray[int(s[0]*fac):int(s[1]*fac)][ind[0]]])
            else:
                hitArray.append([0, -1])
    elif criterion == 2:# This criterion uses jaccard index with 50% threshold        
        for ii, s in enumerate(searchedTS):
            ind = np.where(annotArray[int(s[0]*fac):int(s[1]*fac)]!=-1)[0]
            if len(ind) >0:
                indexes1 = range(int(s[0]*fac),int(s[1]*fac))
                a = anotTS[int(annotArray[int(s[0]*fac):int(s[1]*fac)][ind[0]]),:]
                indexes2 = range(int(a[0]*fac),int(a[1]*fac))
                JaccardInd = float(len(np.intersect1d(indexes1, indexes2)))/len(np.unique(np.append(indexes1, indexes2)))
                if JaccardInd > 0.5:
                     hitArray.append([1, annotArray[int(s[0]*fac):int(s[1]*fac)][ind[0]]])
                else:
                    hitArray.append([0, -1])
                    
            else:
                hitArray.append([0, -1])
    
    hitArray = np.array(hitArray)
    return hitArray
"""
def computeJaccardIndex(s1, e1, s2, e2, resolution):
    
    min_val = np.min([s1,e1,s2,e2])
    max_val = np.max([s1,e1,s2,e2])
    
    valArr = n.arange(np.floor(min_val)
"""            
def averagePrecision(relevanceArray, totalRel):
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
    
    return np.sum(relPrecision)/float(totalRel)      
"""    
    nRel = np.sum(relevanceArray)
    if nRel ==0:
        return 0
    else:
        return np.sum(relPrecision)/float(nRel)                
"""                    
        
        
def evaluateSupSearchNEWFORMAT(searchPatternFile, patternInfoFile, fileListDB, anotExt = '.anot'):
  """
  searchPatternFile = output of the code
  patternInfoFile = file in which pattern info is dumped for each subsequence
  fileListDB = filelist using which the subsequences were dumped
  """
  filelistFiles = open(fileListDB,'r').readlines()
  
  #reading the info file and database file to create a mapping
  pattInfos = np.loadtxt(patternInfoFile)
  
  print pattInfos.shape
    
  lineToType = -1*np.ones(pattInfos.shape[0])
  
  #obtaining only query indeces (and not noise candidate indices)
  qInds = np.where(pattInfos[:,3]>-1)[0]
  
  #iterating over all the unique file indices in the queries
  for fileInd in np.unique(pattInfos[qInds][:,2]):
    
    fileInd = int(fileInd)
    #open the annotation file for this index
    annots = np.loadtxt(changePrefix(filelistFiles[fileInd].strip()+anotExt))
    
    if annots.shape[0] == annots.size:
        annots = np.array([annots])
    
    indSingleFile = np.where(pattInfos[qInds][:,2]==fileInd)[0]
    ind = pattInfos[qInds][indSingleFile,3].astype(int).tolist()
    
    for ii,val in enumerate(annots[ind,2]):
      lineToType[qInds[indSingleFile[ii]]] =  val
    
  searchPatts = np.loadtxt(searchPatternFile)
  
  #print searchPatts.shape
  
  line2TypeSearch = np.zeros(searchPatts.shape)
  
  for ii in range(searchPatts.shape[0]):
    line2TypeSearch[ii,0] = lineToType[searchPatts[ii,0]]
    line2TypeSearch[ii,1] = lineToType[searchPatts[ii,1]]
  
  decArray = np.zeros(searchPatts.shape[0])
  
  indCorrect = np.where(line2TypeSearch[:,0]==line2TypeSearch[:,1])
  decArray[indCorrect] = 1
  
  queryByqueryResults = np.zeros(len(qInds))
  pattIdWiseResults = {}
  
  #count frequency of each pattID
  pattIDCnt = {}
  for p in np.unique(lineToType):
    temp = np.where(lineToType==p)[0]
    pattIDCnt[p]= len(temp)
  
  for indQuery in np.unique(searchPatts[:,0]):
    
    indexes = np.where(searchPatts[:,0]==indQuery)[0]
    pattID = lineToType[indQuery]
    
    AveragePrecision = averagePrecision(decArray[indexes], pattIDCnt[pattID]-1)
    
    if not pattIdWiseResults.has_key(pattID):
      pattIdWiseResults[pattID]=[]
    queryByqueryResults[int(indQuery)] = AveragePrecision
    pattIdWiseResults[pattID].append(AveragePrecision)
  
  return queryByqueryResults, pattIdWiseResults
  
  
def evaluateAllResultsNEWFORMAT(root_dir, fileListDB, SummaryFile, baseName = 'configFiles_', searchResultExt = '.motifSearch.motifSearch', dbPathExt = '.flist', infoFileExt= '.subSeqsInfo', nFiles = 560, TopNResult=1000, outputExtQW= '.MAPQW', outputExtCW= '.MAPCW'):
  
    """
    outputExtQW = for every file the code will create a average precision file  for every query with this extension
    outputExtCW = for every file the code will create a mean average precision file  for every category with this extension
    """
    a = []
    for ii in range(nFiles):
        print ii
        tempBase = root_dir + baseName + str(ii+1)
        filename1 = os.path.join(tempBase + searchResultExt)
        filename2 = os.path.join(tempBase + dbPathExt)
        
        pattInfoFile = open(filename2, 'r').readlines()[0].strip() + infoFileExt
        
        res = evaluateSupSearchNEWFORMAT(filename1, changePrefix(pattInfoFile), fileListDB)
        
        pickle.dump(res[0], open(tempBase+outputExtQW,'w'))
        pickle.dump(res[1], open(tempBase+outputExtCW,'w'))
        
        a.append(np.mean(res[0]))
        print max(a)
    np.savetxt(SummaryFile, a)  
    
    
def performStatisticalSigTest(root_dir, SummaryFile, baseName = 'configFiles_', nFiles = 560, outputExtQW= '.MAPQW', alpha = 0.01):

  MAP_perFile = []
  AP_perQuery = []
  for ii in range(nFiles):
          #print ii
          tempBase = root_dir + baseName + str(ii+1)
          filename1 = os.path.join(tempBase + outputExtQW)
          AP = pickle.load(open(filename1,'r'))
          AP_perQuery.append(AP)
          MAP_perFile.append(np.mean(AP))
          
  #print max(MAP_perFile)
  indSort_AP = np.argsort(MAP_perFile)[::-1]
  
  pVals = []
  pValsCord = []
  statSigMTX = -1*np.ones((nFiles,nFiles))
  for ii in range(0,nFiles):
    for jj in range(ii+1, nFiles):
      statSigMTX[ii,jj]= 0.5
      p = wilcoxon(AP_perQuery[indSort_AP[ii]], AP_perQuery[indSort_AP[jj]])
      pVals.append(p[1])
      pValsCord.append((ii,jj))
      statSigMTX[ii,jj] =0
  
  indSort_PVals = np.argsort(pVals)
  M = len(pVals)
  
  cnt = 0
  
  while pVals[indSort_PVals[cnt]] < (alpha/(M - cnt)):
    cnt+=1
  
  
  
  for ii in range(cnt+1):
    statSigMTX[pValsCord[indSort_PVals[ii]][0], pValsCord[indSort_PVals[ii]][1]] = 1
    
  
  plt.figure()
  plt.imshow(statSigMTX)
  plt.show()
  
  pickle.dump(statSigMTX, open(SummaryFile, 'w'))
  
  return 
  
          
        

def getCatWisePatternStats(patternInfoFile, fileListDB, anotExt = '.anot'):
    """
    searchPatternFile = output of the code
    patternInfoFile = file in which pattern info is dumped for each subsequence
    fileListDB = filelist using which the subsequences were dumped
    """
    filelistFiles = open(fileListDB,'r').readlines()
    
    #reading the info file and database file to create a mapping
    pattInfos = np.loadtxt(patternInfoFile)
      
    lineToType = -1*np.ones(pattInfos.shape[0])
    
    #obtaining only query indeces (and not noise candidate indices)
    qInds = np.where(pattInfos[:,3]>-1)[0]
    
    #iterating over all the unique file indices in the queries
    for fileInd in np.unique(pattInfos[qInds][:,2]):
      
        fileInd = int(fileInd)
        #open the annotation file for this index
        annots = np.loadtxt(changePrefix(filelistFiles[fileInd].strip()+anotExt))
        
        if annots.shape[0] == annots.size:
            annots = np.array([annots])
        
        indSingleFile = np.where(pattInfos[qInds][:,2]==fileInd)[0]
        ind = pattInfos[qInds][indSingleFile,3].astype(int).tolist()
        
        for ii,val in enumerate(annots[ind,2]):
          lineToType[qInds[indSingleFile[ii]]] =  val
          
    stats = {}   
    totallen=[]
    for ii in np.unique(lineToType[qInds]):
        stats[ii]={}
        indCat = np.where(lineToType==ii)[0]
        lengths = pattInfos[indCat,1]
        stats[ii]['mean'] = np.mean(lengths)
        stats[ii]['std'] = np.std(lengths)
        stats[ii]['median'] = np.median(lengths)
        stats[ii]['max'] = np.max(lengths)
        stats[ii]['min'] = np.min(lengths)
        stats[ii]['Reps'] = len(lengths)
        totallen.extend(lengths)
        
    stats['total']={}
    stats['total']['mean'] = np.mean(totallen)
    stats['total']['std'] = np.std(totallen)
    stats['total']['median'] = np.median(totallen)
    stats['total']['max'] = np.max(totallen)
    stats['total']['min'] = np.min(totallen)
    stats['total']['Reps'] = len(totallen)
    

    return stats
        
        
def plotBoxPlotMAPPerPattCategory_CMD(MAP_CW_File, anotExt = '.anot', plotName = -1):

    results = pickle.load(open(MAP_CW_File,'r'))
    
    pattCategories = ['$C_1$', '$C_2$', '$C_3$', '$C_4$', '$C_5$']
    pattCatCode = [1000, 2001, 3000, 5000, 8000]
    
    plotData = []
    for key in pattCatCode:
      print "first key " + str(key)
      plotData.append(results[key])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.boxplot(plotData)

    fsize = 16
    fsize2 = 12
    font="Times New Roman"
    plt.xticks(np.arange(5)+1, pattCategories, size=16)
    #plt.xlim([0,5])
    plt.ylim([-.1,1.1])
    #plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.ylabel("Mean average precision (MAP)", fontsize = fsize, fontname=font, labelpad=fsize2)
    plt.xlabel("Pattern type (PT)", fontsize = fsize, fontname=font, labelpad=fsize2)
    #plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 4, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    #plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1   
  
def plotBoxPlotMAPPerPattCategory(MAP_CW_File_CMD, MAP_CW_File_HMD, anotExt = '.anot', plotName = -1):

    results_CMD = pickle.load(open(MAP_CW_File_CMD,'r'))
    results_HMD = pickle.load(open(MAP_CW_File_HMD,'r'))
    
    pattCategories = ['$C_1$', '$C_2$', '$C_3$', '$C_4$', '$C_5$', '$H_1$', '$H_2$', '$H_3$', '$H_4$', '$H_5$']
    pattCatCode_HMD = [1042, 1043, 1044, 1046, 1047]
    pattCatCode_CMD = [1000, 2001, 3000, 5000, 8000]
    
    plotData = []
    for key in pattCatCode_CMD:
      print "first key " + str(key)
      plotData.append(results_CMD[key])
    for key in pattCatCode_HMD:
      print "first key " + str(key)
      plotData.append(results_HMD[key])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.boxplot(plotData)
    
    fsize = 18
    fsize2 = 12
    font="Times New Roman"
    plt.xticks(np.arange(10)+1, pattCategories, size=18)
    #plt.xlim([0,5])
    plt.ylim([-.1,1.1])
    #plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.ylabel("Average precision", fontsize = fsize, fontname=font, labelpad=fsize2)
    plt.xlabel("PT", fontsize = fsize, fontname=font, labelpad=fsize2)
    #plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 4, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(1.75*float(yLim[1]-yLim[0])))
    #plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1     

def plotStatisticanSigMTX(CMD_MTX_file, HMD_MTX_file, plotName = -1):
    
    CMD_MTX = pickle.load(open(CMD_MTX_file, 'r'))
    HMD_MTX = pickle.load(open(HMD_MTX_file, 'r'))
    
    M = CMD_MTX.shape[0]
    
    CMD_pairs = [] 
    for ii in range(M):
      for jj in range(M):
        if CMD_MTX[ii,jj] ==1:
          CMD_pairs.append([ii,jj])
          
    M = HMD_MTX.shape[0]
    
    HMD_pairs = [] 
    for ii in range(M):
      for jj in range(M):
        if HMD_MTX[ii,jj] ==1:
          HMD_pairs.append([ii,jj])          
    
    
    fig = plt.figure()
    ax = fig.add_subplot(121)    
    plt.scatter(np.array(CMD_pairs)[:,0], np.array(CMD_pairs)[:,1], s=10, alpha=0.25, marker = '1', color = 'b')
    
    fsize = 18
    fsize2 = 12
    font="Times New Roman"
    #plt.xticks(np.arange(10)+1, pattCategories, size=18)
    #plt.xlim([0,5])
    plt.ylim([0,560])
    plt.xlim([0,560])
    #plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.title("CMD")
    plt.ylabel("Variant index", fontsize = fsize, fontname=font, labelpad=fsize2)
    plt.xlabel("Variant index", fontsize = fsize, fontname=font, labelpad=fsize2)
    
    
    
    ax1 = fig.add_subplot(122)    
    plt.scatter(np.array(HMD_pairs)[:,0], np.array(HMD_pairs)[:,1], s=10, alpha=0.25, marker = '1', color = 'b')
    
    
    fsize = 18
    fsize2 = 12
    font="Times New Roman"
    #plt.xticks(np.arange(10)+1, pattCategories, size=18)
    #plt.xlim([0,5])
    plt.ylim([0,560])
    plt.xlim([0,560])
    #plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.title("HMD")
    plt.xlabel("Variant index", fontsize = fsize, fontname=font, labelpad=fsize2)
    #plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 4, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    ax1.set_aspect((xLim[1]-xLim[0])/(float(yLim[1]-yLim[0])))
    #plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)
    
    if isinstance(plotName, int):
        fig.tight_layout()
        plt.show()
    elif isinstance(plotName, str):
        fig.tight_layout()
        fig.savefig(plotName, dpi=300)
        
    return 1
    
    
    
def plotPatterns(pitchFile, anotFile, plotName=-1):
  
    pitchData = np.loadtxt(pitchFile)
    anots = np.loadtxt(anotFile)
  
    linePlot = [12, 14, 15]
    freq = [0.5,1, 2,3,4]
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.hold(True)
    
    for ii, line in enumerate(linePlot):
      start = anots[line,0]
      end = anots[line,1]
      start_ind1 = np.argmin(abs(pitchData[:,0]-start))
      end_ind1 = np.argmin(abs(pitchData[:,0]-end))
      pattern1 = 1200*np.log2(pitchData[start_ind1:end_ind1,1]/110.0) + 600*ii
      p, = plt.plot((128/44100.0)*np.arange(pattern1.size), pattern1, linewidth=2, markersize=4.5)
      
    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("time (s)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency (Cents)", fontsize = fsize, fontname=font, labelpad=fsize2)

    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1           
    
  
