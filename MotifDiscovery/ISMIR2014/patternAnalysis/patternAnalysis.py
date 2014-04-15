#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2 as psy
import sys, os
from mutagen import easyid3
import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing'))

import batchProcessing as BP
import matplotlib.pyplot as plt

try:
    from mutagen.mp3 import MP3
except:
    pass

EPS = np.finfo(float).eps

myUser = 'sankalp'
myDatabase = 'motifDB_CONF1'

root_path = '/media/Data/Datasets/MotifDiscovery_Dataset/CompMusic/'

annotationFileISMIR2014 = '../motifIdentificationEvaluation/EvaluationData/annotations.txt'
patternInfoFileISMIR2014 = '../motifIdentificationEvaluation/EvaluationData/patternInfo.txt'
evalDataInfoFileISMIR2014 = '../motifIdentificationEvaluation/EvaluationData/evaluationFullData.pkl'


def getAudioDurationForDataSet():
    
    cmd1 = "select filename from file where hasseed=1"
    
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1)
        audiofiles = cur.fetchall()
        audiofiles = [x[0] for x in audiofiles]
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
        
    total_duration = 0
    for audiofile in audiofiles:
        total_duration += computeDutationSong(root_path+audiofile)
        
    print "total duration is %d"%total_duration
    print "total number of files is %d"%len(audiofiles)
    
    return total_duration

def computeDutationSong(audiofile): 
    
    filename, ext = os.path.splitext(audiofile)

    if  ext=='.mp3':
        audio = MP3(audiofile)
        duration = audio.info.length
    elif ext=='.wav':
        duration = ES.MetadataReader(filename = audiofile)()[7]
            
    return duration



def getSeedPatternDistancesTXT(root_dir, seedExt = '.2s25Motif_CONF1', distFile = 'seedDistances'):
    
    filenames = BP.GetFileNamesInDir(root_dir,seedExt)
    distArray = []
    for ii, filename in enumerate(filenames):
        #print "processing %d of %d files"%(ii+1, len(filenames))
        seedData = np.loadtxt(filename)
        indValid = np.where(seedData[:,4]<99999999999999999999)[0]
        distArray.extend(seedData[indValid,4].tolist())
        
    #np.save(distFile, distArray)
    
    return distArray

def computeSeedPatternDistHistogramTXT(root_dir, seedExt = '.2s25Motif_CONF1', nBins=100, plotOrSave=0):
    
    dist = getSeedPatternDistancesTXT(root_dir, seedExt)
    
    dist=np.log10(dist)
    
    min_val = np.min(dist)
    max_val = np.max(dist)
    
    #bins = np.arange(0,max_val, 10000)
    bins = np.linspace(min_val, max_val, num=nBins+1)
    
    hist = np.histogram(dist,bins=bins)
    
    fig = plt.figure()
    fsize=14
    plt.plot(hist[1][:-1], hist[0])
    plt.ylabel("Frequency", fontsize=fsize)
    plt.xlabel("Log distance", fontsize=fsize)
    
    if plotOrSave:
        fig.savefig('seedDistanceDistribution.pdf')
    else:
        plt.show()
    
    return hist[0], hist[1]

def getSeedPatternDistancesDB():
    
    cmd1 = "select distance from match where version <0"
    
    distArray = []
    
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1)
        distArray = cur.fetchall()
       
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
    
    return distArray
    
def computeSeedPatternDistHistogramDB(nBins=100, plotOrSave=0):
    
    dist = getSeedPatternDistancesDB()
    
    dist=np.log10(dist)
    
    min_val = np.min(dist)
    max_val = np.max(dist)
    
    #bins = np.arange(0,max_val, 10000)
    bins = np.linspace(min_val, max_val, num=nBins+1)
    
    hist = np.histogram(dist,bins=bins)
    
    fig = plt.figure()
    fsize=14
    plt.plot(hist[1][:-1], hist[0])
    plt.ylabel("Frequency", fontsize=fsize)
    plt.xlabel("Log distance", fontsize=fsize)
        
    if plotOrSave==1:
        fig.savefig('seedDistanceDistribution.pdf')
    elif plotOrSave==2:
        plt.show()
    
    return hist[0], hist[1]




def createISMIR2014EvaluationSubset(nBins=10, totalPatterns=200, nSearchItems=10, nVersions=4, splitLogic = 1):
    """
    this method will sample the seed pattern space and will select a subset such that they are equally disctibuted over stratas of 
    distances, where each strata is basically equi-spaced bin from min to max in log distance domain.
    
    In addtion for each seed pattern top N searched patterns are selected for different versions of the rank refinement.
    """    

    cmd1 = "select source_id, distance from match where version =-1 order by distance"

    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1)
        seedData = cur.fetchall()
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)

    if con:
        con.close()

    seedData = np.array(seedData)
    distArray = seedData[:,1]
    
    distArrayLOG=np.log10(distArray)

    min_val = np.min(distArrayLOG)
    max_val = np.max(distArrayLOG)
    
    #constructing bins with different different logic, very crucial step.
    
    ### Logic 1: Equal width bins from min_val to max_val
    if splitLogic ==1:
        bins = np.linspace(min_val, max_val, num=nBins+1)
    
    ### Logic 2: for the case of three bins taking  min_val < bin1 < mean-2*std < bin2 < mean+2*std < bin3 < max_val
    if splitLogic ==2:
        if nBins ==3:
            mean = np.mean(distArrayLOG)
            std = np.std(distArrayLOG)
            threshold1 = mean - 2*std
            threshold2 = mean + 2*std
            bins = [min_val, threshold1, threshold2, max_val]
    
    
    #just store indexes for each strata
    indStratas=[]
    for ii in range(len(bins)-1):        
        indMore = np.where(distArrayLOG>=bins[ii])[0]
        indLess = np.where(distArrayLOG<bins[ii+1])[0]
        inds = np.intersect1d(indMore, indLess)
        indStratas.append(inds)
    
    #performing a greedy sampling to cover all bins, selecting one from every bin unless total cnt is reached
    indexSample = []
    category = []
    cnt = totalPatterns
    binIndex = 0
    perBinSamples = np.zeros(nBins)
    while(cnt>0):
        binIndex = np.mod(binIndex,nBins)
        leftSize = indStratas[binIndex].size
        
        if leftSize>0:
            indRand = np.random.randint(leftSize)
            indexSample.extend([indStratas[binIndex][indRand]])
            category.extend([binIndex])
            indStratas[binIndex] = np.delete(indStratas[binIndex],indRand)
            perBinSamples[binIndex]+=1
            cnt-=1
        binIndex+=1
            
    seedSubset = seedData[indexSample,0]
    distSubset = seedData[indexSample,1]
    
    #ok so after subsampling now lets fetch the searched patterns
    evalSubSet = -1*np.ones((2+(nVersions*nSearchItems), totalPatterns))
    
    cmd1 = "select target_id from match where source_id=%d and version= %d order by distance"
    cmd2 = "select pair_id from pattern where id=%d"

    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        
        for ii,seed in enumerate(seedSubset):
            evalSubSet[1,ii]=seed
            cur.execute(cmd2%seed)
            seedPair = int(cur.fetchone()[0])
            evalSubSet[0,ii]=seedPair
            
            for jj in range(nVersions):
                cur.execute(cmd1%(seed,jj))   
                searchData = cur.fetchall()
                searchData = [x[0] for x in searchData]
                searchData = np.array(searchData[:nSearchItems])
                evalSubSet[2+(jj*nSearchItems) : 2 + ((jj+1)*nSearchItems),ii] = searchData
 
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
        
    if con:
       con.close()
    
    data = {'seedSubset':seedSubset, 'distSubset':distSubset, 'category':category, 'bins':bins,  'perBinSamples':perBinSamples}
    fid = open("evaluationFullData.pkl", "w")
    pickle.dump(data, fid)
    fid.close()
    
    np.savetxt("patternInfo.txt", evalSubSet.astype(np.int32), fmt="%ld")
    np.savetxt("evalationInfo.txt", 0*evalSubSet.astype(np.int32), fmt="%ld")
    
    return 1


def generateHeatMapPlotForSeedVsSearchEvalSubSetISMIR(patternInfoFile, versionSet= [0]):
    
    #reading the pattern ids from info filename
    patternData = np.loadtxt(patternInfoFile)
    
    seedPatterns = patternData[1,:]
    seedPatternPairs = patternData[0,:]
    
    searchPatternsAll = patternData[2:,:]
    
    cmd1 = "select distance in match where source_id=%d and target_id=%d"
    
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        distVals = np.array([[0,0]])
        for ii, seed in enumerate(seedPatterns):
            for version in versionSet:
                searchPatterns = searchPatternsAll[ii,version*10:(version+1)*10]
                
                for searches in searchPatterns:
                    cur.execute(cmd1%(seed, searches))
                    distVals.append(np.array([[]]))
                
                
                    
            
        
        
        
        seedData = cur.fetchall()
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)

    if con:
        con.close()
        
        
def dumpDistanceRelations(patternInfoFile, distanceFile):
    
    
    cmd1 = 'select distance from match where source_id =%d and target_id=%d and version=%d'
    
    #reading pattern info file 
    patternInfo = np.loadtxt(patternInfoFile)
    searchedInfo = patternInfo[2:,:]
    distanceMtx = np.zeros(patternInfo.shape)
    nPerVersion=10
    nVersions = 4
    
    
    
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        
        for ii, seed in enumerate(patternInfo[1,:]):
            cur.execute(cmd1%(patternInfo[0,ii],patternInfo[1,ii], -2))
            dist = cur.fetchone()[0]
            distanceMtx[0,ii]=dist
            distanceMtx[1,ii]=dist
            
            for jj in range(nVersions):
                for kk in range(nPerVersion):
                    cur.execute(cmd1%(patternInfo[1,ii],searchedInfo[(jj*nPerVersion)+kk,ii],jj))
                    dist2 = cur.fetchone()[0]
                    distanceMtx[2 + (jj*nPerVersion)+kk,ii]=dist2
            
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)

    if con:
        con.close()
    
    np.savetxt(distanceFile, distanceMtx)
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        PLOTTING FUNCTINOS FOR ISMIR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def fetchSearchDistanceClasswise(distanceInfoFile, annotationFile, version,nPerVersion=10):

    distanceInfo = np.loadtxt(distanceInfoFile)
    annotations = np.loadtxt(annotationFile)

    annotations = annotations[2:,:]
    distanceInfo = distanceInfo[2:,:]

    searchAnnots = annotations[version*nPerVersion:(version+1)*nPerVersion,:]
    distanceInfo = distanceInfo[version*nPerVersion:(version+1)*nPerVersion,:]


    indBadMatch = np.where(searchAnnots==0)
    indGoodMatch = np.where(searchAnnots>0)

    distGood = distanceInfo[indGoodMatch]
    distBad = distanceInfo[indBadMatch]

    distGood = np.ndarray.flatten(distGood)
    distBad = np.ndarray.flatten(distBad)

    return distGood, distBad

def fetchSeedDistanceClasswise(distanceInfoFile, annotationFile):

    distanceInfo = np.loadtxt(distanceInfoFile)
    annotations = np.loadtxt(annotationFile)

    annotations = annotations[1,:]
    distanceInfo = distanceInfo[1,:]

    indBadMatch = np.where(annotations==0)
    indGoodMatch = np.where(annotations>0)

    distGood = distanceInfo[indGoodMatch]
    distBad = distanceInfo[indBadMatch]

    return distGood, distBad

def computeDistanceDistribution(distGood, distBad, takeLog =1, nBins=100):
    if takeLog:
        distGoodLog  = np.log10(distGood+1)
        distBadLog = np.log10(distBad+1)
    else:
        distGoodLog  = distGood
        distBadLog = distBad


    min_val = np.min([np.min(distGoodLog), np.min(distBadLog)])
    max_val = np.min([np.max(distGoodLog), np.max(distBadLog)])

    bins = np.linspace(min_val, max_val, num=nBins)

    hist1 = np.histogram(distGoodLog, bins = bins)
    hist2 = np.histogram(distBadLog, bins = bins)

    return hist1[0], hist2[0], hist1[1]



def computeROC(class1Data, class2Data, nSteps = 1000):
    """
    returns true positve , false positive for each step at equal distance from min (data) to max (data)
    """

    minData = np.min([np.min(class1Data), np.min(class2Data)])
    maxData = np.max([np.max(class1Data), np.max(class2Data)])

    thresholds = np.linspace(minData, maxData, num= nSteps)

    truePos = []
    falsePos = []

    for thresh in thresholds:
        indPos = np.where(class1Data<=thresh)[0]
        truePos.append(float(len(indPos))/float(len(class1Data)))

        indNeg = np.where(class2Data<=thresh)[0]
        falsePos.append(float(len(indNeg))/float(len(class2Data)))

    return truePos, falsePos


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


def meanAvgPrecision(annotationFile, version, nPerVersion=10, seedCategory = [0, 1, 2]):

    annotations = np.loadtxt(annotationFile)
    annotations = annotations[2:,:]

    nCols = annotations.shape[1]

    indexes = []
    for cat in  seedCategory:
        indexes.extend(range(cat, nCols, 3))

    #print len(indexes)
    versionAnnots = annotations[version*nPerVersion : (version+1)*nPerVersion, indexes]

    relevanceMTX = np.zeros(versionAnnots.shape)
    indRel = np.where(versionAnnots>0)
    relevanceMTX[indRel] =1

    avgPrecision = []
    for ii in range(relevanceMTX.shape[1]):
        avgPrecision.append(averagePrecision(relevanceMTX[:,ii]))

    return np.mean(avgPrecision), np.array(avgPrecision)



def numbersTopNCombined(patternInfoFile, annotationFile, nPerVersion=10):
    
    #reading the pattern ids from info filename
    patternData = np.loadtxt(patternInfoFile)
    patternData = patternData[2:,:]

    annotations = np.loadtxt(annotationFile)
    annotations = annotations[2:,:]

    indGood = np.where(annotations >0)
    patternsGood = patternData[indGood]

    indBad = np.where(annotations ==0)
    patternsBad = patternData[indBad]

    patternData = np.ndarray.flatten(patternData)
    patternsGood = np.ndarray.flatten(patternsGood)
    patternsBad = np.ndarray.flatten(patternsBad)

    print "Number of Unique patterns in the searched results of all methods are : %d " %(len(np.unique(patternData)))
    print "Number of Unique Good patterns in the searched results of all methods are : %d " %(len(np.unique(patternsGood)))
    print "Number of Unique Bad patterns in the searched results of all methods are : %d " %(len(np.unique(patternsBad)))
    print "Number of intersection in Good and  Bad patterns in the searched results of all methods are : %d " %(len(np.intersect1d(patternsGood, patternsBad)))
    return 

def meanReciprocalRankFist(annotationFile, version, nPerVersion=10, seedCategory = [0,1,2]):


    annotations = np.loadtxt(annotationFile)
    annotations = annotations[2:,:]

    nCols = annotations.shape[1]

    indexes = []
    for cat in  seedCategory:
        indexes.extend(range(cat, nCols, 3))

    versionAnnots = annotations[version*nPerVersion: (version+1)*nPerVersion, indexes]
    RR = []
    for ii in range(versionAnnots.shape[1]):
        indGood = np.where(versionAnnots[:,ii]>0)[0]
        if len(indGood)>0:
            indGood = np.min(indGood)
            RR.append(1/(1+float(indGood)))
        else:
            RR.append(0.0)

    return np.mean(RR)


def plotMeanAvgPrecision(annotationFile, plotName = -1):


    seedCategoryArray = [[0], [1], [2], [0,1,2]]
    CategoryNames = ['Cat.1', 'Cat.2', 'Cat.3', 'all']
    versionArray = [0,1,2,3]
    #totalCnt = np.array([670, 670, 660, 2000]).astype(np.float)
    plotData = np.zeros((len(seedCategoryArray), len(versionArray)))

    for ii in range(len(seedCategoryArray)):
        for jj in range(len(versionArray)):
            plotData[ii,jj] = meanAvgPrecision(annotationFile, jj, nPerVersion=10, seedCategory = seedCategoryArray[ii])[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    colorArr = ['c', 'g', 'm', 'k']
    markerArr = ['^', 's' , 'D', 'o']
    pLeg = []
    for ii in range(len(seedCategoryArray)):
        phand = plt.scatter([1,2,3,4], plotData[ii,:], color = colorArr[ii], marker = markerArr[ii], s = 50)
        pLeg.append(phand)

    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    plt.xticks([1,2,3,4])
    plt.xlim([0,5])
    plt.ylim([0,1])
    plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.ylabel("Mean average precision", fontsize = fsize, fontname=font, labelpad=fsize2)
    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 4, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    ax.set_aspect(5/(1*2))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)

    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1


def plotmeanReciprocalRankFist(annotationFile, plotName = -1):


    seedCategoryArray = [[0], [1], [2], [0,1,2]]
    CategoryNames = ['Cat.1', 'Cat.2', 'Cat.3', 'all']
    versionArray = [0,1,2,3]
    plotData = np.zeros((len(seedCategoryArray), len(versionArray)))

    for ii in range(len(seedCategoryArray)):
        for jj in range(len(versionArray)):
            plotData[ii,jj] = meanReciprocalRankFist(annotationFile, jj, nPerVersion=10, seedCategory = seedCategoryArray[ii])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    colorArr = ['c', 'g', 'm', 'k']
    markerArr = ['^', 's' , 'D', 'o']
    pLeg = []
    for ii in range(len(seedCategoryArray)):
        phand = plt.scatter([1,2,3,4], plotData[ii,:], color = colorArr[ii], marker = markerArr[ii], s = 50)
        pLeg.append(phand)

    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.ylabel("Mean Reciprocal Rank", fontsize = fsize, fontname=font, labelpad=fsize2)
    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='lower right', ncol = 4, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    
    
    plt.xlim([0,5])
    plt.ylim([.25,1.05])
    ax.set_aspect(5/(.8*2))
    plt.xticks([1,2,3,4])
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)

    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1


def plotBoxPlotAveragePrecision(annotationFile, plotName = -1):

    seedCategoryArray = [[0], [1], [2]]
    versionArray = [0,1,2,3]
    
    plotData = []
    for ii in range(len(seedCategoryArray)):
        for jj in range(len(versionArray)):
            data = meanAvgPrecision(annotationFile, jj, nPerVersion=10, seedCategory = seedCategoryArray[ii])[1]
            plotData.append(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.boxplot(plotData)

    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    plt.xticks(np.arange(12)+1, ['V1','V2', 'V3', 'V4', 'V1','V2', 'V3', 'V4', 'V1','V2', 'V3', 'V4'])
    #plt.xlim([0,5])
    plt.ylim([-.1,1.1])
    #plt.xlabel("Version of the rank refinement method", fontsize = fsize, fontname=font)
    plt.ylabel("Average precision", fontsize = fsize, fontname=font, labelpad=fsize2)
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
    
def plotSeedDistVsSearchDist(distanceInfoFile, version, nPerVersion=10, plotName=-1):

    distanceInfo = np.loadtxt(distanceInfoFile)

    scatterData = []

    for ii,seed in enumerate(distanceInfo[1,:]):
        for jj in range(nPerVersion):
            scatterData.append((seed, distanceInfo[2+(version*nPerVersion) + jj, ii]))

    x_scat = np.log10(np.array([x[0] for x in scatterData])+1)
    y_scat = np.log10(np.array([x[1] for x in scatterData])+1)
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(x_scat, y_scat, s=50, alpha=0.75, marker = '1', color = 'k')
    
    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel("Seed pair distance (log)", fontsize = fsize, fontname=font)
    plt.ylabel("Distance (log)", fontsize = fsize, fontname=font, labelpad=fsize2)
    
    xmin = np.min(x_scat)
    xmax = np.max(x_scat)
    xrang = xmax-xmin

    ymin = np.min(y_scat)
    ymax = np.max(y_scat)
    yrang = ymax-ymin
    
    xLim = [xmin-(0.05*xmin), xmax+(0.05*xmax)]
    yLim = [ymin-(0.05*ymin), ymax+(0.05*ymax)]

    plt.xlim(xLim)
    plt.ylim(yLim)
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #ax.tick_params(axis='y', pad=30)

    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1


def plotSearchDistDistributions(distanceInfoFile, annotationFile, version,nPerVersion=10, takeLog =1, nBins=100, plotName=-1):

    d1,d2 = fetchSearchDistanceClasswise(distanceInfoFile, annotationFile, version, nPerVersion)

    h1,h2,bins = computeDistanceDistribution(d1,d2, takeLog, nBins)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pLeg=[]
    CategoryNames = ['similar', 'not-similar']

    plt.hold(True)
    p, = plt.plot(bins[:-1], h1, color = 'b')
    pLeg.append(p)
    p, = plt.plot(bins[:-1], h2, color = 'r')
    pLeg.append(p)

    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel("Distance (log)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency", fontsize = fsize, fontname=font, labelpad=fsize2)

    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='upper left', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1

def plotSeedDistDistributions(distanceInfoFile, annotationFile, takeLog =1, nBins=100, plotName=-1):

    d1,d2 = fetchSeedDistanceClasswise(distanceInfoFile, annotationFile)

    h1,h2,bins = computeDistanceDistribution(d1,d2, takeLog, nBins)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    CategoryNames = ['similar', 'not-similar']
    pLeg = []

    plt.hold(True)
    
    p, = plt.plot(bins[:-1], h1, color = 'b')
    pLeg.append(p)
    p, = plt.plot(bins[:-1], h2, color = 'r')
    pLeg.append(p)
    
    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel("Distance (log)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency", fontsize = fsize, fontname=font, labelpad=fsize2)

    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='upper left', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1


def plotSearchDistROC(distanceInfoFile, annotationFile, version,nPerVersion=10, takeLog =1, steps=1000, plotName=-1):

    d1,d2 = fetchSearchDistanceClasswise(distanceInfoFile, annotationFile, version, nPerVersion)

    if takeLog:
        d1 = np.log10(d1+1)
        d2 = np.log10(d2+1)

    tp, fp = computeROC(d1,d2, nSteps = steps)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    
    plt.plot(fp, tp, color = 'b')
    
    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel(" False Positives", fontsize = fsize, fontname=font)
    plt.ylabel("True Positives", fontsize = fsize, fontname=font, labelpad=fsize2)

    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1


def plotSeedDistROC(distanceInfoFile, annotationFile,takeLog =1, steps=1000, plotName=-1):

    d1,d2 = fetchSeedDistanceClasswise(distanceInfoFile, annotationFile)

    if takeLog:
        d1 = np.log10(d1+1)
        d2 = np.log10(d2+1)

    tp, fp = computeROC(d1,d2, nSteps = steps)

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    
    plt.plot(fp, tp, color = 'b')
    
    fsize = 20
    fsize2 = 14
    font="Times New Roman"
    
    plt.xlabel(" False Positives", fontsize = fsize, fontname=font)
    plt.ylabel("True Positives", fontsize = fsize, fontname=font, labelpad=fsize2)

    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1

def obtainInconsistentAnnotations(annotationFile, patternInfoFile):
    
    annotations = np.loadtxt(annotationFile)
    patternInfo = np.loadtxt(patternInfoFile)
    fid = open('inconsistent.txt', 'w')
    fid.close()
    
    for ii in range(patternInfo.shape[1]):
        patternUnique = np.unique(patternInfo[2:,ii])
        
        for patt in patternUnique:
            
            indPat = np.where(patternInfo[2:,ii]==patt)[0]
            
            ratingRat = annotations[2+indPat,ii]
            ratingUniq = np.unique(ratingRat)
            if len(ratingUniq)>1:
                fid = open('inconsistent.txt', 'ab')
                fid.write("Seed Index =%d | "%(ii+1))
                for jj in range(len(ratingRat)):
                    v,r = index2VersionInd(indPat[jj], 10)
                    fid.write(" | version = %d, searchIndex = %d, ratingGiven = %d | "%(v+1,r+1,ratingRat[jj]))
                fid.write("\n")
                fid.close()
                    
                
                
def index2VersionInd(ind, nPerVersion):
    
    quot = np.mod(ind,nPerVersion)
    return (ind-quot)/nPerVersion , quot
    
    
