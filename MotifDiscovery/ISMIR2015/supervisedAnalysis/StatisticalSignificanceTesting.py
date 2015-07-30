import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import copy
from scipy.stats import wilcoxon

sys.path.append('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RepAndDistComparison/')
import EvaluateSupervisedAnalysis as eval

outputFoldersHindustani = [21, 31, 36, 37, 38, 39]
configFilesHindustani = [462, 462, 492, 492, 532, 532]

outputFoldersCarnatic = [5, 17, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31]
configFilesCarnatic = [524, 524, 524, 524, 454, 454, 454, 494, 494, 494,454,494]



def performStatisticalSignificanceTesting(base_dir, outputFile, tradition):
    
    config_file = 'configFiles_%d'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/%sDB/subSeqDB/'%tradition
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/%sDB/subSeqDB/'%tradition
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/%sDB/audioCollection/AllFiles.flist'%tradition
    
    if tradition == 'hindustani':
        outputfolders = outputFoldersHindustani
        configFiles = configFilesHindustani
        anotExt = '.anotEdit4'
    elif tradition == 'carnatic':
        outputfolders = outputFoldersCarnatic
        configFiles = configFilesCarnatic
        anotExt = '.anotEdit1'
    perQAcc = []
    for ii, folder in enumerate(outputfolders):
        dbFilePath = os.path.join(base_dir+str(folder), methoVariant, config_file%configFiles[ii]+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(folder), methoVariant, config_file%configFiles[ii]+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt=anotExt)
        print np.mean(output[0])
        perQAcc.append(output[0])
    
    nResults = len(outputfolders)
    pVals = []
    combinations = []
    for ii in range(nResults):
        for jj in range(ii+1, nResults):
            pVals.append(wilcoxon(perQAcc[ii], perQAcc[jj])[1])
            combinations.append([outputfolders[ii], outputfolders[jj]])
    
    pVals = np.array(pVals)
    sortInds = np.argsort(pVals)    
    pValsSorted = pVals[sortInds]
    combinations = np.array(combinations)
    N = len(pVals)
    pThsld = 0.01
    condition = True
    index = 0
    while condition:
        print pValsSorted[index]
        if pValsSorted[index]< (pThsld)/(N-index):
            condition = True
            index+=1
        else:
            condition = False
    
    print "These are the statistically significant differences\n"
    fid = open(outputFile, 'w')
    for ii in range(index):
        elem = combinations[sortInds[ii]]
        print "out%d and out%d with p value = %0.20f\n"%(elem[0], elem[1], pValsSorted[ii])
        fid.write("out%d and out%d with p value = %0.20f\n"%(elem[0], elem[1], pValsSorted[ii]))
    fid.close()
    return 1
    
    
