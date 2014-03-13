import numpy as np
import os, sys
import json


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/sectionSegmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/machineLearning'))

import sectionSegmentation as seg
import mlWrapper as mlw
from scipy.stats import mannwhitneyu



def generateARFF4DiffConfigs(percFolder, nonPercFolder, output_dir):
    # this function generates different arff files for different ocnfigurations for experiments
    
    AvgLens = [1, 2, 4]
    FrameLens = (1024.0/44100.0)*np.array([1, 2])
    
    class1 = 'perc'
    class2 = 'nonperc'
    
    for AvgLen in AvgLens:
        for FrameLen in FrameLens:
            arffFile = output_dir + '/' + 'MFCC_CENT_FLAT_' + str(int(AvgLen*10)) + '_' + str(int(FrameLen*44100))+'.arff'
            
            cmd = 'python /home/sankalp/Work/Work_PhD/library_pythonnew/sectionSegmentation/sectionSegmentation.py '+percFolder+ ' '+ nonPercFolder+ ' '+ class1+  ' ' + class2 + ' ' +  arffFile + ' ' + str(FrameLen) +' ' + str(AvgLen)
            os.system(cmd)
            #seg.generateBinaryAggMFCCARFF(percFolder, nonPercFolder, class1, class2, arffFile, FrameLen, FrameLen/2.0, AvgLen)
               
 
def performCompoundTrainTesting(arffDir):
    """NOte that this code doesn't care about avoiding tokens from the same song being considered in both training and testing so be careful and write separate code for that if you need it
    """
    AvgLens = [1, 2, 4]
    FrameLens = (1024.0/44100.0)*np.array([1, 2])
    
    class1 = 'perc'
    class2 = 'nonperc'
    
    featuresAll = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'mCent','vCent','mFlat', 'vFlat']
    featureSet1 = featuresAll
    featureSet2 = ['m1', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm13', 'v1', 'v2', 'v4', 'v12', 'v13', 'vCent', 'vFlat']
    featureSets = [featureSet1, featureSet2]
    classifierSets = [('nbMulti', {'fit_prior':False}), ('kNN',{"n_neighbors":5}), ('tree',{'min_samples_split':10}), ('logReg',{'class_weight':'auto'}), ('svm',{'class_weight':'auto'}), ('randC','default')]
    expSettings1 = { 'nExp':10, 'typeEval': ("kFoldCrossVal",10)}
    expSettings2 = { 'nExp':10, 'typeEval': ("leaveOneID",-1)}
    expSettings = [expSettings1, expSettings2]
    
    for AvgLen in AvgLens:
        for FrameLen in FrameLens:
            
            avgLenFrameLen = str(int(AvgLen*10)) + '_' + str(int(FrameLen*44100))
            arffFile = arffDir + '/' + 'MFCC_CENT_FLAT_' + avgLenFrameLen+'.arff'
            mappFile = arffDir + '/' + 'MFCC_CENT_FLAT_' + avgLenFrameLen+'.mappFileFeat'
            for expSetting in expSettings:
                mlObj = mlw.advanceExperimenter(arffFile = arffFile, mappFile = mappFile)
                mlObj.runCompoundExperiment(featureSets, classifierSets, expSetting, arffDir+'/'+avgLenFrameLen+'_'+expSetting['typeEval'][0], avgLenFrameLen)

            
def ComputeStatisticalSignificane(outputfile, quantity = 'overall', statistical_val = 0.05):

    if quantity == 'boundary':
        column_ind = 2
    elif quantity == 'overall':
        column_ind = 13

    exp_cases = ['PLS_L_TREE', 'PLS_L_KNN','PLS_L_NB','PLS_L_LOGREG','PLS_C_SVM', 'PLS_C_TREE', 'PLS_C_KNN','PLS_C_NB','PLS_C_LOGREG','PLS_C_SVM', 'PLS_LC_TREE', 'PLS_LC_KNN','PLS_LC_NB','PLS_LC_LOGREG','PLS_LC_SVM', 'OWN_L_TREE', 'OWN_L_KNN','OWN_L_NB','OWN_L_LOGREG','OWN_C_SVM', 'OWN_C_TREE', 'OWN_C_KNN','OWN_C_NB','OWN_C_LOGREG','OWN_C_SVM', 'OWN_LC_TREE', 'OWN_LC_KNN','OWN_LC_NB','OWN_LC_LOGREG','OWN_LC_SVM', 'OWN_L_DTW', 'OWN_C_DTW', 'OWN_LC_DTW', 'PLS_L_DTW', 'PLS_C_DTW', 'PLS_LC_DTW', 'BESTRANDOM']
    exp_files = ['90', '91', '92', '93', '94', '126', '127', '128', '129', '130', '138', '139', '140', '141', '142', '18', '19', '20', '21', '22', '54', '55', '56', '57', '58', '66', '67', '68', '69', '70', '1000', '1001','1002','1003','1004','1005', '2000']

    N_cases = len(exp_cases)
    file_prefix = 'experimentResults_WITH_ARTIST_RAG_FILTERING_PAIRWISE_EVAL100_STATISTICALTESTING/ExpData'

    array_combinations = []
    array_combinations_files = []
    for i in range(len(exp_cases)):
        for j in range(i+1, len(exp_cases)):
            array_combinations.append((i,j))
            array_combinations_files.append((file_prefix+exp_files[i]+'.json',file_prefix+exp_files[j]+'.json'))

    P_val_Array = []
    for i, comb in enumerate(array_combinations_files):
        P_val_Array.append(ComputePValueMannWhitney(comb[0],comb[1], column_ind))

    #holm bonferroni
    sort_ind = np.argsort(P_val_Array)
    P_val_Array_sort = np.sort(P_val_Array)
    N = len(P_val_Array)
    condition = True
    index = 0
    while condition:
        if P_val_Array_sort[index]< (statistical_val)/(N-index):
            condition = True
            index+=1
        else:
            condition = False

    print "statistical results are till index %d "%index

    #lets print some results now
    output_MTX = np.zeros([N_cases, N_cases]).astype(np.int16)

    for ind in sort_ind[:index]:
        output_MTX[array_combinations[ind][1], array_combinations[ind][0]]=1

    np.savetxt(outputfile, output_MTX, delimiter='\t', fmt='%d')
    return True

def ComputePValueMannWhitney(file1, file2):

    data1 = np.array(json.load(open(file1)))
    data2 = np.array(json.load(open(file2)))


    U, p_val = mannwhitneyu(np.reshape(data1,data1.size),np.reshape(data2,data2.size))

    return p_val         
    
if __name__=="__main__":
    
    output_dir = sys.argv[1]
    performCompoundTrainTesting(output_dir)
    
    """
    percFolder = sys.argv[1]
    nonPercFolder = sys.argv[2]
    output_dir = sys.argv[3]
    print "Percussion directory %s"% percFolder
    print "Non percussion  directory %s"% nonPercFolder
    print "Output directory %s"% output_dir
    
    generateARFF4DiffConfigs(percFolder, nonPercFolder, output_dir)
    """