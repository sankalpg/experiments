import os, sys
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.sandbox.stats.runs import mcnemar
import pickle
import json


def readMyResults1(filename):
    data = pickle.load(open(filename, 'r'))
    len_samples = len(data['var1']['ind2mbid'].keys())
    mbids = np.array([data['var1']['ind2mbid'][key] for key in range(len_samples)])
    ind_sort = np.argsort(mbids)
    return  np.array(data['var1']['pf_accuracy'])[ind_sort], mbids[ind_sort]

def readMyResults2(filename):
    data = pickle.load(open(filename, 'r'))
    mbids = np.array(data['var2']['mbids'])
    ind_sort = np.argsort(mbids)
    return np.array(data['var2']['pf_accuracy'][0])[ind_sort], mbids[ind_sort]

def readSertanResults(filename):
    data = json.load(open(filename, 'r'))
    mbids = np.array(data['recording_mbid'])
    ind_sort = np.argsort(mbids)
    return np.array(data['eval'])[ind_sort], mbids[ind_sort]

def performStatisticalSignificanceTestingISMIR2016(tradition, outputFile):
    
    
    if tradition == 'hindustani':
        config1 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/hindustani/M_F/experiment_results.pkl', 'M_F', readMyResults1)
        config2 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/hindustani/M_KL/experiment_results.pkl', 'M_KL', readMyResults1)
        config3 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/hindustani/M_B/experiment_results.pkl', 'M_B', readMyResults1)
        config4 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/hindustani/E_PCD/hindustani_30_classes_results.json', 'E_PCD', readSertanResults)
        config5 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/hindustani/E_VSM/experiment_results.pkl', 'E_VSM', readMyResults2)
        configs = [config1, config2, config3, config4, config5]
    else:
        config1 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/carnatic/M_F/experiment_results.pkl', 'M_F', readMyResults1)
        config2 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/carnatic/M_KL/experiment_results.pkl', 'M_KL', readMyResults1)
        config3 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/carnatic/M_B/experiment_results.pkl', 'M_B', readMyResults1)
        config4 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/carnatic/E_PCD/carnatic_40_classes_results.json', 'E_PCD', readSertanResults)
        config5 = ('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ISMIR_2016/ISMIR_2016_Table_Results/carnatic/E_VSM/experiment_results.pkl', 'E_VSM', readMyResults2)
        configs = [config1, config2, config3, config4, config5]



    per_file_acc = []
    for c in configs:
        accs, mbid = c[2](c[0])
        per_file_acc.append(accs)

    nResults = len(configs)
    pVals = []
    combinations = []
    for ii in range(nResults):
        for jj in range(ii+1, nResults):
            if np.sum(abs(per_file_acc[ii] - per_file_acc[jj]))==0:
                pVals.append(1)    
            else:
                pVals.append(mcnemar(per_file_acc[ii], per_file_acc[jj], exact = False)[1])
            combinations.append([configs[ii][1], configs[jj][1]])
    
    pVals = np.array(pVals)
    print pVals, combinations
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
        print "%s and %s with p value = %0.20f\n"%(elem[0], elem[1], pValsSorted[ii])
        fid.write("%s and %s with p value = %0.20f\n"%(elem[0], elem[1], pValsSorted[ii]))
    fid.close()
    return 1
    
    
