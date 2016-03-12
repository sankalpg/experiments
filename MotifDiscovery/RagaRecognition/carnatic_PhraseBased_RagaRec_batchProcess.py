import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/networkAnalysis'))
import ragaRecognition as RR


def run_raga_recognition_V2_gridSearch():
    
    #parameters
    out_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/results/hindustani/gridSearch'
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/__dbInfo__/Hindustani30Ragas.flist_local'
    scratch_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/scratch_folder'
    thresholdBin = range(5,9)
    pattDistExt = ['.pattDistance_2s_config0', '.pattDistance_2s_config1', '.pattDistance_2s_config2', '.pattDistance_2s_config3']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf-idf']
    pre_processing = [2]
    norm_tfidf = [None]
    smooth_idf = [False]
    classifier = [('svm', "default"),
                  ('svm-lin', "default"),
                  ('sgd', "default"),
                  ('nb-multi', "default"),
                  ('nb-gauss', "default"),
                  ('nb-bern', "default"),
                  ('randForest', "default"),
                  ('logReg', "default")]
    n_expts = 1
    database = {'.pattDistance_2s_config0':'Raga_Rec_Hindustani_30Raga_Config0',
                '.pattDistance_2s_config1':'Raga_Rec_Hindustani_30Raga_Config1',
                '.pattDistance_2s_config2':'Raga_Rec_Hindustani_30Raga_Config2',
                '.pattDistance_2s_config3':'Raga_Rec_Hindustani_30Raga_Config3'}

    
    fid = open(os.path.join(out_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*8%("Threshold", "pattExt", "Feature", "pre-proc", "norm_tfidf", "smooth", "classifier", "accuracy"))
    fid.write('\n')
    cnt = 1
    for t in thresholdBin:
        for ext in pattDistExt:
            for f in feature_type:
                for p in pre_processing:
                    for n in norm_tfidf:
                        for s in smooth_idf:
                            for ii, c in enumerate(classifier):
                                dir_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                ###TEMP: remove this step, it was done because the process was stopped in the middle
                                if os.path.isdir(dir_name):
                                    results_data = pickle.load(open(os.path.join(dir_name, 'experiment_results.pkl'),'r'))
                                    result = results_data['var2']['accuracy']
                                else:
                                    result = RR.raga_recognition_V2(dir_name, 
                                                                    scratch_dir,
                                                                    fileListFile,
                                                                    t,
                                                                    ext,
                                                                    network_wght_type = network_wght_type,
                                                                    force_build_network=force_build_network,
                                                                    feature_type = f,
                                                                    pre_processing = p,
                                                                    norm_tfidf = n,
                                                                    smooth_idf = s,
                                                                    classifier = c,
                                                                    n_expts = n_expts,
                                                                    var1 = False,
                                                                    var2 = True,
                                                                    myDatabase = database[ext],
                                                                    myUser = 'sankalp',
                                                                    type_eval = ("LeaveOneOut", -1),
                                                                    balance_classes = 1)
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(ii), str(result)))
                                fid.write('\n')
    fid.close()





def run_raga_recognition_V2_classifiers():
    
    #parameters
    out_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/results/hindustani/classifiers'
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/__dbInfo__/Hindustani30Ragas.flist_local'
    scratch_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/scratch_folder'
    thresholdBin = [5]
    pattDistExt = ['.pattDistance_2s_config2']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf-idf']
    pre_processing = [2]
    norm_tfidf = [None]
    smooth_idf = [False]
    classifier = [    ('nb-multi', "default"),
                      ('nb-multi', {"alpha":0.1}),
                      ('nb-multi', {"alpha":0.25}),
                      ('nb-multi', {"alpha":0.5}),
                      ('nb-multi', {"alpha":0.75}),
                      ('nb-multi', {"alpha":1.0}),
                      ('sgd', "default"),
                      ('sgd', {"loss":"hinge", "penalty":'l2', "alpha":.001, "n_iter": 5, "random_state":42}),
                      ('randForest', "default"),
                      ('randForest', {"min_samples_leaf":1}),
                      ('randForest', {"min_samples_leaf":2}),
                      ('randForest', {"min_samples_leaf":3}),
                      ('randForest', {"min_samples_leaf":4}),
                      ('randForest', {"min_samples_leaf":5}),
                      ('randForest', {"min_samples_leaf":2, "n_estimators":51}),
                      ('randForest', {"min_samples_leaf":2, "n_estimators":101}),
                      ('svm-lin', "default"),
                      ('svm-lin', {"C":0.0001}),
                      ('svm-lin', {"C":0.001}),
                      ('svm-lin', {"C":0.01}),
                      ('svm-lin', {"C":0.1}),
                      ('svm-lin', {"C":1.0}),
                      ('svm-lin', {"C":10}),
                      ('svm-lin', {"C":100}),
                      ('svm-lin', {"C":1000}),
                      ('logReg', "default"),
                      ('logReg', {"C":0.0001}),
                      ('logReg', {"C":0.001}),
                      ('logReg', {"C":0.01}),
                      ('logReg', {"C":0.1}),
                      ('logReg', {"C":1.0}),
                      ('logReg', {"C":10}),
                      ('logReg', {"C":100}),
                      ('logReg', {"C":1000})]

    n_expts = 1
    database = {'.pattDistance_2s_config0':'Raga_Rec_Hindustani_30Raga_Config0',
                '.pattDistance_2s_config1':'Raga_Rec_Hindustani_30Raga_Config1',
                '.pattDistance_2s_config2':'Raga_Rec_Hindustani_30Raga_Config2',
                '.pattDistance_2s_config3':'Raga_Rec_Hindustani_30Raga_Config3'}

    
    fid = open(os.path.join(out_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*8%("Threshold", "pattExt", "Feature", "pre-proc", "norm_tfidf", "smooth", "classifier", "accuracy"))
    fid.write('\n')
    cnt = 1
    for t in thresholdBin:
        for ext in pattDistExt:
            for f in feature_type:
                for p in pre_processing:
                    for n in norm_tfidf:
                        for s in smooth_idf:
                            for ii, c in enumerate(classifier):
                                dir_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                ###TEMP: remove this step, it was done because the process was stopped in the middle
                                if os.path.isdir(dir_name):
                                    results_data = pickle.load(open(os.path.join(dir_name, 'experiment_results.pkl'),'r'))
                                    result = results_data['var2']['accuracy']
                                else:
                                    result = RR.raga_recognition_V2(dir_name, 
                                                                    scratch_dir,
                                                                    fileListFile,
                                                                    t,
                                                                    ext,
                                                                    network_wght_type = network_wght_type,
                                                                    force_build_network=force_build_network,
                                                                    feature_type = f,
                                                                    pre_processing = p,
                                                                    norm_tfidf = n,
                                                                    smooth_idf = s,
                                                                    classifier = c,
                                                                    n_expts = n_expts,
                                                                    var1 = False,
                                                                    var2 = True,
                                                                    myDatabase = database[ext],
                                                                    myUser = 'sankalp',
                                                                    type_eval = ("LeaveOneOut", -1),
                                                                    balance_classes = 1)
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(ii), str(result)))
                                fid.write('\n')
    fid.close()