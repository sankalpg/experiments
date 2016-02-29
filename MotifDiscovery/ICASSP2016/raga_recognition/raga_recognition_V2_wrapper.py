import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis'))
import ragaRecognition as RR


def run_raga_recognition_V2_initial(results_dir):
    
    #parameters
    out_dir = results_dir
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016/__dbInfo__/Carnatic40RagasICASSP2016.flist_local'
    n_fold = 12
    thresholdBin = range(8,16)
    pattDistExt = ['.pattDistance_2s']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf-idf']#['tf', 'tp', 'tf-idf']
    pre_processing = [2]#[-1, 2]
    norm_tfidf = [None]#[None, 'l1', 'l2']
    smooth_idf = [False]#[False, True]
    classifier = [('svm', "default"),
                  ('svm-lin', "default"),
                  ('sgd', "default"),
                  ('nb-multi', "default"),
                  ('nb-gauss', "default"),
                  ('nb-bern', "default"),
                  ('randForest', "default"),
                  ('logReg', "default")]
    n_expts = 10
    database = 'ICASSP2016_40RAGA_2S'
    
    fid = open(os.path.join(results_dir, 'results_summary.txt'),'w')
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
                                dit_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                result = RR.raga_recognition_V2(dit_name, 
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
                                                                n_fold = n_fold,
                                                                n_expts = n_expts,
                                                                var1 = True,
                                                                var2 = True,
                                                                myDatabase = database,
                                                                myUser = 'sankalp')
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(ii), str(result)))
                                fid.write('\n')
    fid.close()
    
    
def run_raga_recognition_V2_classifiers(results_dir, class_type = 'nb-multi'):
    
    #parameters
    out_dir = results_dir
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016/__dbInfo__/Carnatic40RagasICASSP2016.flist_local'
    n_fold = 12
    thresholdBin = [9]
    pattDistExt = ['.pattDistance_2s']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf-idf']#['tf', 'tp', 'tf-idf']
    pre_processing = [2]#[-1, 2]
    norm_tfidf = [None]#[None, 'l1', 'l2']
    smooth_idf = [False]#[False, True]
    
    if class_type == 'nb-multi':
        classifier = [('nb-multi', "default"),
                      ('nb-multi', {"alpha":0.1}),
                      ('nb-multi', {"alpha":0.25}),
                      ('nb-multi', {"alpha":0.5}),
                      ('nb-multi', {"alpha":0.75}),
                      ('nb-multi', {"alpha":1.0}),]
    elif class_type == 'sgd':
        classifier = [('sgd', "default"),
                      ('sgd', {"loss":"hinge", "penalty":'l2', "alpha":.001, "n_iter": 5, "random_state":42})]
    elif class_type == 'randForest':
        classifier = [('randForest', "default"),
                      ('randForest', {"min_samples_leaf":1}),
                      ('randForest', {"min_samples_leaf":2}),
                      ('randForest', {"min_samples_leaf":3}),
                      ('randForest', {"min_samples_leaf":4}),
                      ('randForest', {"min_samples_leaf":5}),
                      ('randForest', {"min_samples_leaf":2, "n_estimators":51}),
                      ('randForest', {"min_samples_leaf":2, "n_estimators":101})]
    elif class_type == 'svm-lin':
        classifier = [('svm-lin', "default"),
                      ('svm-lin', {"C":0.0001}),
                      ('svm-lin', {"C":0.001}),
                      ('svm-lin', {"C":0.01}),
                      ('svm-lin', {"C":0.1}),
                      ('svm-lin', {"C":1.0}),
                      ('svm-lin', {"C":10}),
                      ('svm-lin', {"C":100}),
                      ('svm-lin', {"C":1000})]
    elif class_type == 'logReg':
        classifier = [('logReg', "default"),
                      ('logReg', {"C":0.0001}),
                      ('logReg', {"C":0.001}),
                      ('logReg', {"C":0.01}),
                      ('logReg', {"C":0.1}),
                      ('logReg', {"C":1.0}),
                      ('logReg', {"C":10}),
                      ('logReg', {"C":100}),
                      ('logReg', {"C":1000})]
            

    n_expts = 10
    database = 'ICASSP2016_40RAGA_2S'
    
    fid = open(os.path.join(results_dir, 'results_summary.txt'),'w')
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
                                dit_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                result = RR.raga_recognition_V2(dit_name, 
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
                                                                n_fold = n_fold,
                                                                n_expts = n_expts,
                                                                var1 = True,
                                                                var2 = True,
                                                                myDatabase = database,
                                                                myUser = 'sankalp')
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(ii), str(result)))
                                fid.write('\n')
    fid.close()    

    
def run_raga_recognition_V2_features_classifiers(results_dir):
    
    #parameters
    out_dir = results_dir
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic10RagasICASSP2016/__dbInfo__/Carnatic10RagasICASSP2016.flist_local'
    n_fold = 14
    thresholdBin = [11]
    pattDistExt = ['.pattDistance_2s']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf', 'tp', 'tf-idf']
    pre_processing = [2]#[-1, 2]
    norm_tfidf = [None]#[None, 'l1', 'l2']
    smooth_idf = [False]#[False, True]
    classifier = [('nb-multi', "default"),
                  ('nb-gauss', "default"),
                  ('nb-bern', "default"),
                  ('svm', "default"),
                  ('svm-lin', "default"),
                  ('sgd', "default"),
                  ('randForest', "default"),
                  ('logReg', "default")]
    n_expts = 10
    database = 'ICASSP2016_10RAGA_2S'
    
    fid = open(os.path.join(results_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*8%("Threshold", "pattExt", "Feature", "pre-proc", "norm_tfidf", "smooth", "classifier", "accuracy"))
    fid.write('\n')
    cnt = 1
    for t in thresholdBin:
        for ext in pattDistExt:
            for p in pre_processing:
                for n in norm_tfidf:
                    for s in smooth_idf:
                        for ii, c in enumerate(classifier):
                            for f in feature_type:
                                dit_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                result = RR.raga_recognition_V2(dit_name, 
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
                                                                n_fold = n_fold,
                                                                n_expts = n_expts,
                                                                var1 = False,
                                                                var2 = True,
                                                                myDatabase = database,
                                                                myUser = 'sankalp')
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(c), str(result)))
                                fid.write('\n')
    fid.close()



def run_raga_recognition_V2_thresholds(results_dir):

    #parameters
    out_dir = results_dir
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016/__dbInfo__/Carnatic40RagasICASSP2016.flist_local'
    n_fold = 12
    thresholdBin = range(6,16)
    pattDistExt = ['.pattDistance_2s']
    network_wght_type = -1
    force_build_network=0
    feature_type = ['tf-idf']#['tf', 'tp', 'tf-idf']
    pre_processing = [2]#[-1, 2]
    norm_tfidf = [None]#[None, 'l1', 'l2']
    smooth_idf = [False]#[False, True]
    classifier = [('nb-multi', "default")]
    n_expts = 10
    database = 'ICASSP2016_40RAGA_2S'

    fid = open(os.path.join(results_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*8%("Threshold", "pattExt", "Feature", "pre-proc", "norm_tfidf", "smooth", "classifier", "accuracy"))
    fid.write('\n')
    cnt = 1
    for t in thresholdBin:
        for ext in pattDistExt:
            for p in pre_processing:
                for n in norm_tfidf:
                    for s in smooth_idf:
                        for ii, c in enumerate(classifier):
                            for f in feature_type:
                                dit_name = os.path.join(out_dir, "config_%s"%str(cnt))
                                result = RR.raga_recognition_V2(dit_name,
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
                                                                n_fold = n_fold,
                                                                n_expts = n_expts,
                                                                var1 = False,
                                                                var2 = True,
                                                                myDatabase = database,
                                                                myUser = 'sankalp')
                                cnt+=1
                                fid.write("%s\t"*8%(str(t), ext, f, str(p), str(n), str(s), str(c), str(result)))
                                fid.write('\n')
    fid.close()
