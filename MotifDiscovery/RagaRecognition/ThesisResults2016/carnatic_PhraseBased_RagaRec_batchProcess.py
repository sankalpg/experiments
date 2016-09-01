import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis'))
import ragaRecognition as RR


def run_raga_recognition_V2_THESIS_RESULT_PhraseSection_GRID_CARNATIC():
    
	#Best in ISMIR2016:    8	.pattDistance_2s_config3	tf-idf	2	None	False	1	0.68125

    #parameters
    out_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/PhraseBased/Carnatic/GridSearch'
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016/__dbInfo__/Carnatic40RagasICASSP2016.flist_local'
    scratch_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/PhraseBased/Carnatic/scratch'
    thresholdBin = [7,8,9]
    pattDistExt = ['.pattDistance_2s_config3']
    network_wght_type = -1
    force_build_network=1
    feature_type = ['tf', 'tp', 'tf-idf']
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
    database = {'.pattDistance_2s_config0':'Raga_Rec_Carnatic_40Raga_Config0',
                '.pattDistance_2s_config1':'Raga_Rec_Carnatic_40Raga_Config1',
                '.pattDistance_2s_config2':'Raga_Rec_Carnatic_40Raga_Config2',
                '.pattDistance_2s_config3':'Raga_Rec_Carnatic_40Raga_Config3'}

    
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
                                # ###TEMP: remove this step, it was done because the process was stopped in the middle
                                # if os.path.isdir(dir_name):
                                #     results_data = pickle.load(open(os.path.join(dir_name, 'experiment_results.pkl'),'r'))
                                #     result = results_data['var2']['accuracy']
                                # else:
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




