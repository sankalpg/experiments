import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/melodyProcessing'))
import phaseSpaceEmbedding as p


def run_raga_recognition_V1_gridSearch():
    
    #parameters
    out_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/results/PhaseSpaceEmbedding/V1/hindustani/gridsearch'
    fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/__dbInfo__/Hindustani30Raga300_FILE_MBID_RAGA.txt'
    root_dir = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/pitch_tonic'
    delay = [200 , 300 , 500, 1000, 1500]
    smooth_gauss_sigma = [-1, 1, 2, 3]
    compression = [-1, 0.1, 0.25, 0.5, 0.75]
    normalize = [-1, 1, 2]
    dist_metric = ['Euclidean', 'KLD_symm', 'Bhattacharya']
    KNN= [1, 3, 5]
    phase_ext = {   200:'.phasespace_200',
                    300:'.phasespace_300',
                    500:'.phasespace_500',
                    1000:'.phasespace_1000',
                    1500:'.phasespace_1500'}



    fid = open(os.path.join(out_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*7%("Delay", "Smooth_Sigma", "Compression", "Norm", "Distance", "KNN", "accuracy"))
    fid.write('\n')
    cnt = 1
    for d in delay:
        for sigma in smooth_gauss_sigma:
            for c in compression:
                for n in normalize:
                    for dist in dist_metric:
                        for k in KNN:
                            dir_name = os.path.join(out_dir, "config_%s"%str(cnt))
                            result = p.ragaRecognitionPhaseSpaceKNN_V1(dir_name,
                                                            root_dir, 
                                                            fileListFile,
                                                            phase_ext = phase_ext[d],
                                                            smooth_gauss_sigma=sigma,
                                                            compression = c,
                                                            normalize = n,
                                                            dist_metric = dist,
                                                            KNN = k )
                            fid.write("%s\t"*7%(str(d), str(sigma), str(c), str(n), str(dist), str(k), str(result)))
                            fid.write('\n')
                            cnt+=1
    fid.close()

