import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/melodyProcessing'))
import phaseSpaceEmbedding as p


def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                names.append(path + "/" + f)
    return names

def convertJsontoTxt(root_dir, file_ext, outext):
    """
    This is a very specific function to convert the json files to text files in the appropriate format needed for our code
    """

    filenames  = GetFileNamesInDir(root_dir, file_ext)
    for filename in filenames:
        data = json.load(open(filename, 'r'))
        fname, ext = os.path.splitext(filename)
        fid = open(fname+outext, 'w')
        for d in data:
            fid.write("%s\t%s\t%s\n"%(d[2].replace('/audio/',''),d[0],d[1]))
        fid.close()

    return True

def run_raga_recognition_V1_PhaseSpace_CarnaticDatasetEffect():
    
    #parameters THE CONFIG WHICH ARE CORRESPONDING TO THE EDUCATED GUESS (WHICH WAS PRETTY CLOSE TO THE BEST)
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/DatasetEffect/fileLists/carnatic/'
    feature_dir = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016/audio/'
    delay = 300
    sigma = 2
    compression = 0.75
    normalize = 2
    dist_metric = 'KLD_symm'
    KNN= 1
    phase_ext = {   200:'.phasespace_200',
                    300:'.phasespace_300',
                    500:'.phasespace_500',
                    1000:'.phasespace_1000',
                    1500:'.phasespace_1500'}

    filenames  = GetFileNamesInDir(root_dir, '.txt')
    for filename in filenames:
        result = p.ragaRecognitionPhaseSpaceKNN_V1(filename,
                                                feature_dir, 
                                                filename,
                                                phase_ext = phase_ext[delay],
                                                smooth_gauss_sigma=sigma,
                                                compression = compression,
                                                normalize = normalize,
                                                dist_metric = dist_metric,
                                                KNN = KNN )


def run_raga_recognition_V1_PhaseSpace_HindustaniDatasetEffect():
    
    #parameters THE CONFIG WHICH ARE CORRESPONDING TO THE EDUCATED GUESS (WHICH WAS PRETTY CLOSE TO THE BEST)
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/DatasetEffect/fileLists/hindustani/'
    feature_dir = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas'
    delay = 300
    sigma = 2
    compression = 0.75
    normalize = 2
    dist_metric = 'KLD_symm'
    KNN= 1
    phase_ext = {   200:'.phasespace_200',
                    300:'.phasespace_300',
                    500:'.phasespace_500',
                    1000:'.phasespace_1000',
                    1500:'.phasespace_1500'}

    filenames  = GetFileNamesInDir(root_dir, '.txt')
    for filename in filenames:
        result = p.ragaRecognitionPhaseSpaceKNN_V1(filename,
                                                feature_dir, 
                                                filename,
                                                phase_ext = phase_ext[delay],
                                                smooth_gauss_sigma=sigma,
                                                compression = compression,
                                                normalize = normalize,
                                                dist_metric = dist_metric,
                                                KNN = KNN )        