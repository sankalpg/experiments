import numpy as np
import os, sys
import matplotlib.pyplot as plt
import copy
import pickle
from scipy.stats import wilcoxon
import json
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rc('font',family='Times New Roman')


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






def plotAccuracyVsSizeCarnatic():
    """
    This is not a generic function. It plots basically the boxplot of accuracy Vs size 
    """

    cmd_root = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/DatasetEffect/fileLists/carnatic'
    cmd_folders = {5: '5_classes', 10: '10_classes', 20:'20_classes', 30:'30_classes', 40:'40_classes'}
    file_out_cmd = 'plots/Accuracy_Vs_Size_n_raagas_cmd.pdf'

    result_ext = '.results'

    results_carnatic = {}
    box_inp = []
    #Performing for carnatic
    for n_set in cmd_folders.keys():
        dirname = os.path.join(cmd_root, cmd_folders[n_set])
        filenames = GetFileNamesInDir(dirname, result_ext)
        accuracy = []
        for filename in filenames:
            data = pickle.load(open(filename,'r'))
            accuracy.append(data['var1']['accuracy'])
        results_carnatic[n_set] = accuracy
    
    for n_set in np.sort(results_carnatic.keys()):
        box_inp.append(results_carnatic[n_set])
    
    #print 'CARNATIC', n_set, len(accuracy), np.mean(accuracy), np.std(accuracy)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(box_inp,1, whis = [5, 95], patch_artist=True, showfliers=True)
    for median in bp['medians']:
        median.set(color='r', linewidth=3)
    for box in bp['boxes']:
        box.set( color='b', linewidth=1)        
        box.set( facecolor = 'w' )
    for flier in bp['fliers']:
        flier.set(marker='o', color='g') 

    ax.set_ylim([0.8, 1.01])
    fsize = 16
    fsize2 = 14
    
    plt.xlabel(r'Number of r\={a}gas', fontsize = fsize)
    plt.ylabel(r'Accuracy (\%)', fontsize = fsize)
    #plt.title('Scores by group and gender')
    plt.xticks(1+np.arange(len(box_inp)), np.sort(results_carnatic.keys()), fontsize = fsize2)
    #plt.legend(loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which = 'major', labelsize=fsize2)

    fig.savefig(file_out_cmd, bbox_inches='tight')
    
    



def plotAccuracyVsSizeHindustani():
    """
    This is not a generic function. It plots basically the boxplot of accuracy Vs size 
    """

    hmd_root = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/DatasetEffect/fileLists/hindustani'
    hmd_folders = {5: '5_classes', 10: '10_classes', 15:'15_classes', 20:'20_classes', 25:'25_classes', 30:'30_classes'}
    file_out_hmd = 'plots/Accuracy_Vs_Size_n_raagas_hmd.pdf'

    result_ext = '.results'

    results_hindustani = {}
    box_inp = []
    #Performing for carnatic
    for n_set in hmd_folders.keys():
        dirname = os.path.join(hmd_root, hmd_folders[n_set])
        filenames = GetFileNamesInDir(dirname, result_ext)
        accuracy = []
        for filename in filenames:
            data = pickle.load(open(filename,'r'))
            accuracy.append(data['var1']['accuracy'])
        results_hindustani[n_set] = accuracy
        #print 'HINDUSTANI', n_set, len(accuracy), np.mean(accuracy), np.std(accuracy)
    
    for n_set in np.sort(results_hindustani.keys()):
        box_inp.append(results_hindustani[n_set])
    
    #print 'CARNATIC', n_set, len(accuracy), np.mean(accuracy), np.std(accuracy)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(box_inp,1, whis = [5, 95], patch_artist=True, showfliers=True)
    for median in bp['medians']:
        median.set(color='r', linewidth=3)
    for box in bp['boxes']:
        box.set( color='b', linewidth=1)        
        box.set( facecolor = 'w' )
    for flier in bp['fliers']:
        flier.set(marker='o', color='g') 

    ax.set_ylim([0.9, 1.01])
    fsize = 16
    fsize2 = 14
    
    plt.xlabel(r'Number of r\={a}gas', fontsize = fsize)
    plt.ylabel(r'Accuracy (\%)', fontsize = fsize)
    #plt.title('Scores by group and gender')
    plt.xticks(1+np.arange(len(box_inp)), np.sort(results_hindustani.keys()), fontsize = fsize2)
    #plt.legend(loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which = 'major', labelsize=fsize2)

    fig.savefig(file_out_hmd, bbox_inches='tight')
