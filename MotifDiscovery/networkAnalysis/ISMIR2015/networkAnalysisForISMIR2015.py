import os,sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis/'))

import constructNetwork as cons
import networkProcessing as netPro

def batchGenerateAllNetworkVariants(out_dir, fileListFile, pattDistExt, wghts = [0], d_thslds = [30], confs =[-1, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9]):
    """
    This function generates different network variants from the patterns that we have obtained from
    the unsupervised analysis
    
    :out_dir: directory where to save the final network files (Pajek)
    :fileListFile: file which contains list of filenames which are to be used for the network generation
    :pattDistExt:   file extension for the file where we store pattern distances 
    :wghts:     list of weights for the variants
    :d_thslds:  list of bin values for distance thresholding for the variants

    outputfile naming convention Weight_XX_DThsld_YY_NX.edges
    """
    for conf in confs:
        conf_val = ((1-conf)*100.0)
        outputFolder = os.path.join(out_dir, "conf%0.2f"%conf_val)
        
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)
        
        for w in wghts:
            for t in d_thslds:
                networkFile = os.path.join(outputFolder, "Weight_%d___DThsld_%d___Conf_%0.2f___NX.edges"%(w, t, conf_val))
                cons.constructNetwork_Weighted_NetworkX(fileListFile, networkFile, t, pattDistExt, w, conf)
    
    
    
"""
def batchGenerateDisparityFilteredOutput(networkFile, outputDir, confs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        This function (batch) generates disparity filtered networks.
    The resultant network is stored in the output folder in a folder named confXX where XX is the confidence measure of the disparity filtering.
    
    
    
        netPro.filterAndSaveNetwork(networkFile, outfile, conf)
""" 