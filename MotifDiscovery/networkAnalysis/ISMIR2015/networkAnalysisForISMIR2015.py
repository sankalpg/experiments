import os,sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis/'))

import constructNetwork as cons

def batchGenerateAllNetworkVariants(out_dir, fileListFile, pattDistExt, wghts = [-1, 0, 1], d_thslds = [30]):
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

    for w in wghts:
        for t in d_thslds:
            outfile = "Weight_%d___DThsld_%d___NX.edges"%(w, t)
            cons.constructNetwork_Weighted_NetworkX(fileListFile, os.path.join(out_dir, outfile), t, pattDistExt, w)
    

