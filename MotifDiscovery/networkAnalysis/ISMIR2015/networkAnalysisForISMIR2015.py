import os,sys
import numpy as np
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing'))

import batchProcessing as BP
import constructNetwork as cons
import networkProcessing as netPro
import patternCharacterization as char

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
    
    
def batchProcessCommunityDetection(root_dir, normType = ['tonicNorm', 'pasaNorm'], tRange = range(6,24,2), confs = ['conf200.00', 'conf99.90', 'conf99.00', 'conf90.00', 'conf80.00', 'conf50.00']):
    """
    This function performs batch process of community detection function
    :normType:  'tonicNorm' or 'pasaNorm'
    """
    
    inpExt = '.edges'
    outExt = '.community'
    
    for ii, norm in enumerate(normType):
        for jj, t in enumerate(tRange):
            for kk, conf in enumerate(confs):
                print "processing %d, %d, %d of %d, %d, %d\n"%(ii+1, jj+1, kk+1,len(normType), len(tRange), len(confs))
                fname = os.path.join(root_dir, norm, 'weight0', conf, 'Weight_-1___DTshld_%d___PostFilt___C'%t)
                netPro.detectCommunitiesInNewtworkNX(fname+ inpExt, fname+outExt)
    
def batchProcessBetweennessCentrality(root_dir, normType = ['tonicNorm', 'pasaNorm'], tRange = range(6,24,2), confs = ['conf200.00', 'conf99.90', 'conf99.00', 'conf90.00', 'conf80.00', 'conf50.00']):
    """
    This function performs batch process of community detection function
    :normType:  'tonicNorm' or 'pasaNorm'
    """
    
    netExt = '.edges'
    outExt = '.bwcen'
    
    for ii, norm in enumerate(normType):
        for jj, t in enumerate(tRange):
            for kk, conf in enumerate(confs):
                print "processing %d, %d, %d of %d, %d, %d\n"%(ii+1, jj+1, kk+1,len(normType), len(tRange), len(confs))
                fname = os.path.join(root_dir, norm, 'weight0', conf, 'Weight_-1___DTshld_%d___PostFilt___C'%t)
                netFile = fname+ netExt
                print "Processing: %s\n"%netFile
                netPro.computeBetweenesCentrality(netFile, fname+outExt)    

def batchProcessCommunityRanking(root_dir, communityExt = '.community'):
    """
    This function batch process community ranking to sort out useful communities which can fetch good patterns.
    """
    
    filenames = BP.GetFileNamesInDir(root_dir,communityExt)
    
    for filename in filenames:
        char.rankCommunities(filename, filename+'Rank', myDatabase = 'ISMIR2015_10RAGA_TONICNORM')
    
    
    
def batchCompileCommunityRanking(root_dir, outputFile, communityRankExt = '.communityRank'):
    """
    This function process several community rank files and then just store the community rankings ffrom these files in a readable way.
    By looking at the rankings hopefully we can select one confuguration which results into a decent clustering performance.
    """
    
    filenames = BP.GetFileNamesInDir(root_dir,communityRankExt)
    rankMTX = np.ones((5000, len(filenames)))
    for ii, filename in enumerate(filenames):
        print ii, filename
        data = json.load(open(filename))
        ranks = [r['rank'] for r in data['comRank']]
        rankMTX[1:len(ranks)+1, ii] = ranks
        rankMTX[0,ii] = ii
    
    np.savetxt(outputFile, rankMTX)
    
def batchProcessPhraseDumping(audio_root, root_dir, normType = ['tonicNorm', 'pasaNorm'], tRange = range(6,24,2), confs = ['conf200.00', 'conf99.90', 'conf99.00', 'conf90.00', 'conf80.00', 'conf50.00']):
    """
    This function performs batch process of phrase dumping per configuration
    :normType:  'tonicNorm' or 'pasaNorm'
    """
    
    netExt = '.edges'
    outExt = '.bwcen'
    myDatabase = 'ISMIR2015_10RAGA_TONICNORM'
    comRankExt = '.communityRank'
    
    
    for ii, norm in enumerate(normType):
        for jj, t in enumerate(tRange):
            for kk, conf in enumerate(confs):
                print "processing %d, %d, %d of %d, %d, %d\n"%(ii+1, jj+1, kk+1,len(normType), len(tRange), len(confs))
                fname = os.path.join(root_dir, norm, 'weight0', conf, 'Weight_-1___DTshld_%d___PostFilt___C'%t)                
                char.extractRagaPhrases(audio_root, fname, 10, comRankExt =comRankExt, netExt= netExt, myDatabase= myDatabase)
            
            
            
            