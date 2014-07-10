import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../Library_PythonNew/batchProcessing/'))

import batchProcessing as BP

def getComputationStats(root_dir, Ext = '.2s25MotifLogs_CONF1'):
  
    filenames = BP.GetFileNamesInDir(root_dir, Ext)
    dataFile = np.zeros(5)
    for filename in filenames:
        lines = open(filename,"r").readlines()
        
        lineInd_FL = 20
        lineInd_LB_EQ = 21
        lineInd_LB_EC = 22
        lineInd_DTW = 23
        lineInd_Updates = 24
        
        procLines = [lineInd_FL, lineInd_LB_EQ , lineInd_LB_EC , lineInd_DTW , lineInd_Updates]
        
        for ii, lineInd in enumerate(procLines):
            
            line = lines[lineInd].split(':')[-1]
            line = line.strip()
            dataFile[ii] += int(line)
            
    return dataFile
        
        
if __name__=="__main__":
    root_dir = sys.argv[1]
    
    
    #discovert stats
    disc = getComputationStats(root_dir, '.2s25MotifLogs_CONF1')
    
    #searching stats
    search = getComputationStats(root_dir, '.2s25SearchProcLog_CONF1')
    
    np.savetxt("DiscoveryStats.txt", disc)
    np.savetxt("SearchStats.txt", search)
    
    
        