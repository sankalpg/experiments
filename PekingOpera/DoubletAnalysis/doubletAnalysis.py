import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/similarityMeasures/dtw/'))

def find_nearest_element_ind(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    

def readDoubletAnnotations(filename):
    
    lines = open(filename).readlines()
    
    doublets= []
    lastDoubletInd = -1
    
    for line in lines:
        
        lineSplit = line.split(',')
        label = str(lineSplit[1])
        start = float(lineSplit[2])
        end = float(lineSplit[3])
        
        if label== 'U11':
            lastDoubletInd+=1
            doublets.append({})
        
        doublets[lastDoubletInd][label] = (start, end)
        
    return doublets

def computeIntraDoubletSimilarities(baseName, annotExt, pitchExt):
    
    #reading pitch file
    timePitch = np.loadtxt(baseName + pitchExt, delimiter = ',')
    
    #reading annotations 
    doublets = readDoubletAnnotations(baseName + annotExt)
    
    for ii, doublet in enumerate(doublets):
        
        parts = doublet.keys()
        
        #For this analysis if we dont have 6 sub divisions of a double exit!
        if len(parts)!=6:
            continue
        
        for mm in range(0, len(parts)):
            
            for nn in range(mm+1, len(parts)):
                
                s1 = find_nearest_element_ind(timePitch[:,0], doublet[parts[mm]][0])
                e1 = find_nearest_element_ind(timePitch[:,0], doublet[parts[mm]][1])
                
                s2 = find_nearest_element_ind(timePitch[:,0], doublet[parts[nn]][0])
                e2 = find_nearest_element_ind(timePitch[:,0], doublet[parts[nn]][1])
                
                dist = 
            
        
        
    
    
    
        
    
    
    