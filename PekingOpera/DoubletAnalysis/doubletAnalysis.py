import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/similarityMeasures/dtw/'))
import matplotlib.pyplot as plt
import dtw


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
                
                q = timePitch[s1:e1,1]
                t = timePitch[s2:e2,1]
                
                q = np.delete(q,np.where(q<0))
                t = np.delete(t,np.where(t<0))
                
                q = 1200*np.log2(q/55)
                t = 1200*np.log2(t/55)
                
                minLength = min(q.size, t.size)
                print minLength
                
                dist, pathLen, path, cost = dtw.dtw1dSubLocalBand(q,t, {'Output':4, 'Ldistance':{'type':1}, 'Constraint':{'CVal':int(minLength*0.1)}})
                print (parts[mm],parts[nn]), dist/pathLen
                plt.figure()
                plt.imshow(cost)
                drawAlignment(q,t,path )
                
                
def computeInterSectionSimilarity(baseName1, sec1, dInd1, baseName2, sec2, dInd2, annotExt, pitchExt):
    
    #reading pitch file
    timePitch1 = np.loadtxt(baseName1 + pitchExt, delimiter = ',')
    
    #reading pitch file
    timePitch2 = np.loadtxt(baseName2 + pitchExt, delimiter = ',')
    
    #reading annotations 
    doublets1 = readDoubletAnnotations(baseName1 + annotExt)
    
    #reading annotations 
    doublets2 = readDoubletAnnotations(baseName2 + annotExt)
    
    s1 = find_nearest_element_ind(timePitch1[:,0], doublets1[dInd1][sec1][0])
    e1 = find_nearest_element_ind(timePitch1[:,0], doublets1[dInd1][sec1][1])

    s2 = find_nearest_element_ind(timePitch2[:,0], doublets2[dInd2][sec2][0])
    e2 = find_nearest_element_ind(timePitch2[:,0], doublets2[dInd2][sec2][1])
    
    q = timePitch1[s1:e1,1]
    t = timePitch2[s2:e2,1]

    q = np.delete(q,np.where(q<=0))
    t = np.delete(t,np.where(t<=0))

    q = 1200*np.log2(q/55)
    t = 1200*np.log2(t/55)

    minLength = min(q.size, t.size)
    
    dist, pathLen, path, cost = dtw.dtw1dSubLocalBand(q,t, {'Output':4, 'Ldistance':{'type':1}, 'Constraint':{'CVal':int(minLength*0.2)}})
    print dist/pathLen
    #plt.figure()
    #plt.imshow(cost)
    drawAlignment(q,t,path )
                
                
            
def drawAlignment(x , y , path):
    
    plt.plot(x, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    
    plt.figure()
    plt.plot(x, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    for ii, row in enumerate(path[0]):
        
        if ii%10==0:
            x1 = path[0][ii]
            y1 = x[path[0][ii]]
            
            x2 = path[1][ii]
            y2 = y[path[1][ii]]
            
            plt.plot([x1,x2], [y1,y2], color='r')
        
        
    plt.show()        
        
    
    
    
        
    
    
    