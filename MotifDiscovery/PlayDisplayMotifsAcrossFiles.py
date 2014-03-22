import sys,os, copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../library_pythonnew/similarityMeasures/dtw/'))

import dtw
from scipy.interpolate import interp1d




def sortDistancesGenerateMapping(motifFile, mappFile, motifPairIndex, motifIndex_1_2):
    
    data = np.loadtxt(motifFile)
    mappData = open(mappFile,'r').readlines()
    
    dataMotif = data[:, (motifPairIndex-1)*10 + (motifIndex_1_2-1)*5: (motifPairIndex-1)*10 + motifIndex_1_2*5]
    fileArray = []
    mappArray = np.zeros(dataMotif.shape[0])
    for ii, line in enumerate(mappData):
        filename, start, end = line.split('\t')
        filename = filename.strip()
        start = int(start.strip())
        end = int(end.strip())
        mappArray[start-1:end]=int(ii)
        fileArray.append(filename)
    
    blackInd = np.where(dataMotif[:,4]==-1)[0]
    dataMotif = np.delete(dataMotif, [blackInd], axis=0)
    mappArray = np.delete(mappArray, blackInd).astype(np.int)
    dist = dataMotif[:,4]
    
    sortInd = np.argsort(dist, kind='mergesort')
    
    return sortInd, dataMotif, mappArray, fileArray
    
def exportSearchDataMapping(motifFile, mappFile, motifPairIndex, motifIndex_1_2):
    
    data = np.loadtxt(motifFile)
    mappData = open(mappFile,'r').readlines()
    
    dataMotif = data[:, (motifPairIndex-1)*12 + (motifIndex_1_2-1)*6: (motifPairIndex-1)*12 + motifIndex_1_2*6]
    fileArray = []
    mappArray = np.zeros(dataMotif.shape[0])
    for ii, line in enumerate(mappData):
        index, filename = line.split('\t')
        filename = filename.strip()
        fileArray.append(filename)
    
    mappArray = copy.deepcopy(dataMotif[:,5]).astype(np.int)
    
    sortInd = np.arange(dataMotif.shape[0])
    return sortInd, dataMotif, mappArray, fileArray



def PlotPlayMotifsAcrossFiles(motifFile, mappFile, motifPairIndex, formatNEWOLD):
    
    #sorting distances and generating mapping to files
    if (formatNEWOLD == 'formatNew'):
        sortInd, dataMotif, mappArray, fileArray = exportSearchDataMapping(motifFile, mappFile, motifPairIndex, 1)
    elif(formatNEWOLD == 'formatOld'):
        sortInd, dataMotif, mappArray, fileArray = sortDistancesGenerateMapping(motifFile, mappFile, motifPairIndex, 1)
    else:
        print "Please specify valid format"
        return -1
    
    
    audioExt = '.mp3'
    pitchExt = '.pitch'
    tonicExt = '.tonic'
    
    serverSuffix = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/compMusic/audio_3D'
    localSuffix = '/media/Data/Datasets/MotifDiscovery_Dataset/CompMusic/audio_3D'
    
    
    fnameSeed, ext = os.path.splitext(motifFile)
    audioFileSeed = fnameSeed + audioExt
    pitchFileSeed = fnameSeed + pitchExt
    tonicFileSeed = fnameSeed + tonicExt
    
    #load pitch data
    print("Loading pitch corresponding to the seed motif, it might take a while, please be patient!!!")
    pitchDataSeed = np.loadtxt(pitchFileSeed)
    tonicDataSeed = np.loadtxt(tonicFileSeed)
    
    
    fileSearched = fileArray[mappArray[sortInd[0]]].replace(serverSuffix, localSuffix)
    fnameSearch, extSeed = os.path.splitext(fileSearched)
    audioFileSearch = fnameSearch + audioExt
    pitchFileSearch = fnameSearch + pitchExt
    tonicFileSearch = fnameSearch + tonicExt
    print("Loading pitch corresponding to the searched motif, it might take a while, please be patient!!!")
    pitchDataSearch = np.loadtxt(pitchFileSearch)
    tonicDataSearch = np.loadtxt(tonicFileSearch)
    fileSearched_last = fileSearched;
            
    
    for ii, motif in enumerate(dataMotif[sortInd]):
        
        usrInp = raw_input("Do you want to play and display (Y) or exit (N)")
        
        if usrInp.lower() == "y" or usrInp.lower() == "yes":
            
            fileSearched = fileArray[mappArray[sortInd[ii]]].replace(serverSuffix, localSuffix)
            if not fileSearched == fileSearched_last:
                fnameSearch, extSeed = os.path.splitext(fileSearched)
                audioFileSearch = fnameSearch + audioExt
                pitchFileSearch = fnameSearch + pitchExt
                tonicFileSearch = fnameSearch + tonicExt
                print("Loading pitch corresponding to the searched motif again as the file changed this time, it might take a while, please be patient!!!")
                pitchDataSearch = np.loadtxt(pitchFileSearch)
                tonicDataSearch = np.loadtxt(tonicFileSearch)
                fileSearched_last = fileSearched;
            
            str1 = motif[0]
            end1 = motif[1]
            str2 = motif[2]
            end2 = motif[3]
            dist = motif[4]
            
            print("playing and displaying motif pair %d/%d which has a distance of %f"%(ii+1, dataMotif.shape[0], dist))
            start_ind1 = np.argmin(abs(pitchDataSeed[:,0]-str1))
            end_ind1 = np.argmin(abs(pitchDataSeed[:,0]-end1))
            end_ind11 = np.argmin(abs(pitchDataSeed[:,0]-(end1+5)))
            start_ind2 = np.argmin(abs(pitchDataSearch[:,0]-str2))
            end_ind2 = np.argmin(abs(pitchDataSearch[:,0]-end2))
            end_ind22 = np.argmin(abs(pitchDataSearch[:,0]-(end2+5)))
            
            cmd1 = "play '%s' trim %f %f"%(audioFileSeed, str1, end1-str1)
            os.system(cmd1)
            print "Searched audio file is '%s'"%(audioFileSearch)
            cmd2 = "play '%s' trim %f %f"%(audioFileSearch, str2, end2-str2)
            os.system(cmd2)
            
            
            #computeLeastDistanceInterp(pitchDataSeed[start_ind1:end_ind11,1], pitchDataSearch[start_ind2:end_ind22,1], downSample, DTWBand, numSamples, DTWBand)
            
            
            if (0):
                plt.plot(120*np.log2((pitchDataSeed[start_ind1:end_ind1,1]+1)/tonicDataSeed))
                plt.hold(True)
                plt.plot(120*np.log2((pitchDataSearch[start_ind2:end_ind2,1]+ 1)/tonicDataSearch))
                plt.show()
            else:
                plt.plot(pitchDataSeed[start_ind1:end_ind1,1])
                plt.hold(True)
                plt.plot(pitchDataSearch[start_ind2:end_ind2,1])
                plt.show()
                
            
        else:
            break

def drawAlignment(x , y , path):
    
    plt.plot(x+53, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    
    plt.figure()
    plt.plot(x+53, color='b')
    plt.hold(True)
    plt.plot(y,color='g')
    for ii, row in enumerate(path[0]):
        
        if ii%10==0:
            x1 = path[0][ii]
            y1 = x[path[0][ii]]+53
            
            x2 = path[1][ii]
            y2 = y[path[1][ii]]
            
            plt.plot([x1,x2], [y1,y2], color='r')
        
        
    plt.show()
    
    
def computeLeastDistanceInterp(pitch_seed, pitch_search, downSample, interpFactor, numSamples, DTWBand):
        
        
        indLow = (1-interpFactor)*np.arange(numSamples)
        indHigh = (1+interpFactor)*np.arange(numSamples)
        
        #Extracting the path and distance
        pitch_seed = pitch_seed[range(0,pitch_seed.shape[0],downSample)]
        ind_sil = np.where(pitch_seed<60)[0]
        pitch_seed = np.delete(pitch_seed, ind_sil)
        
        pitch_search = pitch_search[range(0,pitch_seed.shape[0],downSample)]
        ind_sil = np.where(pitch_search<60)[0]
        pitch_search = np.delete(pitch_search, ind_sil)
        
        seed = []
        interpFunc = interp1d(np.arange(pitch_seed.shape[0]), pitch_seed, kind='cubic')        
        seed.append(pitch_seed[0:numSamples])
        seed.append(interpFunc(indLow))
        seed.append(interpFunc(indHigh))
        
        search = []
        interpFunc = interp1d(np.arange(pitch_search.shape[0]), pitch_search, kind='cubic')        
        search.append(pitch_search[0:numSamples])
        search.append(interpFunc(indLow))
        search.append(interpFunc(indHigh))
        
        dist_max = 999999999
        for ii,seedElem in enumerate(seed):
            for jj,searchElem in enumerate(search):
                dist = dtw.dtw1dLocalBand(pitch_seed, pitch_search, {'Output':1, 'Ldistance':{'type':0}, 'Constraint':{'CVal':int(DTWBand*numSamples)}})
                if (dist < dist_max):
                    dist_max = dist
                    config = (ii,jj)
         
        dist, pathLen, path, cost = dtw.dtw1dLocalBand(seed[config[0]], search[config[1]], {'Output':4, 'Ldistance':{'type':0}, 'Constraint':{'CVal':int(DTWBand*numSamples)}})
        
        drawAlignment(seed[config[0]], search[config[1]], path)
        
        return 1
 
if __name__=="__main__":
    
    motifFile = sys.argv[1]
    mappFile = sys.argv[2]
    motifPairIndex = sys.argv[3]
    formatNEWOLD = sys.argv[4]
    
    PlotPlayMotifsAcrossFiles(motifFile, mappFile, int(motifPairIndex),formatNEWOLD)
    
    
