import sys,os
import numpy as np
import matplotlib.pyplot as plt

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
    


def PlotPlayMotifsAcrossFiles(motifFile, mappFile, motifPairIndex):
    
    #sorting distances and generating mapping to files
    sortInd, dataMotif, mappArray, fileArray = sortDistancesGenerateMapping(motifFile, mappFile, motifPairIndex, 1)
    
    
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
            start_ind2 = np.argmin(abs(pitchDataSearch[:,0]-str2))
            end_ind2 = np.argmin(abs(pitchDataSearch[:,0]-end2))
            
            cmd1 = "play '%s' trim %f %f"%(audioFileSeed, str1, end1-str1)
            os.system(cmd1)
            cmd2 = "play '%s' trim %f %f"%(audioFileSearch, str2, end2-str2)
            os.system(cmd2)
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
            
 
if __name__=="__main__":
    
    motifFile = sys.argv[1]
    mappFile = sys.argv[2]
    motifPairIndex = sys.argv[3]
    
    PlotPlayMotifsAcrossFiles(motifFile, mappFile, int(motifPairIndex))
    
    
