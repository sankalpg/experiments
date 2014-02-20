import sys,os
import numpy as np
import matplotlib.pyplot as plt
import yaml

def plotMotifs(pitchfile1, startTime1, duration1, pitchfile2, startTime2, duration2):
    
    print pitchfile1, startTime1,duration1
    
    
    pitchData1 = np.loadtxt(pitchfile1)
    hop1 = pitchData1[1,0]-pitchData1[0,0]
    numSamples1 = int(np.round(duration1/hop1))
    start_ind1 = np.argmin(abs(pitchData1[:,0]-startTime1))
    
    
    pitchData2 = np.loadtxt(pitchfile2)
    hop2 = pitchData2[1,0]-pitchData2[0,0]
    numSamples2 = int(np.round(duration2/hop2))
    start_ind2 = np.argmin(abs(pitchData2[:,0]-startTime2))
    
    
    plt.plot(pitchData1[start_ind1:start_ind1+numSamples1,1])
    plt.hold(True)
    plt.plot(pitchData2[start_ind2:start_ind2+numSamples2,1])
    plt.show()
    
    return 1

def playMotifs(audioFile1,startTime1, duration1, audioFile2, startTime2, duration2):
    
    cmd1 = "play %s trim %f %f"%(audioFile1, startTime1, duration1)
    os.system(cmd1)
    
    cmd2 = "play %s trim %f %f"%(audioFile2, startTime2, duration2)    
    os.system(cmd2)
    

if __name__=="__main__":
    
    motifFile = sys.argv[1]
    motifId1 = int(sys.argv[2])
    motifId2 = int(sys.argv[3])
    
    duration = float(sys.argv[4])

    stream = file(motifFile,'r')
    motifsInfoDict = yaml.load(stream)
    stream.close()
    
    motifInfo1 = motifsInfoDict[motifId1]
    motifInfo2 = motifsInfoDict[motifId2]
    
    pitchfile1 = motifInfo1['file']
    audiofile1 = pitchfile1.split('.')[0]+'.mp3'
    str1 = motifInfo1['time']
    
    pitchfile2 = motifInfo2['file']
    audiofile2 = pitchfile2.split('.')[0]+'.mp3'
    str2 = motifInfo2['time']
    
    
    playMotifs(audiofile1, str1, duration, audiofile2, str2, duration)
    plotMotifs(pitchfile1, str1, duration, pitchfile2, str2, duration)
    
    