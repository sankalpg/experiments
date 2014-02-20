import sys,os
import numpy as np
import matplotlib.pyplot as plt


def PlotPlayMotifsWithinFile(motifFile):
    
    audioExt = '.mp3'
    pitchExt = '.pitch'
    
    motifs = np.loadtxt(motifFile)
    
    fname, ext = os.path.splitext(motifFile)
    audioFile = fname + audioExt
    pitchFile = fname + pitchExt
    
    #load pitch data
    print("Loading pitch file, it might take a while, please be patient!!!")
    pitchData = np.loadtxt(pitchFile)
    hop = pitchData[1,0]-pitchData[0,0]
    
    for ii, motif in enumerate(motifs):
        usrInp = raw_input("Do you want to play and display (Y) or exit (N)")
        
        if usrInp.lower() == "y" or usrInp.lower() == "yes":
            
            str1 = motif[0]
            end1 = motif[1]
            str2 = motif[2]
            end2 = motif[3]
            dist = motif[4]
            
            print("playing and displaying motif pair %d/%d which has a distance of %f"%(ii+1, motifs.shape[0], dist))
            start_ind1 = np.argmin(abs(pitchData[:,0]-str1))
            end_ind1 = np.argmin(abs(pitchData[:,0]-end1))
            start_ind2 = np.argmin(abs(pitchData[:,0]-str2))
            end_ind2 = np.argmin(abs(pitchData[:,0]-end2))
            
            cmd1 = "play '%s' trim %f %f"%(audioFile, str1, end1-str1)
            os.system(cmd1)
            cmd2 = "play '%s' trim %f %f"%(audioFile, str2, end2-str2)
            os.system(cmd2)
            
            plt.plot(pitchData[start_ind1:end_ind1,1])
            plt.hold(True)
            plt.plot(pitchData[start_ind2:end_ind2,1])
            plt.show()
        else:
            break
            
 
if __name__=="__main__":
    
    motifFile = sys.argv[1]
    
    PlotPlayMotifsWithinFile(motifFile)
    
    
