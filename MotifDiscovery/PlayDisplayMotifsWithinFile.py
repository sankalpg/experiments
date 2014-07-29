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
            
            """plt.plot(pitchData[start_ind1:end_ind1,1])
            plt.hold(True)
            plt.plot(pitchData[start_ind2:end_ind2,1])
            plt.show()
            """
            plotMotifPairs(pitchData[start_ind1:end_ind1,1],  pitchData[start_ind2:end_ind2,1])
            
        else:
            break

def plotMotifPairs(pattern1, pattern2, plotName=-1):
    
    
    colors = ['g', 'r']
#   linewidths = [3,0.1 ,0.1 , 3]

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    pLeg = []
    
    p, = plt.plot((196/44100.0)*np.arange(pattern1.size), pattern1, 'r', linewidth=2, markersize=4.5)
    pLeg.append(p)
    
    p, = plt.plot((196/44100.0)*np.arange(pattern2.size), pattern2, 'k', linewidth=2, markersize=4.5)
    pLeg.append(p)

    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel("time (s)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency (Hz)", fontsize = fsize, fontname=font, labelpad=fsize2)

    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName)

    return 1            
 
if __name__=="__main__":
    
    motifFile = sys.argv[1]
    
    PlotPlayMotifsWithinFile(motifFile)
    
    
