import numpy as np
import os,sys
import mmap


def mapcount(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    f.close()
    return lines



def extractTopKMotifs(exeFile, dataset_dir, duration, hop, refPoints = 10, numMotif=1, motifDumpFile = 'Motifs', stdDumpFile='STDOUT', timeseriesFile = 'AggPitch.txt'):
    
    lengthTimeSeries = mapcount(dataset_dir+'/'+timeseriesFile)
    
    motifSamples = int(np.round(duration/hop))
    
    fsuffix='_'+str(duration)+'sec.txt'
    
    bsf = -1
    
    motifs=[]
    
    for ii in range(0,numMotif):
    
        cmd = "./%s %s %d %d %d %d %f %s 1 >>%s"%(exeFile, dataset_dir+timeseriesFile, lengthTimeSeries, motifSamples, motifSamples, refPoints, bsf, dataset_dir+motifDumpFile+fsuffix, dataset_dir+stdDumpFile+fsuffix)
    
        os.system(cmd)
        
        dump = np.loadtxt(dataset_dir+motifDumpFile+fsuffix)
        motifs.append(dump.tolist())
        
        bsf = dump[-1]
               
    np.savetxt(dataset_dir+motifDumpFile+fsuffix, motifs)
