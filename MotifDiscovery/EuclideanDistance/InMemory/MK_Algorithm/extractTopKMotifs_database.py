import numpy as np
import os,sys
import mmap, time, yaml


def mapcount(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    f.close()
    return lines

def getFileNameTime(index, timeStamps,index2fileMap):
    
    for row in index2fileMap:
        if index>row[0] and index<row[1]:
            return row[2], timeStamps[index]
        

def extractTopKMotifs(exeFile, dataset_dir, duration, blacklistDuration, hop, refPoints = 10, numMotif=1, motifDumpFile = 'Motifs', stdDumpFile='STDOUT', timeseriesFile = 'AggPitch.txt', timeStampFile = 'AggTime.txt', fileInfoFILE = 'fileInfo.yaml', blacklistFile = 'blacklist.txt'):
    
    lengthTimeSeries = mapcount(dataset_dir+'/'+timeseriesFile)
    
    motifSamples = int(np.round(duration/hop))
        
    fsuffix='_'+str(duration)+'sec.txt'
    fsuffix2='_'+str(duration)+'sec.yaml'
    
    bsf = -1
    
    
    init_time = time.time()
    
    
    timeStamps = np.loadtxt(dataset_dir+timeStampFile)
    stream = file(dataset_dir+fileInfoFILE,'r')
    fileinfo = yaml.load(stream)
    index2fileMap = []
    for key in fileinfo.keys():
        index2fileMap.append([fileinfo[key][0], fileinfo[key][1], key])
   
    motif_count=1
    
    #making file empty
    fid_blacklist = open(dataset_dir+blacklistFile,'w')
    fid_blacklist.close()
    
    
    for ii in range(0,numMotif):
        blacklist_ind = []
        fid_blacklist = open(dataset_dir+blacklistFile,'ab')
        print "processing for motif %d out of %d\n"%(ii+1,numMotif)
        print "time elapsed %f\n"%(time.time()-init_time)
        bsf=0#no need to use this because anyway we are using blacklisting
        cmd = "%s %s %d %d %d %f %s %s 1 >>%s"%(exeFile, dataset_dir+timeseriesFile, lengthTimeSeries, motifSamples, refPoints, bsf, dataset_dir+motifDumpFile+fsuffix,dataset_dir+blacklistFile, dataset_dir+stdDumpFile+fsuffix)
        proc_str = time.time()
        os.system(cmd)
        proc_end = time.time()
        dump = np.loadtxt(dataset_dir+motifDumpFile+fsuffix)
        loc1 = int(dump[0])
        loc2 = int(dump[1])
        dist = float(dump[2])
        
        filename1, timeLoc1 = getFileNameTime(loc1, timeStamps, index2fileMap)
        filename2, timeLoc2 = getFileNameTime(loc2, timeStamps, index2fileMap)
        
        if ii==0:
            motifs={}
        else:
            stream = file(dataset_dir+motifDumpFile+fsuffix2,'r')
            motifs = yaml.load(stream)
            stream.close()
        
        motifs[motif_count] = {'file':filename1, 'time':float(timeLoc1), 'pairID':motif_count+1, 'dist':dist, 'indDB':loc1,'timeProc':proc_end-proc_str}
        motif_count+=1
        motifs[motif_count] = {'file':filename2, 'time':float(timeLoc2), 'pairID':motif_count-1, 'dist':dist, 'indDB':loc2,'timeProc':proc_end-proc_str}
        motif_count+=1
        
        bsf = dist
               
        stream = file(dataset_dir+motifDumpFile+fsuffix2,'w')
        yaml.dump(motifs, stream)
        stream.close()
        
        #computing blacklist indexes
        offset=0
        print loc1
        while loc1+offset < timeStamps.shape[0]:
            if not abs(timeStamps[loc1]-timeStamps[loc1+offset]) < blacklistDuration:
                break
            blacklist_ind.append(loc1+offset)
            offset+=1
            
        offset=-1
        while loc1+offset > 0:
            if not abs(timeStamps[loc1]-timeStamps[loc1+offset]) < blacklistDuration:
                break
            blacklist_ind.append(loc1+offset)
            offset-=1
    
        offset=0
        print loc2
        while loc1+offset < timeStamps.shape[0]:
            if not abs(timeStamps[loc2]-timeStamps[loc2+offset]) < blacklistDuration:
                break
            blacklist_ind.append(loc2+offset)
            offset+=1
            
        offset=-1
        while loc1+offset > 0:
            if not abs(timeStamps[loc2]-timeStamps[loc2+offset]) < blacklistDuration:
                break
            blacklist_ind.append(loc2+offset)
            offset-=1    
        
        for kk in blacklist_ind:
            fid_blacklist.write("%s\n"%str(kk))
            
        fid_blacklist.close()
    
if __name__=="__main__":
    
    exeFile = sys.argv[1]
    dataset_dir = sys.argv[2]
    duration = float(sys.argv[3])
    blacklistDuration = float(sys.argv[4])
    hop = float(sys.argv[5])
    refPoints = int(sys.argv[6])
    numMotif = int(sys.argv[7])
    
    extractTopKMotifs(exeFile, dataset_dir, duration, blacklistDuration, hop, refPoints = refPoints, numMotif=numMotif)