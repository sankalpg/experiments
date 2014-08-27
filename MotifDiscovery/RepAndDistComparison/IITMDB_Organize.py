import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP

def renameOrganizeMotifsPerRagaIITMDB(motifMapFile, inpAnnotFileExt, outAnnotFileExt, filterOutput, dumpFile):
    """
    When filterOutput is set to 1 only the motifs which are indicated in motifmapfile appear in the output
    """
    
    lines = open(motifMapFile, "r").readlines()
    
    dump = open(dumpFile, "w");
    
    for line in lines:
        splitLine = line.split("\t")
        root_dir = splitLine[0].strip()
        baseNumber = int(splitLine[1].strip())
        motifs = splitLine[2:]
        motifDB={}
        dump.write("%s\t%d\t"%(root_dir, baseNumber))
        
        for m in motifs:
            motifDB[m.strip()]=[]
        
        filenames = BP.GetFileNamesInDir(root_dir, inpAnnotFileExt)
        motifNumber = {}
        for filename in filenames:
            
            if filterOutput==1 and len(motifDB.keys())==0:
                continue
            
            fname, ext = os.path.splitext(filename)
            outfile = open(fname + outAnnotFileExt, "w");
            
            
            annotLines = open(filename, "r").readlines()
            
            try: 
                for annotLine in annotLines:
                    splitAnnot = annotLine.split(' ')
                    start = float(splitAnnot[0].strip())
                    dur = float(splitAnnot[1].strip())
                    label = str(splitAnnot[2].strip())
                    if  not label[0]=="m":
                        print "not M: " + filename
                    if not motifNumber.has_key(label):
                        motifNumber[label] = baseNumber
                        dump.write("(%s\t%d)\t"%(label, motifNumber[label]))
                        baseNumber+=1
                    if filterOutput==1:
                        if motifDB.has_key(label):    
                            outfile.write("%f\t%f\t%d\n"%(start, start + dur, motifNumber[label]))
                    else:
                        outfile.write("%f\t%f\t%d\n"%(start, start + dur, motifNumber[label]))
            except:
                print "No format: " + filename
                    
            outfile.close()
        dump.write("\n")
        
    dump.close()
        

def validateMotifSearchDB(root_dir, fileout):
    
    filenames = BP.GetFileNamesInDir(root_dir, '.wav')
    
    ExtensionsRef = ['.pitchEssentiaIntp', '.tonic', '.anot']
    
    fid = open(fileout, "w")
    
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        
        for ext in ExtensionsRef:
            filecheck = fname + ext
            if not os.path.isfile(filecheck):
                fid.write("%s\t%s\n"%(filename, ext))
                #break
            
    fid.close()
    
def generateFileList(root_dir, fileOut):
    
    filenames = BP.GetFileNamesInDir(root_dir, '.wav')
    
    fid = open(fileOut, "w")
    for f in filenames:
        filename, ext = os.path.splitext(f)
        fid.write("%s\n"%filename)
    
    fid.close()    

def generateDBStats(fileListFile, anotExt = '.anot'):
    
    lines = open(fileListFile,"r").readlines()
    motifDB={}
    for jj, line in enumerate(lines):
        filename = line.strip() + anotExt
        annotations = np.loadtxt(filename)
        if annotations.size ==0:
            continue
        if len(annotations.shape)==1:
            annotations = np.array([annotations])
        for ii in np.arange(annotations.shape[0]):
            line = annotations[ii,:]
            id = int(line[2])
            if not motifDB.has_key(id):
                motifDB[id]=[]
            motifDB[id].append((jj,(line[0], line[1])))
    
    temp=[]
    for key in motifDB.keys():
        temp.append([len(motifDB[key]), key])
    
    temp=np.array(temp)
    ind =  np.argsort(temp[:,0])
    temp = temp[ind,:]
    print temp

    return motifDB



def topMmotifNFilesPRaga(fileListFile, fileListOutput, anotExt = '.anot', M=20,N=10):
    
    lines = open(fileListFile,"r").readlines()
    
    output = open(fileListOutput, "w")
    
    
    motifDB = generateDBStats(fileListFile)
    temp=[]
    for key in motifDB.keys():
        temp.append([len(motifDB[key]), key])
    
    temp=np.array(temp)
    ind =  np.argsort(temp[:,0], )
    ind = ind[::-1]
    ind = ind[:min(len(ind),M)]
    motifNums = temp[ind,1]
    print motifNums
    motifBases = np.round(motifNums/1000)*1000
    
    d1 = {}
    for ii,motifNum in enumerate(motifNums):
        if not d1.has_key(motifBases[ii]):
            d1[motifBases[ii]]={}
        
        for elem in motifDB[motifNum]:
            if not d1[motifBases[ii]].has_key(elem[0]):
                d1[motifBases[ii]][elem[0]]=0
                
            d1[motifBases[ii]][elem[0]]+=1
            
    outFiles = {}
    for key in d1.keys():
        temp = []
        for e in d1[key].keys():
            temp.append([d1[key][e], e])
        temp=np.array(temp)
        ind =  np.argsort(temp[:,0], )
        ind = ind[::-1]
        ind = ind[:min(len(ind),N)]
        outFiles[key]= temp[ind,1]
        for ind in outFiles[key]:
            fname = lines[ind].strip()
            output.write("%s\n"%fname)
            
    output.close()
    
    
        
        
            
        
        
        
        
        
        
    