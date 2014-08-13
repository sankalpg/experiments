import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP

def renameOrganizeMotifsPerRagaIITMDB(motifMapFile, inpAnnotFileExt, outAnnotFileExt):
    
    lines = open(motifMapFile, "r").readlines()
    
    for line in lines:
        splitLine = line.split("\t")
        root_dir = splitLine[0].strip()
        baseNumber = int(splitLine[1].strip())
        motifs = splitLine[2:]
        motifDB={}
        for m in motifs:
            motifDB[m.strip()]=[]
        
        filenames = BP.GetFileNamesInDir(root_dir, inpAnnotFileExt)
        motifNumber = {}
        for filename in filenames:
            
            if len(motifDB.keys())==0:
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
                        baseNumber+=1
                    if motifDB.has_key(label):    
                        outfile.write("%f\t%f\t%d\n"%(start, start + dur, motifNumber[label]))
            except:
                print "No format: " + filename
                    
            outfile.close()

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
    for ii, line in enumerate(lines):
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
            motifDB[id].append((ii,(line[0], line[1])))
    
    #for key in motifDB.keys():
        #print key, len(motifDB[key])
        
    return motifDB
        
            
            
        
    