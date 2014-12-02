import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/labelParser/'))

import batchProcessing as BP
import textgridParser as TGParser


allowedPhrases = ['DnDP', 'mnDP', 'GmGRGP']

allowedPhrases_NEWFORMAT = ['A', 'B','C', 'D','F']

def organizeIITBAnnots(root_Dir, inpExt, outExt, FilterOutput, dumpFile):
    
    filenames = BP.GetFileNamesInDir(root_Dir, inpExt)
    
    motifDB={}
    baseNumber = 1000
    
    for filename in filenames:
        annots = TGParser.TextGrid2Dict(filename, 'Kaustuv_Anotations_Simple')
        fname, ext = os.path.splitext(filename)
        
        outFile = open(fname + outExt, "w")
        
        for phrase in annots.keys():
            if not motifDB.has_key(phrase):
                motifDB[phrase]=baseNumber
                baseNumber = baseNumber + 1
            for phraseSub in annots[phrase].keys():                
                if allowedPhrases.count(phrase)>0:
                    for patt in annots[phrase][phraseSub]:
                        outFile.write("%f\t%f\t%d\n"%(patt[0], patt[1], motifDB[phrase]))
                    
    
    outFile.close()
    
    dump = open(dumpFile,"w")
    for key in motifDB.keys():
        dump.write("%s\t%d\n"%(key, motifDB[key]))
    dump.close()
    
def organizeIITBAnnots_NEWFORMAT(root_Dir, inpExt, outExt, FilterOutput, dumpFile):
    
    filenames = BP.GetFileNamesInDir(root_Dir, inpExt)
    
    motifDB={}
    baseNumber = 1000
    
    for filename in filenames:
        annots = TGParser.TextGrid2Dict_NEWFORMAT(filename)
        fname, ext = os.path.splitext(filename)
        
        outFile = open(fname + outExt, "w")
        
        for phrase in annots.keys():
            print phrase
            if not motifDB.has_key(phrase):
                motifDB[phrase]=baseNumber
                baseNumber = baseNumber + 1
            for patt in annots[phrase]:                
                if FilterOutput ==1:
                  if allowedPhrases_NEWFORMAT.count(phrase)>0:
                    outFile.write("%f\t%f\t%d\n"%(float(patt[0]), float(patt[1]), motifDB[phrase]))
                else:
                  outFile.write("%f\t%f\t%d\n"%(float(patt[0]), float(patt[1]), motifDB[phrase]))
                    
    
    outFile.close()
    
    dump = open(dumpFile,"w")
    for key in motifDB.keys():
        dump.write("%s\t%d\n"%(key, motifDB[key]))
    dump.close()    
    
    
def generateDBStats(root_Dir, anotExt = '.anot'):
    
    filenames = BP.GetFileNamesInDir(root_Dir, anotExt)
    motifDB= {}
    for filename in filenames:
        lines = open(filename).readlines()
        for line in lines:
            splitLine = line.split()
            label = splitLine[2].strip()
            if not motifDB.has_key(label):
                motifDB[label]=0
            motifDB[label] = motifDB[label]+1
    
    print "Total annotations"
    for key in motifDB:
        print key, motifDB[key]
        


