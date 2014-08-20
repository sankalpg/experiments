import csv,sys,os
import numpy as np


def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter in f.lower():
                #print(path+"/"+f)
                #print(path)
                #ftxt.write(path + "/" + f + "\n")
                names.append(path + "/" + f)
                
    return names


def GetTonicFromFile(tonicfile, filename):    #needed for handling Vignesh's database

    tonicvals = csv.reader(open(tonicfile,'r'),delimiter='\t')
    for row in tonicvals:
        if row[0] == filename:
            return row[1]
    return -1

def GetFileNameFromVigneshConvention(pitchfilename): #needed for handling vignesh's database

    filename ,ext = os.path.splitext(pitchfilename)
    out = filename.split('/')[-1].split('.')[0] + '.wav'

    return out

def generateIndividualTonicFiles(root_dir):
    
    audiofiles = GetFileNamesInDir(root_dir)
    
    for audiofile in audiofiles:
        
        filename ,ext = os.path.splitext(audiofile)
        out = filename.split('/')[-1].split('.')[0] + '.wav'
    
        tonic = GetTonicFromFile('correct_tonic_values.txt', out)
        if tonic >0:
            np.savetxt(filename+'.tonic',np.array([np.float(tonic)]),fmt='%.3f')
        else:
            print filename
            

if __name__=="__main__":
    
    generateIndividualTonicFiles(sys.argv[1])
        