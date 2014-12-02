import numpy as np
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP
eps = np.finfo(np.float64).resolution

serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/CarnaticAlaps_IITM_edited/'
localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/CarnaticAlaps_IITM_edited/'

def changePrefix(audiofile):
    
    if audiofile.count(serverPrefix):
        audiofile = localPrefix + audiofile.split(serverPrefix)[1]
    return audiofile

def find_nearest_element_ind(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    

def plotPatternsOverlay(root_dir, patternID, annotExt = '.anot', pitchExt = '.pitchEssentiaIntp'):
    
    afiles = BP.GetFileNamesInDir(root_dir, annotExt)
    
    plt.hold(True)
    
    for f in afiles:
        lines = open(f).readlines()
        #reading the pitch 
        
        fname,ext = os.path.splitext(f)
        timePitch = np.loadtxt(fname + pitchExt)
        
        for line in lines:
            lineSplit = line.split()
            sTime = float(lineSplit[0])
            eTime = float(lineSplit[1])
            pattID = int(lineSplit[2])
            
            if pattID == int(patternID):
            
                ind1 = find_nearest_element_ind(timePitch[:,0], sTime)
                ind2 = find_nearest_element_ind(timePitch[:,0], eTime)
            
                plt.plot(timePitch[ind1:ind2,1])
    plt.show()


def find_nearest_element_ind(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    

def plotPatternsSave(root_dir, outputDir, patternID, annotExt = '.anot', pitchExt = '.pitchEssentiaIntp'):
    
    afiles = BP.GetFileNamesInDir(root_dir, annotExt)
    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    inDir = os.path.join(outputDir,str(patternID))
    if not os.path.exists(inDir):
        os.mkdir(inDir)
    fig = plt.figure()
    for f in afiles:
        lines = open(f).readlines()
        #reading the pitch 
        
        fname,ext = os.path.splitext(f)
        timePitch = np.loadtxt(fname + pitchExt)
        
        for ii, line in enumerate(lines):
            lineSplit = line.split()
            sTime = float(lineSplit[0])
            eTime = float(lineSplit[1])
            pattID = int(lineSplit[2])
            
            if pattID == int(patternID):
            
                ind1 = find_nearest_element_ind(timePitch[:,0], sTime)
                ind2 = find_nearest_element_ind(timePitch[:,0], eTime)
                
                
                plt.plot(1200*np.log2((timePitch[ind1:ind2,1]+eps)/55.0))
                fig.tight_layout()
                outFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.png')
                plt.axis([0,2000, 0, 5000])
                fig.savefig(outFileName, dpi=75, bbox_inches='tight')
                fig.clear()

def clipAudio(inFile, outFile, start, end):
    
    cmd = "sox \"%s\" \"%s\" trim %s =%s"%(inFile, outFile, convertFormat(start), convertFormat(end))
    os.system(cmd)
    
    return 1    
  
def convertFormat(sec):
        
    hours = int(np.floor(sec/3600))
    minutes = int(np.floor((sec - (hours*3600))/60))
    
    seconds = sec - ( hours*3600 + minutes*60)
    
    return str(hours) + ':' + str(minutes) + ':' + str(seconds)  

def dumpAudioMelodyTrueGTAnots(root_dir, outputDir, patternID, annotExt = '.anot', pitchExt = '.pitchEssentiaIntp', audioExt = '.wav', tonicExt = '.tonic'):
  
  afiles = BP.GetFileNamesInDir(root_dir, annotExt)
  
  inDir = os.path.join(outputDir,str(patternID))
  if not os.path.exists(inDir):
      os.mkdir(inDir)
  fig = plt.figure()
  for f in afiles:
      lines = open(f).readlines()
      #reading the pitch       
      fname,ext = os.path.splitext(f)
      timePitch = np.loadtxt(fname + pitchExt)
      tonic = np.loadtxt(fname + tonicExt)
      audioFile = fname + audioExt
      
      for ii, line in enumerate(lines):
          lineSplit = line.split()
          sTime = float(lineSplit[0])
          eTime = float(lineSplit[1])
          pattID = int(lineSplit[2])
          
          if pattID == int(patternID):          
              ind1 = find_nearest_element_ind(timePitch[:,0], sTime)
              ind2 = find_nearest_element_ind(timePitch[:,0], eTime)
              
              plt.plot(1200*np.log2((timePitch[ind1:ind2,1]+eps)/tonic))
              fig.tight_layout()
              outFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.png')
              outAudioFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.mp3')
              plt.axis([-500,1000, 0, 5000])
              fig.savefig(outFileName, dpi=75, bbox_inches='tight')
              fig.clear()
              clipAudio(audioFile, outAudioFileName, sTime, eTime)
              
def dumpFalseAlarms(searchPatternFile, patternInfoFile, fileListDB, anotExt = '.anot', topN = 200):
  """
  This function dumps all the top false alarms by the system. 
  """
  
  filelistFiles = open(fileListDB,'r').readlines()
  
  #reading the info file and database file to create a mapping
  pattInfos = np.loadtxt(patternInfoFile)

  lineToType = -1*np.ones(pattInfos.shape[0])
  
  #obtaining only query indeces (and not noise candidate indices)
  qInds = np.where(pattInfos[:,3]>-1)[0]
  
  #iterating over all the unique file indices in the queries
  for fileInd in np.unique(pattInfos[qInds][:,2]):
    
    fileInd = int(fileInd)
    #open the annotation file for this index
    annots = np.loadtxt(changePrefix(filelistFiles[fileInd].strip()+anotExt))
    
    if annots.shape[0] == annots.size:
        annots = np.array([annots])
    
    indSingleFile = np.where(pattInfos[qInds][:,2]==fileInd)[0]
    ind = pattInfos[qInds][indSingleFile,3].astype(int).tolist()
    
    for ii,val in enumerate(annots[ind,2]):
      lineToType[qInds[indSingleFile[ii]]] =  val
      
  searchPatts = np.loadtxt(searchPatternFile)
    
  line2TypeSearch = np.zeros(searchPatts.shape)
  
  for ii in range(searchPatts.shape[0]):
    line2TypeSearch[ii,0] = lineToType[searchPatts[ii,0]]
    line2TypeSearch[ii,1] = lineToType[searchPatts[ii,1]]
  
  totalPattTypes = np.unique(lineToType[qInds])
    
  decArray = np.zeros(searchPatts.shape[0])
  
  indCorrect = np.where(line2TypeSearch[:,0]==line2TypeSearch[:,1])
  decArray[indCorrect] = 1
  
  totalPattTypes = np.unique(lineToType[qInds])
  
  dumpInfoGlobal = np.zeros((1,6))
  for queryPattType in totalPattTypes:
    storeInds = np.array([])
    indQPattType = np.where(line2TypeSearch[:,0]==queryPattType)[0]
    indCorrect = np.where(line2TypeSearch[indQPattType,0]==line2TypeSearch[indQPattType,1])[0]
    print indCorrect.shape
    indWrong = np.where(line2TypeSearch[indQPattType,0]!=line2TypeSearch[indQPattType,1])[0]
    indSort = np.argsort(searchPatts[indQPattType[indWrong],2])[:topN]
    storeInds = np.concatenate((storeInds, indQPattType[indCorrect]))
    storeInds = np.concatenate((storeInds, indQPattType[indWrong[indSort]]))
    print indQPattType[indWrong[indSort]]
    lineStore = searchPatts[storeInds.astype(np.int),1]
    lineStore = np.unique(lineStore)
    dumpInfo = np.zeros((lineStore.size,6))
    dumpInfo[:,4]= queryPattType
    dumpInfo[:,:4]= pattInfos[lineStore.astype(np.int),:4]
    
    for searchPattType in totalPattTypes:
      indCat = np.where(lineToType[lineStore.astype(np.int)]==searchPattType)[0]
      dumpInfo[indCat,5]= searchPattType
    
    dumpInfoGlobal = np.vstack((dumpInfoGlobal, dumpInfo))
   
  return dumpInfoGlobal
  
