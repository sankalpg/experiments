import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP
eps = np.finfo(np.float64).resolution

serverPrefixC = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/CarnaticAlaps_IITM_edited/'
localPrefixC = '/media/Data/Datasets/MotifDiscovery_Dataset/CarnaticAlaps_IITM_edited/'

serverPrefixH = '/homedtic/sgulati/motifDiscovery/dataset/hindustani/IITB_Dataset_New/'
localPrefixH = '/media/Data/Datasets/MotifDiscovery_Dataset/IITB_Dataset_New/'


def changePrefix(audiofile):
    
    if audiofile.count(serverPrefixH):
        audiofile = localPrefixH + audiofile.split(serverPrefixH)[1]
    if audiofile.count(serverPrefixC):
        audiofile = localPrefixC + audiofile.split(serverPrefixC)[1]
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
              
              
def dumpAudioMelodyTrueGTAnotsFromDB(subSeqFileTNFNC, subSeqFileTN,  patternInfoFileFNC, patternInfoFile, outputDir, fileListDB, anotExt = '.anot', audioExt = '.wav', hopSize=0.01, subSeqLen = 800):
  
  
  filelistFiles = open(fileListDB,'r').readlines()
  
  subData = np.fromfile(subSeqFileTN)
  if np.mod(len(subData),subSeqLen)!=0:
    print "Please provide a subsequence database and subSeqLen which make sense, total number of elements in the database should be multiple of subSeqLen"
    return -1
  subData = np.reshape(subData, (len(subData)/float(subSeqLen), subSeqLen))
  
  subDataFNC = np.fromfile(subSeqFileTNFNC)
  if np.mod(len(subDataFNC),subSeqLen)!=0:
    print "Please provide a subsequence database and subSeqLen which make sense, total number of elements in the database should be multiple of subSeqLen"
    return -1
  subDataFNC = np.reshape(subDataFNC, (len(subDataFNC)/float(subSeqLen), subSeqLen))  
  
  #reading the info file and database file to create a mapping
  pattInfosFNC = np.loadtxt(patternInfoFileFNC)
  pattInfos = np.loadtxt(patternInfoFile)

  lineToType = -1*np.ones(pattInfosFNC.shape[0])
  
  #obtaining only query indeces (and not noise candidate indices)
  qInds = np.where(pattInfosFNC[:,3]>-1)[0]
  
  #iterating over all the unique file indices in the queries
  for fileInd in np.unique(pattInfosFNC[qInds][:,2]):
    
    fileInd = int(fileInd)
    #open the annotation file for this index
    annots = np.loadtxt(changePrefix(filelistFiles[fileInd].strip()+anotExt))
    
    if annots.shape[0] == annots.size:
        annots = np.array([annots])
    
    indSingleFile = np.where(pattInfosFNC[qInds][:,2]==fileInd)[0]
    ind = pattInfosFNC[qInds][indSingleFile,3].astype(int).tolist()
    
    for ii,val in enumerate(annots[ind,2]):
      lineToType[qInds[indSingleFile[ii]]] =  val
  
  totalPattTypes = np.unique(lineToType[qInds])
  
  fig = plt.figure()
  for pattType in totalPattTypes:
    localDir = os.path.join(outputDir, str(pattType))
    if not os.path.exists(localDir):
      os.makedirs(localDir)
    indsPatt = np.where(lineToType==pattType)[0]
    for indPatt in indsPatt:
        outFileName = os.path.join(localDir, str(indPatt)+'.png')
        outAudioFileName = os.path.join(localDir, str(indPatt)+audioExt)
        pitch = subData[indPatt, :np.floor(pattInfos[indPatt,1]/hopSize)]
        pitchFNC = subDataFNC[indPatt, :np.floor(pattInfosFNC[indPatt,1]/hopSize)]
        audioFile = changePrefix(filelistFiles[pattInfosFNC[indPatt,2].astype(np.int)].strip() + audioExt)
        plt.plot(pitch,'k.')
        plt.plot(pitchFNC, 'r')
        fig.tight_layout()        
        plt.axis([0,600, -500, 3000])
        fig.savefig(outFileName, dpi=75, bbox_inches='tight')
        fig.clear()
        clipAudio(audioFile, outAudioFileName, pattInfos[indPatt,0], pattInfos[indPatt,0] + pattInfos[indPatt,1])

              
def dumpAudioMelodyTrueGTAnotsWithLoudness(root_dir, outputDir, patternID, annotExt = '.anot', pitchExt = '.tpe', audioExt = '.wav', tonicExt = '.tonic', loudExt = '.loudness'):
  
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
      timeLoudness = np.loadtxt(fname + loudExt)
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
              pitch = timePitch[ind1:ind2,1]
              pitch = 1200*np.log2((pitch+eps)/tonic)
              pitch = pitch - np.mean(pitch)
              plt.plot(pitch,'b')
              loudness = timeLoudness[ind1:ind2,1]
              loudness = loudness-np.mean(loudness)
              loudness = (loudness)*200.0/np.std(loudness)
              plt.plot(loudness,'r')
              fig.tight_layout()
              outFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.png')
              outAudioFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.mp3')
              plt.axis([0,800, -900, 900])
              fig.savefig(outFileName, dpi=75, bbox_inches='tight')
              fig.clear()
              clipAudio(audioFile, outAudioFileName, sTime, eTime)
              
def dumpAudioMelodyTrueGTAnotsMarkingFlatNotes(root_dir, outputDir, patternID, annotExt = '.anot', pitchExt = '.pitchEssentiaIntp', audioExt = '.wav', tonicExt = '.tonic', segExt = '.segmentsNyas'):
  
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
      segments = np.loadtxt(fname + segExt)
      
      flatSegs = np.zeros(timePitch.shape[0])
      for segment in segments:
        ind1 = find_nearest_element_ind(timePitch[:,0], segment[0])
        ind2 = find_nearest_element_ind(timePitch[:,0], segment[1])
        flatSegs[ind1:ind2+1]=1
        
      
      for ii, line in enumerate(lines):
          lineSplit = line.split()
          sTime = float(lineSplit[0])
          eTime = float(lineSplit[1])
          pattID = int(lineSplit[2])
          
          if pattID == int(patternID):          
              ind1 = find_nearest_element_ind(timePitch[:,0], sTime)
              ind2 = find_nearest_element_ind(timePitch[:,0], eTime)
              
              indFlats = np.where(flatSegs[ind1:ind2]==1)[0]
              pitch  = timePitch[ind1:ind2,1]
              plt.plot(1200*np.log2((pitch+eps)/tonic), 'b')
              plt.scatter(np.arange(len(pitch))[indFlats], 1200*np.log2((pitch[indFlats]+eps)/tonic))
              outFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.png')
              outAudioFileName = os.path.join(inDir, fname.split('/')[-1]+ '_' + str(ii)+'.mp3')
              plt.axis([0,500, 0, 1200])
              fig.savefig(outFileName, dpi=75, bbox_inches='tight')
              fig.clear()
              clipAudio(audioFile, outAudioFileName, sTime, eTime)              
              
def dumpFalseAlarms(outputDir, searchPatternFile, patternInfoFile, fileListDB, anotExt = '.anot', topN = 200, pitchExt = '.pitchIntrp', tonicExt = '.tonic', audioExt = '.mp3'):
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
  dumpInfoGlobal = np.delete(dumpInfoGlobal,0,0)
  indUniqFiles = np.unique(dumpInfoGlobal[:,2])
  fig = plt.figure()
  for fileId in indUniqFiles:
    indFiles = np.where(dumpInfoGlobal[:,2]==fileId)[0]
    fname = changePrefix(filelistFiles[fileId.astype(np.int)]).strip()
    timePitch = np.loadtxt(fname + pitchExt)
    tonic = np.loadtxt(fname + tonicExt)
    for ind in indFiles:
      dirName = os.path.join(outputDir, str(dumpInfoGlobal[ind,4].astype(int)), str(dumpInfoGlobal[ind,5].astype(np.int)))
      if not os.path.exists(dirName):
        os.makedirs(dirName)
      ind1 = find_nearest_element_ind(timePitch[:,0], dumpInfoGlobal[ind,0])
      ind2 = find_nearest_element_ind(timePitch[:,0], dumpInfoGlobal[ind,0]+dumpInfoGlobal[ind,1])
      pitch = timePitch[ind1:ind2,1]
      plt.plot(1200*np.log2((pitch+eps)/tonic), 'b')
      outFileName = os.path.join(dirName,  str(ind.astype(np.int))+'.png')
      outAudioFileName = os.path.join(dirName,  str(ind.astype(np.int))+'.mp3')
      plt.axis([0,500, 0, 3000])
      fig.savefig(outFileName, dpi=75, bbox_inches='tight')
      fig.clear()
      print dumpInfoGlobal[ind,0], dumpInfoGlobal[ind,0]+dumpInfoGlobal[ind,1]
      clipAudio(fname+audioExt, outAudioFileName, dumpInfoGlobal[ind,0], dumpInfoGlobal[ind,0]+dumpInfoGlobal[ind,1])
  
  np.savetxt(os.path.join(outputDir, 'falseAlarmsDetails.txt'),dumpInfoGlobal)
  return dumpInfoGlobal



def dumpFalseAlarmsCombined(outputDir, searchPatternFile, patternInfoFile, fileListDB, anotExt = '.anot', topNGlobal = 200, topNPerQ = 20, pitchExt = '.pitch', tonicExt = '.tonic', audioExtIn = '.mp3', audioExtOut = '.mp3', withContext=0):
  """
  This function dumps false positives.
  topNGlobal number of false positives per pattern class and topNPerQ number of false positives per query
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
  
  
  
  falsePostivesInfo = {}
  for queryPattType in totalPattTypes:
    
    indQPattType = np.where(line2TypeSearch[:,0]==queryPattType)[0]
    
    #Obtaining top topNGlobal false positives per pattern class, making a list of their ids (line numbers)
    falsePostivesInfo[queryPattType] = {}
    indWrong = np.where(line2TypeSearch[indQPattType,0]!=line2TypeSearch[indQPattType,1])[0]
    
    indSort = np.argsort(searchPatts[indQPattType[indWrong],2])[:topNGlobal]
    storeInds = indQPattType[indWrong[indSort]]
    
    lineStore = searchPatts[storeInds.astype(np.int),1]
    lineStore = np.unique(lineStore)
    
    falsePostivesInfo[queryPattType]['gobal'] = lineStore.tolist()
    
    #obtaining top topNPerQ number of false positives per query for this class.
    queryIndsUnique = np.unique(searchPatts[indQPattType,0])
    falsePostivesInfo[queryPattType]['local'] = []
    for Qind in queryIndsUnique:
      indQs = np.where(searchPatts[indQPattType,0]==Qind)[0]
      indWrong = np.where(line2TypeSearch[indQPattType[indQs],0]!=line2TypeSearch[indQPattType[indQs],1])[0]
      indSort = np.argsort(searchPatts[indQPattType[indQs[indWrong]],2])[:topNPerQ]
      storeInds = indQPattType[indQs[indWrong[indSort]]]
      lineStore = searchPatts[storeInds.astype(np.int),1]
      lineStore = np.unique(lineStore)
      falsePostivesInfo[queryPattType]['local'].extend(lineStore.tolist())
 
  dumpInfoOverall = np.zeros((1,7))
  for queryPattType in totalPattTypes:
    linesToDump = []
    linesToDump.extend(falsePostivesInfo[queryPattType]['gobal'])
    linesToDump.extend(falsePostivesInfo[queryPattType]['local'])
    linesToDump = np.unique(np.array(linesToDump))
    print len(linesToDump)
    dumpInfo = np.zeros((linesToDump.size,7))
    dumpInfo[:,:4]= pattInfos[linesToDump.astype(np.int),:4]
    dumpInfo[:,4]= queryPattType
    dumpInfo[:,6]= linesToDump.astype(np.int)
    dumpInfoOverall = np.vstack((dumpInfoOverall, dumpInfo))
  
  dumpInfoOverall = np.delete(dumpInfoOverall,0,0)
 
  indUniqFiles = np.unique(dumpInfoOverall[:,2])
  fig = plt.figure()
  for fileId in indUniqFiles:
    indFiles = np.where(dumpInfoOverall[:,2]==fileId)[0]
    fname = changePrefix(filelistFiles[fileId.astype(np.int)]).strip()
    timePitch = np.loadtxt(fname + pitchExt)
    tonic = np.loadtxt(fname + tonicExt)
    for ind in indFiles:
      if (withContext):
        starTime = max(0, dumpInfoOverall[ind,0]-1)#take one second before
        EndTime = min(timePitch[-1,0],dumpInfoOverall[ind,0] + dumpInfoOverall[ind,1]+1) #take one second after
      else:
        starTime = max(0, dumpInfoOverall[ind,0])
        EndTime = min(timePitch[-1,0],dumpInfoOverall[ind,0] + dumpInfoOverall[ind,1])
      dirName = os.path.join(outputDir, str(dumpInfoOverall[ind,4].astype(int)), str(dumpInfoOverall[ind,5].astype(np.int)))
      if not os.path.exists(dirName):
        os.makedirs(dirName)
      ind1 = find_nearest_element_ind(timePitch[:,0], starTime)
      ind2 = find_nearest_element_ind(timePitch[:,0], EndTime)
      pitch = timePitch[ind1:ind2,1]
      plt.plot(1200*np.log2((pitch+eps)/tonic), 'b')
      outFileName = os.path.join(dirName,  str(dumpInfoOverall[ind.astype(np.int),6])+'.png')
      outAudioFileName = os.path.join(dirName, str(dumpInfoOverall[ind.astype(np.int),6]) + audioExtIn)
      plt.axis([0,800, -700, 3000])
      fig.savefig(outFileName, dpi=75, bbox_inches='tight')
      fig.clear()
      clipAudio(fname+audioExtOut, outAudioFileName, starTime, EndTime)
  
  np.savetxt(os.path.join(outputDir, 'falseAlarmsDetails.txt'),dumpInfoOverall)
  return 1

  

def plotTwoMatchedPatterns(pattDataFile, nSamplesPerPatt, ind1, ind2):
  data = np.fromfile(pattDataFile)
  data1 = np.reshape(data,(len(data)/nSamplesPerPatt, nSamplesPerPatt))
  plt.plot(data1[ind1,:]-np.mean(data1[ind1,:]),'b')
  plt.plot(data1[ind2,:]-np.mean(data1[ind2,:]),'r')
  plt.show()
  
  
  
def dumpPatterns(outputDir, searchPatternFile, DBFile, fileListDB, topNGlobal = 200, topNPerQ = 20, patternInfoExt = '.SubSeqsInfo', anotExt = '.anot', tonicExt = '.tonic', audioExtIn = '.mp3', audioExtOut = '.mp3', withContext=0, data2Dump = [['FILE', '.pitch', 1, '.tonic']], nSamplesDB=-1):
  """
  This function dumps false positives.
  topNGlobal number of false positives per pattern class and topNPerQ number of false positives per query
  
  , ['DB', '.complexity1', 1], ['FILE', '.loudness',1]
  """
  
  filelistFiles = open(fileListDB,'r').readlines()
  
  #reading the info file and database file to create a mapping
  pattInfos = np.loadtxt(DBFile+patternInfoExt)

  lineToType = -1*np.ones(pattInfos.shape[0])
  
  #obtaining only query indeces (and not noise candidate indices)
  qInds = np.where(pattInfos[:,3]>-1)[0]
  
  #Since DB dumps created for experiments are single files they can be read once and for all
  DBdata = []
  for dataInfo in data2Dump:
    if dataInfo[0]=='DB':
      dataTemp = np.fromfile(DBFile+dataInfo[1])
      dataTemp = np.reshape(dataTemp, (len(dataTemp)/nSamplesDB, nSamplesDB))
      DBdata.append(dataTemp)
    elif dataInfo[0]=='FILE':
      DBdata.append(-1)
    else:
      print "Please provide a valid type of mode to read, Either 'DB' to read data from the DB Dump or 'FILE' to read from the text files."
      return 0
    
  
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
  
  falsePostivesInfo = {}
  for queryPattType in totalPattTypes:
    
    indQPattType = np.where(line2TypeSearch[:,0]==queryPattType)[0]
    
    #Obtaining top topNGlobal false positives per pattern class, making a list of their ids (line numbers)
    falsePostivesInfo[queryPattType] = {}
    indWrong = np.where(line2TypeSearch[indQPattType,0]!=line2TypeSearch[indQPattType,1])[0]
    
    indSort = np.argsort(searchPatts[indQPattType[indWrong],2])[:topNGlobal]
    storeInds = indQPattType[indWrong[indSort]]
    
    lineStore = searchPatts[storeInds.astype(np.int),1]
    lineStore = np.unique(lineStore)
    
    falsePostivesInfo[queryPattType]['gobal'] = lineStore.tolist()
    
    #obtaining top topNPerQ number of false positives per query for this class.
    queryIndsUnique = np.unique(searchPatts[indQPattType,0])
    falsePostivesInfo[queryPattType]['local'] = []
    for Qind in queryIndsUnique:
      indQs = np.where(searchPatts[indQPattType,0]==Qind)[0]
      indWrong = np.where(line2TypeSearch[indQPattType[indQs],0]!=line2TypeSearch[indQPattType[indQs],1])[0]
      indSort = np.argsort(searchPatts[indQPattType[indQs[indWrong]],2])[:topNPerQ]
      storeInds = indQPattType[indQs[indWrong[indSort]]]
      lineStore = searchPatts[storeInds.astype(np.int),1]
      lineStore = np.unique(lineStore)
      falsePostivesInfo[queryPattType]['local'].extend(lineStore.tolist())
    
    #also storing true positives
    indTrue = np.where(lineToType==queryPattType)[0]
    falsePostivesInfo[queryPattType]['true'] = indTrue.tolist()
 
  dumpInfoOverall = np.zeros((1,7))
  for queryPattType in totalPattTypes:
    linesToDump = []
    linesToDump.extend(falsePostivesInfo[queryPattType]['gobal'])
    linesToDump.extend(falsePostivesInfo[queryPattType]['local'])
    linesToDump = np.unique(np.array(linesToDump)).tolist()
    
    NTrue = len(falsePostivesInfo[queryPattType]['true'])
    linesToDump.extend(falsePostivesInfo[queryPattType]['true'])
    print len(linesToDump)
    linesToDump = np.array(linesToDump)
    
    dumpInfo = np.zeros((linesToDump.size,7))
    dumpInfo[:,:4]= pattInfos[linesToDump.astype(np.int),:4]
    dumpInfo[:,4]= queryPattType
    dumpInfo[:,6]= linesToDump.astype(np.int)
    dumpInfo[-NTrue:,5] = queryPattType
    dumpInfoOverall = np.vstack((dumpInfoOverall, dumpInfo))
  
  dumpInfoOverall = np.delete(dumpInfoOverall,0,0)
 
  indUniqFiles = np.unique(dumpInfoOverall[:,2])
  fig = plt.figure()
  colors = ['r', 'b', 'g', 'k']
  for fileId in indUniqFiles:
    indFiles = np.where(dumpInfoOverall[:,2]==fileId)[0]
    fname = changePrefix(filelistFiles[fileId.astype(np.int)]).strip()
    FILEdata = []
    tonic=1
    
    for dataInfo in data2Dump:
      if dataInfo[0]=='FILE':
        tempData = np.loadtxt(fname + dataInfo[1])
        FILEdata.append(tempData)
        #if there are 4 elements in the array, that means its a pitch data to be displayed and the 4th element is tonic extionsion
        if len(dataInfo)==4:
          tonic = float(np.loadtxt(fname + dataInfo[3]).astype(np.float))
      else:
        FILEdata.append(-1)
    
    for ind in indFiles:
      #creating directory to store the dump if it doesn;t exist
      dirName = os.path.join(outputDir, str(dumpInfoOverall[ind,4].astype(int)), str(dumpInfoOverall[ind,5].astype(np.int)))
      if not os.path.exists(dirName):
        os.makedirs(dirName)
        
      for ii,dataInfo in enumerate(data2Dump):
        if  dataInfo[0]=='DB':
          plt.plot(DBdata[ii][dumpInfoOverall[ind,6]]/dataInfo[2], colors[ii])
        else:
          if (withContext):
            starTime = max(0, dumpInfoOverall[ind,0]-1)#take one second before
            EndTime = min(FILEdata[ii][-1,0],dumpInfoOverall[ind,0] + dumpInfoOverall[ind,1]+1) #take one second after
          else:
            starTime = max(0, dumpInfoOverall[ind,0])
            EndTime = min(FILEdata[ii][-1,0],dumpInfoOverall[ind,0] + dumpInfoOverall[ind,1])
          ind1 = find_nearest_element_ind(FILEdata[ii][:,0], starTime)
          ind2 = find_nearest_element_ind(FILEdata[ii][:,0], EndTime)
          data2Plot = FILEdata[ii][ind1:ind2,1]
          if len(dataInfo)==4:
            data2Plot = 1200*np.log2((data2Plot+eps)/tonic)
          plt.plot(data2Plot/dataInfo[2], colors[ii])
      
      outFileName = os.path.join(dirName,  str(dumpInfoOverall[ind.astype(np.int),6])+'.png')
      outAudioFileName = os.path.join(dirName, str(dumpInfoOverall[ind.astype(np.int),6]) + audioExtIn)
      plt.axis([0,800, -700, 3000])
      fig.savefig(outFileName, dpi=150, bbox_inches='tight')
      fig.clear()
      #clipAudio(fname+audioExtOut, outAudioFileName, starTime, EndTime)
  
  np.savetxt(os.path.join(outputDir, 'falseAlarmsDetails.txt'),dumpInfoOverall)
  return 1
