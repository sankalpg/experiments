import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP
import copy

def createPerFilePerAnotationsAdditions(anotationsAdditionFile, anotDetailFile, outFile):
  """
  This function will create a text file which will have annotation details to add in that file. This way we can edit file by file which is efficient 
  """
  
  fid = open(outFile,"w")
  
  anotColms = [1000, 2001, 3000, 5000, 8000]
  
  anotInfoData = np.loadtxt(anotDetailFile)
  
  anotInd2Add = np.loadtxt(anotationsAdditionFile)
  
  catWiseData = {}
  filesList = []
  
  for ii, anot in enumerate(anotColms):
    catWiseData[anot]=[]
    newAnnotations = anotInd2Add[:,ii]
    for jj, newanot in enumerate(newAnnotations):
      if newanot == -1:
        continue
      indAnot = np.where(anotInfoData[:,-1]==newanot)[0]
      catWiseData[anot].extend(anotInfoData[indAnot,:3].tolist())
      filesList.extend(anotInfoData[indAnot,2])
      
  filesList = np.unique(np.array(filesList)).tolist()
  
  fileWiseData = {}
  for fileID in filesList:
    fileWiseData[fileID] = []
    for k in catWiseData.keys():
      for info in catWiseData[k]:
        if info[-1] == fileID:
          #fileWiseData[fileID].append(info)
          infoTemp = copy.copy(info)
          infoTemp.append(k)
          #print infoTemp
          fileWiseData[fileID].append(infoTemp)
    
  
  print catWiseData
  print fileWiseData
  
  for fileId in fileWiseData.keys():
    for info in fileWiseData[fileId]:
      fid.write("%f\t"*len(info)%tuple(info))
      fid.write("\n")
      
  fid.close()
  
  
  
  
      
    
  
  
  
  
  
  