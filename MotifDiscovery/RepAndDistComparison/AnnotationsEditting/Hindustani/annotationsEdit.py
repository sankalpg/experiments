import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP
import copy
import shutil

def find_nearest_element_ind(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx


def findAnotsWithSil(root_dir, outputFile,  anotExt = '.anot', pitchExt = '.pitch', silVal = 0):

	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	fid = open(outputFile, 'w')

	for filename in filenames:
		fname, ext = os.path.splitext(filename)
		timePitch = np.loadtxt(fname + pitchExt)
		anots = np.loadtxt(filename)

		for ii, anot in enumerate(anots):
			indStart = find_nearest_element_ind(timePitch[:,0], anot[0])
			indEnd = find_nearest_element_ind(timePitch[:,0], anot[1])
			if len(np.where(timePitch[indStart: indEnd,1]<=silVal)[0]) > 0 :
				fid.write("%s\t%d\t%d\t%d\n"%(filename, ii, anot[0], anot[1]))

	fid.close()

def findSmallAnots(root_dir, outputFile, anotExt = '.anot', Duration = 0.5):
	
	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	fid = open(outputFile, 'w')

	for filename in filenames:
		anots = np.loadtxt(filename)
		for ii, anot in enumerate(anots):
			if anot[1]-anot[0] <= Duration:
				fid.write("%s\t%d\t%f\t%f\n"%(filename, ii, anot[0], anot[1]))

	fid.close()		

def convertSonicAnotsToMine(root_dir, anotExt = '.anot', outExt = '.anot'):

	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	for filename in filenames:
		#print filename
		fname, ext = os.path.splitext(filename)
		anots = np.loadtxt(filename)
		anots[:,1] = anots[:,0] + anots[:,1]
		if anots[:,1].any()>anots[:,0].any():
			print filename
		np.savetxt(fname + outExt, anots, fmt='%.3f\t%.3f\t%d')

def renameAnotFiles(root_dir, inpExt = '.anot', outExt = '.anotEdit1'):

	filenames = BP.GetFileNamesInDir(root_dir, inpExt)

	for filename in filenames:
		fname, ext = os.path.splitext(filename)
		os.rename(filename, fname+outExt)
		
def copyAnotFiles(root_dir, inpExt = '', outExt = ''):

    filenames = BP.GetFileNamesInDir(root_dir, inpExt)

    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        shutil.copy(filename, fname+outExt)		

def formatAnots(root_dir, inpExt = '.anot'):

	filenames = BP.GetFileNamesInDir(root_dir, inpExt)

	for filename in filenames:
		anots = np.loadtxt(filename)
		np.savetxt(filename, anots, fmt='%.3f\t%.3f\t%d')

def createPerFilePerAnotationsAdditions(anotationsAdditionFile, anotDetailFile, outFile):
  """
  This function will create a text file which will have annotation details to add in that file. This way we can edit file by file which is efficient 
  """
  
  fid = open(outFile,"w")
  
  anotColms = [1043, 1046, 1042, 1047, 1044]
  
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
      #print anotInfoData[indAnot,:].tolist()
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
      info[1] = info[0] + info[1]
      fid.write("%f\t"*len(info)%tuple(info))
      fid.write("\n")
      
  fid.close()