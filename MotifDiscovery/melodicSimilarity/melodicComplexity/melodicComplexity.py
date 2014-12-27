import numpy as np
import os, sys
sys.path.append('../../RepAndDistComparison/')
sys.path.append('../../RepAndDistComparison/PerformanceAnalysis')
import matplotlib.pyplot as plt
eps = np.finfo(np.float).eps
import EvaluateSupervisedAnalysis as Eval

serverPrefix = '/homedtic/sgulati/motifDiscovery/dataset/carnatic/CarnaticAlaps_IITM_edited/'
localPrefix = '/media/Data/Datasets/MotifDiscovery_Dataset/CarnaticAlaps_IITM_edited/'

def changePrefix(audiofile):
    
    if audiofile.count(serverPrefix):
        audiofile = localPrefix + audiofile.split(serverPrefix)[1]
    return audiofile

def computeGlobalMelodicComplexity(pitch):
  
  diff = pitch[1:]-pitch[:-1]  
  inds = np.where(abs(diff)>=600)[0]
  diff = np.delete(diff, inds)
  if len(diff)!=0:
    complexity = np.sqrt(np.sum(np.power(diff,2)))/len(diff)
  else:
    print "Here comes zero length diff"
    complexity = 10000
  
  return complexity

def computeLocalMelodicComplexity(pitch, halfWinLen):
  N = pitch.size
  complexity = np.zeros(pitch.size)
  diff = np.zeros(pitch.size)
  diff[:-1] = pitch[1:]-pitch[:-1]
  diff[-1] = diff[-2]
  diff = np.power(diff,2)
  for ii in range(N):
    start = min( max(0,ii-halfWinLen), N - (2*halfWinLen + 1))
    end  = start + (2*halfWinLen + 1)
    complexity[ii] = np.sum(diff[start:end])

  return complexity


def generateComplexityDB(outputDBFile, pattDataFile, pattInfoFile, nSamplesPerPatt, hopSize, winLen = 0.3):
  fid = open(outputDBFile, 'w')
  fid.close()
  # read the information about the patterns
  pattInfo = np.loadtxt(pattInfoFile)
  #read the pattern data
  pattData = np.fromfile(pattDataFile)
  pattData = np.reshape(pattData, (len(pattData)/nSamplesPerPatt, nSamplesPerPatt))

  fid = open(outputDBFile, 'ab')
  halfWinSamples = np.round((winLen/2.0)/hopSize).astype(np.int)
  temp = np.zeros(nSamplesPerPatt)
  for ii, pattern in enumerate(pattData):
    temp = temp*0
    pattLen = np.round(pattInfo[ii,1]/hopSize).astype(np.int)
    pattPitch = pattData[ii,:pattLen+1]
    temp[:pattLen+1] = computeLocalMelodicComplexity(pattPitch, halfWinSamples)
    fid.write(temp)

  fid.close()





    



def getGlobalComplexityFor_FP_TP(searchPatternFile, pattDataFile, pattInfoFile, fileListFile, nSamplesPerPatt, hopSize, tonicExt = '.tonic'):
  """
  This function computes global complexity measure for all the true hits and false hits to see which parameters of complexity computation provides best separation in the twoclasses
  """
  
  ###first computing global complexities for all the patterns in the database
  
  # read the information about the patterns
  pattInfo = np.loadtxt(pattInfoFile)
  #read the pattern data
  pattData = np.fromfile(pattDataFile)
  pattData = np.reshape(pattData, (len(pattData)/nSamplesPerPatt, nSamplesPerPatt))
  
  #read tonic values of each file
  tonic = []
  lines = open(fileListFile).readlines()
  for line in lines:
    fname = changePrefix(line.strip())
    tonic.append(np.loadtxt(fname + tonicExt))
  
  #compute global complexity of each pattern
  globalComplexity = np.zeros(pattInfo.shape[0])
  
  for ii, pattern in enumerate(pattData):
    pattLen = np.round(pattInfo[ii,1]/hopSize).astype(np.int)
    if pattLen == 0:
      print "Print what the heck is this, 0 Pattern length, non sense!!", ii
    pattPitch = pattData[ii,:pattLen+1]
    globalComplexity[ii] = computeGlobalMelodicComplexity(pattPitch)
  
  
  results = Eval.evaluateSupSearchNEWFORMAT(searchPatternFile, pattInfoFile, fileListFile)
  
  #reading searched pattern details 
  searchedData = np.loadtxt(searchPatternFile)
  complexityMeasure = np.zeros(searchedData.shape[0])
  for ii, line in enumerate(searchedData):
    cmp1 = globalComplexity[line[0].astype(np.int)]
    cmp2 = globalComplexity[line[1].astype(np.int)]
    if float(min(cmp1,cmp2)) !=0:
      complexityMeasure[ii] = max(cmp1, cmp2)/float(min(cmp1,cmp2))
    else:
      complexityMeasure[ii] = 10000
  
  indCorrect = np.where(results[2]==1)[0]
  indWrong = np.where(results[2]==0)[0]
  print len(indCorrect), len(indWrong)
  print complexityMeasure[indCorrect], complexityMeasure[indWrong]
  
  val1 = complexityMeasure[indCorrect]
  len1= len(val1)
  val2 = complexityMeasure[indWrong]
  len2= len(val2)
  
  #plt.plot(val1)
  #plt.plot(val2)
  #plt.show()
  
  searchedData = np.hstack((searchedData, np.transpose(np.array([results[2]]))))
  searchedData = np.hstack((searchedData, np.transpose(np.array([complexityMeasure]))))
  
  np.savetxt('complexityData.txt', searchedData)
  
  #valArray = np.append(val1,val2)
  #valArray = np.sort(valArray)
  
  #recall=[]
  #precision=[]
  #for ii in range(0, len(valArray), 100):
    #threshold = valArray[ii]
    #ind1 = len(np.where(val1<threshold)[0])
    #ind2 = len(np.where(val2<threshold)[0])
    ##print threshold, ind1+ind2
    #if (ind1+ind2)>0:
      #precision.append(ind1/float(ind1+ind2))
      #recall.append(ind1/float(len1))
      #print len(recall)
      
  #recall = np.array(recall)
  #precision = np.array(precision)
  #plt.plot(1-precision,recall)
  #plt.show()
    
    
def plotPatternsComputeComplexity(pattDataFile, pattInfoFile, fileListFile, nSamplesPerPatt, hopSize, tonicExt = '.tonic'):
  
  # read the information about the patterns
  pattInfo = np.loadtxt(pattInfoFile)
  #read the pattern data
  pattData = np.fromfile(pattDataFile)
  pattData = np.reshape(pattData, (len(pattData)/nSamplesPerPatt, nSamplesPerPatt))
  
  #read tonic values of each file
  tonic = []
  lines = open(fileListFile).readlines()
  for line in lines:
    fname = changePrefix(line.strip())
    tonic.append(np.loadtxt(fname + tonicExt))
    
   #compute global complexity of each pattern
  globalComplexity = np.zeros(pattInfo.shape[0])
  
  for ii, pattern in enumerate(pattData):
    pattLen = np.round(pattInfo[ii,1]/hopSize).astype(np.int)
    pattPitch = pattData[ii,:pattLen+1]
    globalComplexity[ii] = computeGlobalMelodicComplexity(pattPitch)
    plt.plot(pattPitch)
    plt.show()
    print globalComplexity[ii]
  
  
  
  
  
    
  
  
  
  
  
