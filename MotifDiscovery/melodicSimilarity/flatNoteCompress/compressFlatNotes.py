import numpy as np
import os, sys
sys.path.append('../../../../library_pythonnew/melodyProcessing/')
sys.path.append('../../../../library_pythonnew/batchProcessing/')
import segmentation as seg
import batchProcessing as BP


eps = np.finfo(np.float).eps

#1) Using PLS method - various criterions for segmentation, Modified PLS?

#Since PLS doesn't explicitely tell us about flat note but does segmentation, first step is to obtain segments

def flatNoteSegmentationPLS(pitchFile, maxAbsError, segFileExt = '.segmentsPLS'):
  """
  This function performs flat note segmentation using PLS method. I use the implementation of the Joan Serra that he gave me for the Nyas identification baseline.
  Input:
    pitchFile = csv file of the pitch values <first column time stamps> <second pitch values in Hz>
    maxAbsError = maximum absolute error used in the PLS method for deteting a segment
  Output:
   this function writes a segmentation file which should contain only the flat segments
  """
  
  #reading pitch file
  timePitch = np.loadtxt(pitchFile)
  
  #converting Hz to Cents
  timePitch[:,1] = 1200*np.log2((timePitch[:,1]+eps)/55.0)
  hopSize = timePitch[1,0]-timePitch[0,0]
  
  msObj = seg.melodySegmentation()
  segments = msObj.segmentPitchPLS(timePitch[:,1], hopSize, maxAbsError)
  
  fname, ext = os.path.splitext(pitchFile)
  np.savetxt(fname + segFileExt, segments)
  
#2) Using the methods used in Nyas identification - different parameters for getting a good segmentation

def flatNoteSegmentationNyas(pitchFile, tonicExt = '.tonic', segFileExt = '.segmentsNyas', vicinityThsld=30, varWinLen=100.0, varThsld=100.0, timeAwayThsld=100.0, min_nyas_duration=150.0):
  """
  This function performs flat note segmentation using heuristic based algorithm used in Nyas identification work.
  Input:
    pitchFile = csv file of the pitch values <first column time stamps> <second pitch values in Hz>
  Output:
   this function writes a segmentation file which should contain only the flat segments
  """
  fname, ext = os.path.splitext(pitchFile)
  
  #reading pitch file
  timePitch = np.loadtxt(pitchFile)
  
  tonic = float(np.loadtxt(fname + tonicExt))
  
  #converting Hz to Cents
  hopSize = timePitch[1,0]-timePitch[0,0]
  
  msObj = seg.melodySegmentation()
  # this is a new implementation of segmenting flat regions with Nyas and it has improved code!! check it out!
  flatSegments = msObj.segmentPitchNyas(timePitch[:,1], tonic, hopSize, vicinityThsld = vicinityThsld, varWinLen=varWinLen, varThsld=varThsld, timeAwayThsld=timeAwayThsld, min_nyas_duration=min_nyas_duration)
  
  np.savetxt(fname + segFileExt, flatSegments*hopSize)

def batchProcessflatNoteSegmentationNyas(root_dir, ext2Proc = '.wav', pitchExt = '.pitch', tonicExt = '.tonic', segExt = '.flatSegNyas', vicinityThsld=30, varWinLen=100.0, varThsld=100.0, timeAwayThsld=100.0, min_nyas_duration=150.0):  
  
  filenames = BP.GetFileNamesInDir(root_dir, ext2Proc)
  for ii, filename in enumerate(filenames):
    print "processing file %d : %s "%(ii+1, filename)
    fname, ext = os.path.splitext(filename)
    flatNoteSegmentationNyas(fname + pitchExt, tonicExt = tonicExt, segFileExt = segExt, vicinityThsld=vicinityThsld, varWinLen=varWinLen, varThsld=varThsld, timeAwayThsld=timeAwayThsld, min_nyas_duration=min_nyas_duration)

def flatNoteSegmentationVariance(pitchFile, tonicExt = '.tonic', segFileExt = '.segmentsVar'):
  """
  This function performs flat note segmentation using just the pitch variance
  Input:
    pitchFile = csv file of the pitch values <first column time stamps> <second pitch values in Hz>
  Output:
   this function writes a segmentation file which should contain only the flat segments

  With this function I was trying to figure out optimal variance length and variance threshold. I seems like 100 ms duration and 75 cents variance is a decent choice.
  """
  fname, ext = os.path.splitext(pitchFile)
  
  #reading pitch file
  timePitch = np.loadtxt(pitchFile)
  
  tonic = float(np.loadtxt(fname + tonicExt))
  
  #converting Hz to Cents
  timePitch[:,1] = 1200*np.log2((timePitch[:,1]+eps)/tonic)
  hopSize = timePitch[1,0]-timePitch[0,0]

  
  msObj = seg.melodySegmentation()
  flatSegments = msObj.flatSegmentsPitchVariance(timePitch[:,1], hopSize, varWinLen = 200, varThsld = 75)*hopSize
  np.savetxt(fname + segFileExt, flatSegments)  


#3) Very simple method based on searching for stable regions around mean swar values and then joining very closely spaced points.
def flatNoteSegmentationCloseVicinity(pitchFile, tonicExt = '.tonic', segFileExt = '.segmentsVicinity'):
  """
  This function performs flat note segmentation based on the vicinity around the swar locations.
  Input:
    pitchFile = csv file of the pitch values <first column time stamps> <second pitch values in Hz>
  Output:
   this function writes a segmentation file which should contain only the flat segments
  """

  fname, ext = os.path.splitext(pitchFile)
  
  #reading pitch file
  timePitch = np.loadtxt(pitchFile)
  
  tonic = float(np.loadtxt(fname + tonicExt))
  
  #converting Hz to Cents
  hopSize = timePitch[1,0]-timePitch[0,0]

  
  msObj = seg.melodySegmentation()
  flatSegments = msObj.flatSegmentsSwarVicinity(timePitch[:,1], tonic, hopSize, vicinityThsld = 30)*hopSize
  np.savetxt(fname + segFileExt, flatSegments) 

def probableSegmentationPOints(pitchFile, tonicExt = '.tonic', segFileExt = '.segmentsCandidates'):
  """
  """

  fname, ext = os.path.splitext(pitchFile)
  
  #reading pitch file
  timePitch = np.loadtxt(pitchFile)
  
  tonic = float(np.loadtxt(fname + tonicExt))
  
  #converting Hz to Cents
  hopSize = timePitch[1,0]-timePitch[0,0]

  
  msObj = seg.melodySegmentation()
  candSegs = msObj.estimateProbableSegmentationPoints(timePitch[:,1], tonic, hopSize, vicinityThsld = 30, varWinLen=100, varThsld=100)
  np.savetxt(fname + segFileExt, candSegs*hopSize)   
  #np.savetxt(fname + '.segVar', segVar*hopSize)    
  #np.savetxt(fname + '.segVic', segVic*hopSize)    
  #np.savetxt(fname + '.segSlope', segSlope*hopSize)    

#4) Using sliding window local variance based method which was used for the segment filtering step in motif paper

#5) Use microsoft types method but it needs training and labelled data!!!

#5) Using above mentioned methods but using a low pass or median  filtered pitch contour.