import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/melodyProcessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/batchProcessing/'))

import segmentation as seg
import pitchHistogram as PH
import batchProcessing as BP
import copy


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
  np.savetxt(fname + segFileExt, segments, delimiter = "\t", fmt = '%.5f')
  
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
  
  np.savetxt(fname + segFileExt, flatSegments*hopSize, delimiter = "\t", fmt = '%.5f')

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
  np.savetxt(fname + segFileExt, flatSegments, delimiter = "\t", fmt = '%.5f')


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
  np.savetxt(fname + segFileExt, flatSegments, delimiter = "\t", fmt = '%.5f')

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
  np.savetxt(fname + segFileExt, candSegs*hopSize, delimiter = "\t", fmt = '%.5f')
  #np.savetxt(fname + '.segVar', segVar*hopSize)    
  #np.savetxt(fname + '.segVic', segVic*hopSize)    
  #np.savetxt(fname + '.segSlope', segSlope*hopSize)    

#4) Using sliding window local variance based method which was used for the segment filtering step in motif paper

#5) Use microsoft types method but it needs training and labelled data!!!

class flatNoteCompression():


  def __init__(self, pitch, tonic, flatSegments, hopSize, saturateLen=0.3):
    
    "Initializing the object and computing pitch histogram..."
    phObj = PH.PitchHistogram(pitch, tonic)
    #getting valid swar locations (within one octave)
    phObj.ValidSwarLocEstimation(Oct_fold=1)
    #converting swar locations in indexes to cents
    phObj.SwarLoc2Cents()
    #since the swar locations are computed using octave folded histogram, propagating them to other octaves
    phObj.ExtendSwarOctaves()
    self.swarCents = phObj.swarCents
    self.pitch = pitch
    self.pCents = phObj.pCents
    self.phop = hopSize
    self.tonic = tonic
    self.maxFlatLen = np.round(saturateLen/hopSize).astype(np.int)
    
    if flatSegments.size == len(flatSegments): #sometimes if ther is only one segment in a file, np.loadtxt reads 1 d array
      flatSegments = np.array([flatSegments])

    self.flatSegs = flatSegments

    self.flatArrayFlag = np.zeros(self.pCents.size)
    for segs in self.flatSegs:
      startInd = np.round(segs[0]/hopSize).astype(np.int)
      endInd = np.round(segs[1]/hopSize).astype(np.int)
      self.flatArrayFlag[startInd:endInd+1] = 1
    
    self.flatInds = np.where(self.flatArrayFlag==1)[0]

  def compress(self, phraseStartInd, phraseLen, segOutLen):
    """
    This function compresses flat notes which are longer in duration than segOutLen and make them equal to segOutLen. Also the pitch sequence is substituted with the note frequency computed automatically using pitch histograms.
    """

    #differentiate between phrase length and segOutLen. SegOutLen is needed becasue output segments dumped are all of constant length and before compressino we dont know how much a segment will compress
    segmentLen = 4*segOutLen
    pitchSegCandidate = copy.deepcopy(self.pitch[phraseStartInd:phraseStartInd+segmentLen])
    #find flatnotes in the given segment
    indSegments = np.arange(phraseStartInd, phraseStartInd+segmentLen)

    indFlatRegions = np.intersect1d(self.flatInds, indSegments)
    if len(indFlatRegions)==0:
      return pitchSegCandidate[:segOutLen], phraseLen
    indFlatSegments = seg.groupIndices(indFlatRegions)
    lenSegments = indFlatSegments[:,1] - indFlatSegments[:,0]
    indLongNotes = np.where(lenSegments>=self.maxFlatLen)[0]
    
    if len(indLongNotes)==0:
      return pitchSegCandidate[:segOutLen], phraseLen
    else:
      delInds = []
      for indLongNote in indLongNotes:
        segStart = indFlatSegments[indLongNote,0]
        segEnd = indFlatSegments[indLongNote,1]
        segPitch = self.pCents[segStart:segEnd+1]
        segPitch = np.sort(segPitch)
        meanSegment = np.mean(segPitch[np.round(len(segPitch)*0.1).astype(np.int):np.round(len(segPitch)*0.9).astype(np.int)])
        indMin = np.argmin(abs(self.swarCents-meanSegment))
        pVal = self.tonic*np.power(2,self.swarCents[indMin]/1200)
        pitchSegCandidate[segStart-phraseStartInd:segEnd-phraseStartInd]=pVal
        delInds.extend(np.arange(segStart+self.maxFlatLen-phraseStartInd, segEnd-phraseStartInd+1))
      delInds = np.array(delInds)
      reductionLen = len(np.where(delInds<phraseLen)[0])
      pitchSegCandidate = np.delete(pitchSegCandidate, delInds)
      return pitchSegCandidate[:segOutLen], phraseLen-reductionLen

  def supressOrnamentation(self, phraseStartInd, segOutLen):
    """
    This function substitute a flat note (musical flat note which might have many ornamentations) with a constant pitch value that corresponds to the note that was sung
    """

    #differentiate between phrase length and segOutLen. SegOutLen is needed becasue output segments dumped are all of constant length and before compressino we dont know how much a segment will compress
    segmentLen = 4*segOutLen
    pitchSegCandidate = copy.deepcopy(self.pitch[phraseStartInd:phraseStartInd+segmentLen])
    
    #find flatnotes in the given segment
    indSegments = np.arange(phraseStartInd, phraseStartInd+segmentLen)

    indFlatRegions = np.intersect1d(self.flatInds, indSegments)
    if len(indFlatRegions)==0:
      return pitchSegCandidate[:segOutLen]
    indFlatSegments = seg.groupIndices(indFlatRegions)
    lenSegments = indFlatSegments[:,1] - indFlatSegments[:,0]
    indLongNotes = np.where(lenSegments>=self.maxFlatLen)[0]
    
    if len(indLongNotes)==0:
      return pitchSegCandidate[:segOutLen]
    else:
      #delInds = []
      for indLongNote in indLongNotes:
        segStart = indFlatSegments[indLongNote,0]
        segEnd = indFlatSegments[indLongNote,1]
        segPitch = self.pCents[segStart:segEnd+1]
        segPitch = np.sort(segPitch)
        meanSegment = np.mean(segPitch[np.round(len(segPitch)*0.1).astype(np.int):np.round(len(segPitch)*0.9).astype(np.int)])
        indMin = np.argmin(abs(self.swarCents-meanSegment))
        pVal = self.tonic*np.power(2,self.swarCents[indMin]/1200)
        pitchSegCandidate[segStart-phraseStartInd:segEnd-phraseStartInd]=pVal
        #delInds.extend(range(segStart+self.maxFlatLen-phraseStartInd, segEnd-phraseStartInd+1))
      #delInds = np.array(delInds)
      #reductionLen = len(np.where(delInds<phraseLen)[0])
      #pitchSegCandidate = np.delete(pitchSegCandidate, delInds)
      return pitchSegCandidate[:segOutLen]






