import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../melodicSimilarity/flatNoteCompress/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/batchProcessing/'))

import compressFlatNotes as cmp
import batchProcessing as BP

def performFlatNoteCompression(pitchFile, tonicFile, flatNoteFile, outFile, saturationLen):
    """
    This function compresses/truncates stable note regions in the melody
    flatNoteFile contains this flat note segmentation
    """
    fname, ext = os.path.splitext(pitchFile)
    
    pitchTime = np.loadtxt(pitchFile)
    tonic = np.loadtxt(tonicFile)
    flats = np.loadtxt(flatNoteFile)
    hopPitch = (pitchTime[-1,0]-pitchTime[0,0])/float(pitchTime.shape[0]-1)
    print "HopSize is %f"%hopPitch
    
    objComp = cmp.flatNoteCompression(pitchTime[:,1], float(tonic), flats, hopPitch, saturateLen = saturationLen)
    
    timePitchCompress = objComp.compressEntirePitchTrack()
    
    np.savetxt(outFile, timePitchCompress)


def batchProcessFlatNoteCompression(root_dir, searchExt = '.mp3', pitchExt = '.pitch', tonicExt = '.tonic', flatnoteExt = '.flatNyas', compressExt = '.pitchFlatCompress', saturationLen = 0.3):
    """
    This is a batch processing function for performing flat note compression
    """
    
    filenames = BP.GetFileNamesInDir(root_dir, searchExt)
    
    for filename in filenames:
        fname, ext = os.path.splitext(filename)
        performFlatNoteCompression(fname + pitchExt, fname + tonicExt, fname + flatnoteExt, fname + compressExt, saturationLen)
    
    return True