import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/MelodySegmentation/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/TimeSeriesAnalysis/classification/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/Batch_Processing/'))

import Batch_Proc_Essentia as BP
import MelodySegmentation as ms
import timeSeriesClassification as tsc
import numpy as np

def classificationDTWkNN(root_dir):

    if type(root_dir) == list:
            filenames = root_dir
    else:
            filenames = BP.GetFileNamesInDir(root_dir,filter=".wav")

    nyasproc = ms.NyasProcessing()

    pitchArray=np.array([])
    segmentsArray = np.array([])
    labelsArray = np.array([])

    for i,filename in enumerate(filenames):

        file, ext = os.path.splitext(filename)
        nyasproc.ReadNyasAnnotationsAsList(file + ".NyasAnnotation.TextGrid")

        ph_obj = ms.PitchProcessing(pitchfile = file + '.essentia.pitch', tonicfile = file +'.tonic')
        ph_obj.PitchHz2Cents()
        nyasproc.removeTrivialSegmentsFromList(ph_obj.timepitch, ph_obj.phop)

        annotations = np.array(nyasproc.getAnnotations())
        segments = annotations[:,:2].astype(np.float)
        labels = annotations[:,2]
        segments= (segments/ph_obj.phop).astype(np.int)

        if i==0:
            pitchArray = ph_obj.pCents
            segmentsArray = segments
            labelsArray = labels
        else:
            time_off = ph_obj.pCents.shape[0]
            segments = segments + time_off
            pitchArray = np.append(pitchArray, ph_obj.pCents,axis=0)
            segmentsArray = np.append(segmentsArray,segments,axis=0)
            labelsArray = np.append(labelsArray, labels, axis=0)


    tsc_obj = tsc.tsClassification()
    accuracy, decArray, classification_output = tsc_obj.classificationDTWkNN(pitchArray,segmentsArray, labelsArray)

    decArray = np.array(decArray)
    ind_nyas = np.where(labels == 'nyas')[0]
    nyasAccuracy = sum(decArray[ind_nyas])/float(len(ind_nyas))

    return accuracy, nyasAccuracy, classification_output




