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

    for filename in filenames:

        file, ext = os.path.splitext(filename)
        nyasproc.ReadNyasAnnotationsAsList(file + ".NyasAnnotation.TextGrid")

        ph_obj = ms.PitchProcessing(pitchfile = file + '.essentia.pitch', tonicfile = file +'.tonic')
        ph_obj.PitchHz2Cents()
        nyasproc.removeTrivialSegmentsFromList(ph_obj.timepitch, ph_obj.phop)

        annotations = np.array(nyasproc.getAnnotations())
        segments = annotations[:,:2].astype(np.float)
        labels = annotations[:,2]
        segments= (segments/ph_obj.phop).astype(np.int)

        tsc_obj = tsc.tsClassification()
        accuracy, confMTX, classification_output = tsc_obj.classificationDTWkNN(ph_obj.pCents,segments, labels)






