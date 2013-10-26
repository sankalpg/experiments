
import numpy as np
import sys,os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/MelodySegmentation/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/Batch_Processing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/TextGrid_Parsing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/mlWrapper/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/TimeSeriesAnalysis/classification/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_Python/TimeSeriesAnalysis/similarity/'))

import textgrid as tgp
import Batch_Proc_Essentia as BP
import timeSeriesClassification as tsc
import MelodySegmentation as MS
import timeSeriesSimilarity as tss
import mlWrapper as mlw

nyasAnnotationFileSuffix = ".nyas"

class nyasAnnotations():
    """
    This class assists in the process of obtaining nyas annotations. The idea is to help the annotator by already providing him a textgrid file with segmented flat regions. So that the
    just have to matk whether a section is nyas or not. Also if a nyas swar is divided into two segments the space between these two segments are marked as "c" by the annotator to say join
    this segment for obtaining nyas annotations. Also the annotator validated the segment boundaries.
    """

    seedFileExt = ".wav"
    tonicFileExt = ".tonic"
    pitchFileExt = ".essentia.pitch"
    segmentTGFileSuffix = ".NyasCand.textgrid"
    annotatedFileSuffix = ".NyasAnnotation.TextGrid"
    nyasAnnotationFileSuffix = nyasAnnotationFileSuffix


    def __init__(self, root_dir):
        self.root_dir  = root_dir
        pass



    def generateTG2Annotate(self):
        """
        This function generates the textgrid file containing segments corresponding to flat regions in pitch. These are kind of nyas candidates and the annotator has to mark which of these segments
        are nyas swars
        """
        filenames = BP.GetFileNamesInDir(self.root_dir,filter=self.seedFileExt)

        for filename in filenames:
            print "Generating segmentation textgrid for file %s"%filename
            file, ext = os.path.splitext(filename)
            nobj = MS.PitchProcessing(pitchfile = file+ self.pitchFileExt, tonicfile=file+ self.tonicFileExt)
            nobj.ComputeNyasCandidates()
            nobj.FilterNyasCandidates()
            nobj.DumpNyasInfo(filename=file+self.segmentTGFileSuffix)


    def extractNyasAnnotations(self):
        """
        This function takes the annotation file and combine the "c" segments into neighboring nyas segments and generate another text file with just nyas segments and nothing else.
        """

        filenames = BP.GetFileNamesInDir(self.root_dir,filter=self.seedFileExt)

        for filename in filenames:
            print "Extract only the Nyas regions from annotated textgrid and making another textgrid %s"%filename + self.nyasAnnotationFileSuffix
            file, ext = os.path.splitext(filename)
            par_obj = tgp.TextGrid.load(file+self.annotatedFileSuffix)	#loading the object
            tiers= tgp.TextGrid._find_tiers(par_obj)	#finding existing tiers

            #reading the segments from the annotation file
            nyasSegments = []
            for tier in tiers:
                tier_details = tier.make_simple_transcript()
                for line in tier_details:

                    if line[2].find('-y')!=-1:
                        nyasSegments.append([float(line[0]), float(line[1]),'nyas'])
                    elif line[2] == "c":
                        nyasSegments.append([float(line[0]), float(line[1]),'c'])

            #combining the "c" labelled segments to their neighboring nyas segments
            popList = []
            for i,segment in enumerate(nyasSegments):
                if segment[2] =="c":
                    cAlreadyCombined=0
                    try:
                        if abs(nyasSegments[i-1][1]-segment[0])<.02 and abs(nyasSegments[i+1][0]-segment[1])<.02:     #if the diff in time is less than 20 ms which it will, just for a safety purpose
                            nyasSegments[i-1][1] = nyasSegments[i+1][1]
                            popList.append(i)
                            popList.append(i+1)
                            cAlreadyCombined=1

                    except:
                            print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]

                    try:
                        if cAlreadyCombined ==0 and abs(nyasSegments[i-1][1]-segment[0])<.02:     #if the diff in time is less than 20 ms which it will, just for a safety purpose
                            nyasSegments[i-1][1] = segment[1]
                            popList.append(i)
                            cAlreadyCombined=1

                    except:
                            print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]

                    try:
                        if cAlreadyCombined ==0 and abs(nyasSegments[i+1][0]-segment[1])<.02:    #if the diff in time is less than 20 ms which it will, just for a safety purpose
                            nyasSegments[i+1][0] = segment[0]
                            popList.append(i)
                            cAlreadyCombined=1
                    except:
                            print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]

            #removing the combined segments
            for i in reversed(popList):
                nyasSegments.pop(i)

            fid = open(file+self.nyasAnnotationFileSuffix,'w')

            for segment in nyasSegments:
                fid.write("%f\t%f\t%s\n"%(segment[0],segment[1],segment[2]))

            fid.close()


class nyasIdentification1():

    seedFileExt = ".wav"
    tonicFileExt = ".tonic"
    pitchFileExt = ".essentia.pitch"
    nyasAnnotationFileSuffix = nyasAnnotationFileSuffix


    def __init__(self, root_dir):
        self.root_dir = root_dir

    def pitchSegmentation(self, root_dir, method, segmentFileExt = ""):

        if isinstance(root_dir,list):
            filenames = root_dir
        else:
            filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)


        for filename in filenames:

            file, ext = os.path.splitext(filename)

            #initializing an object for pitch proessing
            pPbj = MS.PitchProcessing(pitchfile = file + self.pitchFileExt, tonicfile = file + self.tonicFileExt)
            pPbj.PitchHz2Cents()

            if method =="own":
                segmentsAll, segmentsNyas = pPbj.segmentPitch()
            elif method =="keogh":
                segmentsAll = pPbj.segmentPitchKeogh(75)
            else:
                print "Please specify a valid method name"

            #generating segmentation file
            np.savetxt(file + segmentFileExt, segmentsAll,delimiter='\t')

    def extractFeatures(self, root_dir, segmentFileExt, featureFileName):

        if isinstance(root_dir,list):
            filenames = root_dir
        else:
            filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)

        aggLabels = []
        aggFeatures = []

        for filename in filenames:
            print "processing file %s "%filename
            file, ext = os.path.splitext(filename)

            #initializing an object for pitch proessing
            nObj = MS.NyasProcessing()

            segments, labels, features = nObj.NyasFeatureExtraction(file + self.pitchFileExt, file + self.tonicFileExt, file + segmentFileExt, file+ self.nyasAnnotationFileSuffix)
            for i,label in enumerate(labels):
                features[i]['type']=label
            aggLabels.extend(labels)
            aggFeatures.extend(features)

        nAggObj = MS.NyasAggFeatureProcessing()
        nAggObj.GenerateNyasARFF(featureFileName, aggFeatures)


    def evalClassifiers(self):

        pass


class nyasIdentification2():

    seedFileExt = ".wav"
    tonicFileExt = ".tonic"
    pitchFileExt = ".essentia.pitch"
    nyasAnnotationFileSuffix = nyasAnnotationFileSuffix

    def __init__(self, root_dir):
        pass

    def segmentation(self):
        pass

    def extractFeatures(self, root_dir, segmentFileExt):

        if isinstance(root_dir,list):
            filenames = root_dir
        else:
            filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)

        #initializing the nyas processing class object
        nyasproc = MS.NyasProcessing()

        pitchArray=np.array([])
        segmentsArray = np.array([])
        labelsArray = np.array([])

        for i,filename in enumerate(filenames):
            print "processing file %s "%filename
            file, ext = os.path.splitext(filename)

            ph_obj = MS.PitchProcessing(pitchfile = file + '.essentia.pitch', tonicfile = file +'.tonic')
            ph_obj.PitchHz2Cents()

            pitchSegments = np.loadtxt(file + segmentFileExt)

            #Since all the segments are read, remove the trivial once which have mainly silence in them.
            pitchSegments = nyasproc.removeSegmentsWithSilence(ph_obj.timepitch, ph_obj.phop,pitchSegments)


            labels = np.array(nyasproc.obtainClassLabels(pitchSegments,file+nyasAnnotationFileSuffix, ph_obj.phop ,ph_obj.pCents.shape[0]))

            pitchSegments = pitchSegments/ph_obj.phop

            if i==0:
                pitchArray = ph_obj.pCents
                segmentsArray = pitchSegments
                labelsArray = labels
            else:
                sample_off = pitchArray.shape[0]
                segments = pitchSegments + sample_off
                pitchArray = np.append(pitchArray, ph_obj.pCents,axis=0)
                segmentsArray = np.append(segmentsArray,segments,axis=0)
                labelsArray = np.append(labelsArray, labels, axis=0)

        tsc_obj = tsc.tsClassification()
        accuracy, decArray, classification_output = tsc_obj.classificationDTWkNN(pitchArray,segmentsArray, labelsArray)

        decArray = np.array(decArray)
        ind_nyas = np.where(labels == 'nyas')[0]
        nyasAccuracy = sum(decArray[ind_nyas])/float(len(ind_nyas))

        return accuracy, nyasAccuracy, classification_output


    def evalKnnDtw(self):

        pass



class nyasIdentification3():

    seedFileExt = ".wav"
    tonicFileExt = ".tonic"
    pitchFileExt = ".essentia.pitch"
    nyasAnnotationFileSuffix = nyasAnnotationFileSuffix

    def __init__(self, root_dir):
        pass

    def segmentation(self):
        pass

    def extractFeatures(self, root_dir, segmentFileExt, featureType):

        if isinstance(root_dir,list):
            filenames = root_dir
        else:
            filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)

        #initializing the nyas processing class object
        nyasproc = MS.NyasProcessing()

        pitchArray=np.array([])
        segmentsArray = np.array([])
        labelsArray = np.array([])

        for i,filename in enumerate(filenames):
            print "processing file %s "%filename
            file, ext = os.path.splitext(filename)

            ph_obj = MS.PitchProcessing(pitchfile = file + '.essentia.pitch', tonicfile = file +'.tonic')
            ph_obj.PitchHz2Cents()

            pitchSegments = np.loadtxt(file + segmentFileExt)

            #Since all the segments are read, remove the trivial once which have mainly silence in them.
            pitchSegments = nyasproc.removeSegmentsWithSilence(ph_obj.timepitch, ph_obj.phop,pitchSegments)


            labels = np.array(nyasproc.obtainClassLabels(pitchSegments,file+nyasAnnotationFileSuffix, ph_obj.phop ,ph_obj.pCents.shape[0]))

            pitchSegments = pitchSegments/ph_obj.phop

            if i==0:
                pitchArray = ph_obj.pCents
                segmentsArray = pitchSegments
                labelsArray = labels
            else:
                sample_off = pitchArray.shape[0]
                segments = pitchSegments + sample_off
                pitchArray = np.append(pitchArray, ph_obj.pCents,axis=0)
                segmentsArray = np.append(segmentsArray,segments,axis=0)
                labelsArray = np.append(labelsArray, labels, axis=0)

        matchMTX = tss.computeMatchMatrix(pitchArray, segmentsArray, pitchArray, segmentsArray, featureType)

        #filling the other half of matrix
        for i in xrange(matchMTX.shape[0]):
            for j in range(i):
                matchMTX[i,j]=matchMTX[j,i]
        np.save('matchMTX' + featureType+segmentFileExt, matchMTX)
        #matchMTX = np.load('matchMTX_all.npy')
        accuracy, decArray = self.evalKnnDtw(matchMTX, labelsArray)

        return accuracy


    def evalKnnDtw(self, matrix, labels):

        mlObj = mlw.experimenter()

        mlObj.setFeaturesAndClassLabels(matrix,labels)

        mlObj.setExperimentParams(nExp = 10, typeEval = ("kFoldCrossVal",-1), nInstPerClass = -1, classifier = ('mYkNN',"default"))

        mlObj.runExperiment()

        return mlObj.overallAccuracy, mlObj.decArray

class classificationExperiment():

    expParam = {}

    def __init__(self, nExp = 10, typeEval = ("kFoldCrossVal",10)):

        self.expParam['nExp'] = nExp
        self.expParam['typeEval'] = typeEval

    def featureSelectionManual(self):

        arffFile = 'KeoghSegAllFeat.arff'
        result_dir = 'context'
        logfile = 'context.txt'

        classifierSet = [('svm','default'), ('tree','default'),('kNN','default'),('nbMulti','default'),('logReg','default'),('randC','default')]

        local = ['mean','varPeakDist', 'variance', 'meanPeakDist', 'meanPeakAmp', 'varPeakAmp','tCentroid', 'length', 'isflat']
        context= ['post_sil_dur', 'rel_len_longest', 'rel_len_pre_segment', 'rel_len_post_segment', 'rel_len_BP', 'pre_sil_dur', 'prev_variance', 'prev_mean', 'prev_tCentroid', 'prev_meanPeakDist', 'prev_varPeakDist', 'prev_meanPeakAmp', 'prev_varPeakAmp', 'prev_length', 'prev_isflat']

        try_local = [local[:i] for i in range(1,len(local)+1)]
        try_context = [context[:i] for i in range(1,len(context)+1)]
        local_context = local+ context
        try_local_context = [local_context[:i] for i in range(1,len(local_context)+1)]
        mlObj = mlw.advanceExperimenter(arffFile=arffFile)
        mlObj.runCompoundExperiment([local_context],classifierSet,self.expParam, result_dir, logfile)


