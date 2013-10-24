
import numpy as np
import sys,os, json, pickle

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


class nyasIdentification():

    seedFileExt = ".wav"
    tonicFileExt = ".tonic"
    pitchFileExt = ".essentia.pitch"
    nyasAnnotationFileSuffix = ".nyas"



    def __init__(self, root_dir):
        self.root_dir = root_dir

    def pitchSegmentation(self, root_dir, method, segmentFileExt = "", max_error=-1):
        """
        This function generates segments of pitch sequence using either our method or piecewise linear segmentation method
        max_error is only used for PLS
        """

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
                if max_error ==-1:
                    print "Please provide max_error for PLS case"
                    return False
                segmentsAll = pPbj.segmentPitchKeogh(max_error)
            else:
                print "Please specify a valid method name"

            #generating segmentation file
            np.savetxt(file + segmentFileExt, segmentsAll,delimiter='\t')


    def extractFeatures(self, root_dir, segmentFileExt, featureFileName, foldInformationFileName, category = 'features'):
        """
        This function generates with features for each segment or computes DTW distances between segments. You have to give informaiton regarding which segmentation extension (which eventually is the segmentation logic) to use.
        And what should be the name of the output file and what is the name of the file in which you will store the information regarding indices of each fold, i.e. indices for training and testing of each fold.
        This is decided based on equal dividsion of ground truth nyas segments and such that there is no data deom testing set during training. This is really important to understand.
        """

        #obtaining all the names of the files in the root directory for processing. Based on a seeffile extension
        filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)

        #Find out what is the minimum number of nyas annotations per file. This will decide how big should be the folds in terms of number of instances.
        #NOTE that fold splitting should be done based on ground truth nyas labels and not by segments obtained after segmentation
        min_Num_Nyas=1000000000000000
        file_min_NyasSeg = ""
        total_nyas_segments = 0
        for filename in filenames:
            fname, ext = os.path.splitext(filename)
            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            total_nyas_segments+=nyasSeg.shape[0]
            if min_Num_Nyas>nyasSeg.shape[0]:
                min_Num_Nyas = nyasSeg.shape[0]
                file_min_NyasSeg = nfile
        print "Minimum number of nyas segments are %d for file %s :"%(min_Num_Nyas, file_min_NyasSeg)
        print "Total number of nyas segments are %d"%total_nyas_segments

        #now take each fold so that it has minimum number of nyas segments (computed above)
        aggLabels = []
        aggFeatures = []
        foldInfo={}
        segment_index_pointer=0
        for filename in filenames:
            foldInfo[filename]=[]

            fname, ext = os.path.splitext(filename)

            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            nyasSeg = nyasSeg[:,:2]
            nyasSeg = nyasSeg.astype(np.float)

            #initializing an object for pitch proessing
            nyasObj = MS.NyasProcessing()
            segments, labels, features = nyasObj.NyasFeatureExtraction(fname + self.pitchFileExt, fname + self.tonicFileExt, fname + segmentFileExt, fname+ self.nyasAnnotationFileSuffix)
            for i,label in enumerate(labels):
                features[i]['type']=label
            aggLabels.extend(labels)
            aggFeatures.extend(features)


            n_folds_current = np.floor((nyasSeg.shape[0]/min_Num_Nyas)).astype(np.int16)
            remainder = nyasSeg.shape[0] - n_folds_current*min_Num_Nyas
            str_fold=0
            for i in range(n_folds_current):
                ind_fold_more = np.where(segments[:,0]>=str_fold)[0]
                if not (i == n_folds_current-1):
                    ind_fold_less = np.where(segments[:,0]<nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1])[0]
                    ind_fold = np.array(list(set(list(ind_fold_more)) & set(list(ind_fold_less))))
                else:
                    ind_fold = ind_fold_more

                foldInfo[filename].append([segments[ind_fold,:].tolist(), (ind_fold+segment_index_pointer).tolist()])

                str_fold = nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1]

            segment_index_pointer+=segments.shape[0]

        nAggObj = MS.NyasAggFeatureProcessing()
        nAggObj.GenerateNyasARFF(featureFileName, aggFeatures)

        fid = open(foldInformationFileName,'w')
        json.dump(foldInfo,fid, indent=4)
        fid.close()


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
                time_off = ph_obj.pCents.shape[0]
                segments = pitchSegments + time_off
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
                time_off = ph_obj.pCents.shape[0]
                segments = pitchSegments + time_off
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


