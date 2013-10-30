
import numpy as np
import sys,os, json, pickle, shutil

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
            thereisC=1
            while thereisC ==1:
                thereisC=0
                for i,segment in enumerate(nyasSegments):
                    if segment[2] =="c":
                        try:
                            if abs(nyasSegments[i-1][1]-segment[0])<.02 and abs(nyasSegments[i+1][0]-segment[1])<.02:     #if the diff in time is less than 20 ms which it will, just for a safety purpose
                                nyasSegments[i-1][1] = nyasSegments[i+1][1]
                                nyasSegments.pop(i+1)
                                nyasSegments.pop(i)
                                thereisC=1
                                break
                        except:
                                print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]

                        try:
                            if abs(nyasSegments[i-1][1]-segment[0])<.02:     #if the diff in time is less than 20 ms which it will, just for a safety purpose
                                nyasSegments[i-1][1] = segment[1]
                                nyasSegments.pop(i)
                                thereisC=1
                                break
                        except:
                                print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]

                        try:
                            if abs(nyasSegments[i+1][0]-segment[1])<.02:     #if the diff in time is less than 20 ms which it will, just for a safety purpose
                                nyasSegments[i+1][0] = segment[0]
                                nyasSegments.pop(i)
                                thereisC=1
                                break
                        except:
                                print "This is a rare scenario, check the cause. Details are: C segment starting %f"%segment[0]




            """popList = []
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
            """

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


    def extractFeatures(self, root_dir, segmentFileExt, featureFileName, foldInformationFileName, segMthd = 'own'):
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
            foldInfo[filename]={}

            fname, ext = os.path.splitext(filename)

            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            nyasSeg = nyasSeg[:,:2]
            nyasSeg = nyasSeg.astype(np.float)

            #because in the annotations there were two tired for annotation, all the nyas segments in thsi files might not be in order
            #sorting nyas annotations
            sort_arg = np.argsort(nyasSeg[:,0])
            nyasSeg = nyasSeg[sort_arg,:]

            #initializing an object for pitch proessing
            nyasObj = MS.NyasProcessing()
            segments, labels, features = nyasObj.NyasFeatureExtraction(fname + self.pitchFileExt, fname + self.tonicFileExt, fname + segmentFileExt, fname+ self.nyasAnnotationFileSuffix,segMthod = segMthd)
            for i,label in enumerate(labels):
                features[i]['type']=label
            aggLabels.extend(labels)
            aggFeatures.extend(features)


            n_folds_current = np.floor((nyasSeg.shape[0]/min_Num_Nyas)).astype(np.int16)
            remainder = nyasSeg.shape[0] - n_folds_current*min_Num_Nyas
            str_fold=0
            foldInfo[filename]['guessSeg']=[]
            foldInfo[filename]['trueSeg']=[]
            for i in range(n_folds_current):
                ind_fold_more = np.where(segments[:,0]>=str_fold)[0]
                if not (i == n_folds_current-1):
                    ind_fold_less = np.where(segments[:,0]<nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1])[0]
                    ind_fold = np.array(list(set(list(ind_fold_more)) & set(list(ind_fold_less))))
                    foldInfo[filename]['trueSeg'].append(nyasSeg[i*min_Num_Nyas:i*min_Num_Nyas+min_Num_Nyas,:].tolist())

                else:
                    ind_fold = ind_fold_more
                    foldInfo[filename]['trueSeg'].append(nyasSeg[i*min_Num_Nyas:,:].tolist())

                if ind_fold.shape[0] == 0:
                    print "Something is terribly wrong"

                foldInfo[filename]['guessSeg'].append([segments[ind_fold,:].tolist(), (ind_fold+segment_index_pointer).tolist()])

                str_fold = nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1]

            segment_index_pointer+=segments.shape[0]

        nAggObj = MS.NyasAggFeatureProcessing()
        nAggObj.GenerateNyasARFF(featureFileName, aggFeatures)

        fid = open(foldInformationFileName,'w')
        json.dump(foldInfo,fid, indent=4)
        fid.close()

    def performTrainTest(self, featureFIle, foldINfoFIle, finalFeatureSet, THRESHOLD_FINE, classifierInfo = ('svm','default'), mergeGuessNyasSegments = 1, DoArtistRagFiltering=1):

        """localFeatures = ['mean','varPeakDist', 'variance', 'meanPeakDist', 'meanPeakAmp', 'varPeakAmp','tCentroid', 'length', 'isflat']
        contextFeatures= ['post_sil_dur', 'rel_len_longest', 'rel_len_pre_segment', 'rel_len_post_segment', 'rel_len_BP', 'pre_sil_dur', 'prev_variance', 'prev_mean', 'prev_tCentroid', 'prev_meanPeakDist', 'prev_varPeakDist', 'prev_meanPeakAmp', 'prev_varPeakAmp', 'prev_length', 'prev_isflat']

        if featureType == 'local':
            finalFeatureSet = localFeatures
        elif featureType == 'context':
            finalFeatureSet = contextFeatures
        elif featureType == 'local_context':
            finalFeatureSet =  local + contextFeatures
        else:
            print "Please select a valid feature set"
            return False
        """

        mlObj = mlw.experimenter()
        mlObj.readArffFile(featureFIle)

        mlObj.setExperimentParams(classifier = classifierInfo)

        #feature selection step
        mlObj.featureSelection(features2Use=finalFeatureSet)

        #feature normalization
        mlObj.normalizeFeatures()

        total_features = mlObj.features.shape[0]
        total_indices = range(total_features)

        foldInfo = json.load(open(foldINfoFIle))

        self.accuracy=[]
        self.stats=[]
        fold_cnt=0

        for key, val in foldInfo.items():
            indicesFile = []
            #calculating indices of the same file
            for singleFileInfo in val['guessSeg']:
                indicesFile.extend(singleFileInfo[1])
            #calculating indices of the same artist and rag
            ind_artist_rag=[]
            if DoArtistRagFiltering:
                ind_artist_rag = self.computeSameArtistRagIndices(foldInfo, key)

            train_indices = list(set(total_indices) - (set(indicesFile) | set(ind_artist_rag)))

            for i,singleFileInfo in enumerate(val['guessSeg']):
                test_ind = singleFileInfo[1]

                prediction = mlObj.performTrainTest(mlObj.featuresSelected[train_indices,:], mlObj.classLabelsInt[train_indices], mlObj.featuresSelected[test_ind,:])

                resultPFold = np.where(prediction==mlObj.classLabelsInt[test_ind])[0]
                accuracy = float(resultPFold.shape[0])/float(len(test_ind))
                self.accuracy.append(accuracy)

                #computing affective boundaries generated by overall classification using class label information
                predictedClasses = np.array(mlObj.cNames)[prediction]
                predicted_nyas_ind = np.where(predictedClasses=='nyas')[0]
                pred_nyas_segments = np.array(val['guessSeg'][i][0])[predicted_nyas_ind].tolist()

                if mergeGuessNyasSegments:
                    #merge nyas segments if they are close
                    mergeFlag=1
                    while mergeFlag ==1:
                        mergeFlag=0
                        for k in range(len(pred_nyas_segments)-1):
                            if abs(pred_nyas_segments[k][1]-pred_nyas_segments[k+1][0])<.02:
                                pred_nyas_segments[k][1] = pred_nyas_segments[k+1][1]
                                pred_nyas_segments.pop(k+1)
                                mergeFlag =1
                                break

                pred_nyas_segments = np.array(pred_nyas_segments)
                #guessboundaries = np.append(pred_nyas_segments[:,0],pred_nyas_segments[:,1])   #will be a problem when its null
                guessboundaries = np.reshape(pred_nyas_segments,pred_nyas_segments.size)
                guessboundaries = list(set(guessboundaries.tolist()))
                guessboundaries = np.sort(guessboundaries).tolist()


                gt_nyas_segments = val['trueSeg'][i]
                gt_nyas_segments = np.array(gt_nyas_segments)
                #trueboundaries = np.append(gt_nyas_segments[:,0],gt_nyas_segments[:,1])
                trueboundaries = np.reshape(gt_nyas_segments,gt_nyas_segments.size)
                trueboundaries = list(set(trueboundaries.tolist()))
                trueboundaries = np.sort(trueboundaries).tolist()

                boundP, boundR, boundF, meangtt, meanttg = self.calculateBoundaryPRF(guessboundaries, trueboundaries, THRESHOLD_FINE)

                overlapP, overlapR, overlapF = self.calculateOverlapPRF(pred_nyas_segments, gt_nyas_segments)

                self.stats.append([ boundP, boundR, boundF, meangtt, meanttg, overlapP, overlapR, overlapF, accuracy])
                fold_cnt+=1

        return self.stats

    def computeSameArtistRagIndices(self, foldInfo, curr_key):

        #open info file
        artistRagInfo = json.load(open('artist_rag_mapping.json'))

        curr_rag = artistRagInfo[curr_key][0]
        curr_artist = artistRagInfo[curr_key][1]

        ind_same_artist_rags=[]
        for key in foldInfo.keys():
            if artistRagInfo[curr_key][0]==artistRagInfo[key][0] or artistRagInfo[curr_key][1]==artistRagInfo[key][1]:
                for folds in foldInfo[key]['guessSeg']:
                    ind_same_artist_rags.extend(folds[1])

        return ind_same_artist_rags


    def DTWkNNClassification(self, matchMTXFile, foldINfoFile, THRESHOLD_FINE, mergeGuessNyasSegments = 1):


        foldInfo = json.load(open(foldINfoFile))

        out = json.load(open(matchMTXFile))
        matchMTX = np.array(out['matchMTX'])
        labels = np.array(out['labels'])

        mlObj = mlw.experimenter()

        mlObj.setExperimentParams(classifier = ('mYkNN','default'))

        mlObj.setFeaturesAndClassLabels(matchMTX, labels)

        #feature selection step
        mlObj.featureSelection(features2Use=-1)


        total_features = mlObj.features.shape[0]
        total_indices = range(total_features)



        self.accuracy=[]
        self.stats=[]
        fold_cnt=0

        for key, val in foldInfo.items():
            indicesFile = []
            for singleFileInfo in val['guessSeg']:
                indicesFile.extend(singleFileInfo[1])

            train_indices = list(set(total_indices) - set(indicesFile))

            for i,singleFileInfo in enumerate(val['guessSeg']):
                test_ind = singleFileInfo[1]

                prediction=np.array([]).astype(np.int16)

                for test_ind_one in test_ind:
                    out = mlObj.performTrainTest(mlObj.featuresSelected[train_indices,:], mlObj.classLabelsInt[train_indices], mlObj.featuresSelected[[test_ind_one],:])
                    prediction = np.append(prediction,out)

                resultPFold = np.where(prediction==mlObj.classLabelsInt[test_ind])[0]
                accuracy = float(resultPFold.shape[0])/float(len(test_ind))
                self.accuracy.append(accuracy)

                #computing affective boundaries generated by overall classification using class label information
                predictedClasses = np.array(mlObj.cNames)[prediction]
                predicted_nyas_ind = np.where(predictedClasses=='nyas')[0]
                pred_nyas_segments = np.array(val['guessSeg'][i][0])[predicted_nyas_ind].tolist()

                if mergeGuessNyasSegments:
                    #merge nyas segments if they are close
                    mergeFlag=1
                    while mergeFlag ==1:
                        mergeFlag=0
                        for k in range(len(pred_nyas_segments)-1):
                            if abs(pred_nyas_segments[k][1]-pred_nyas_segments[k+1][0])<.02:
                                pred_nyas_segments[k][1] = pred_nyas_segments[k+1][1]
                                pred_nyas_segments.pop(k+1)
                                mergeFlag =1
                                break

                pred_nyas_segments = np.array(pred_nyas_segments)
                #guessboundaries = np.append(pred_nyas_segments[:,0],pred_nyas_segments[:,1])   #will be a problem when its null
                guessboundaries = np.reshape(pred_nyas_segments,pred_nyas_segments.size)
                guessboundaries = list(set(guessboundaries.tolist()))
                guessboundaries = np.sort(guessboundaries).tolist()


                gt_nyas_segments = val['trueSeg'][i]
                gt_nyas_segments = np.array(gt_nyas_segments)
                #trueboundaries = np.append(gt_nyas_segments[:,0],gt_nyas_segments[:,1])
                trueboundaries = np.reshape(gt_nyas_segments,gt_nyas_segments.size)
                trueboundaries = list(set(trueboundaries.tolist()))
                trueboundaries = np.sort(trueboundaries).tolist()

                boundP, boundR, boundF, meangtt, meanttg = self.calculateBoundaryPRF(guessboundaries, trueboundaries, THRESHOLD_FINE)

                overlapP, overlapR, overlapF = self.calculateOverlapPRF(pred_nyas_segments, gt_nyas_segments)

                self.stats.append([ boundP, boundR, boundF, meangtt, meanttg, overlapP, overlapR, overlapF, accuracy])
                fold_cnt+=1

        return self.stats

    def randomBaselineExperiments(self, root_dir, THRESHOLD_FINE, baselineMethod=1):


        self.accuracy=[]
        self.stats=[]
        UNIFORM_BOUNDARY_GAP = 0.1 #seconds

        #obtaining all the names of the files in the root directory for processing. Based on a seeffile extension
        filenames = BP.GetFileNamesInDir(root_dir,filter=self.seedFileExt)

        #computing fraction of dutation of songs which has nyas as a tag
        total_dur=0
        nyas_dur =0
        inter_boundary_interval=[]
        for filename in filenames:
            fname, ext = os.path.splitext(filename)

            phObj =  MS.PitchProcessing(pitchfile = fname+ self.pitchFileExt, tonicfile=fname + self.tonicFileExt)
            hopsize = phObj.phop
            time_last = phObj.timepitch.shape[0]*hopsize
            total_dur+=time_last

            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            nyasSeg = nyasSeg[:,:2]
            nyasSeg = nyasSeg.astype(np.float)

            for seg in nyasSeg:
                nyas_dur+=seg[1]-seg[0]


            trueboundaries = np.reshape(nyasSeg,nyasSeg.size)
            trueboundaries = list(set(trueboundaries.tolist()))
            trueboundaries = np.sort(trueboundaries).tolist()

            for i,val in enumerate(trueboundaries[:-1]):
                inter_boundary_interval.append(trueboundaries[i+1]-trueboundaries[i])

        np.random.shuffle(inter_boundary_interval) # this store the inter boundary intervals. We can randomly draw samples from this to obtain inter boundary differences
        nyas_prob = (float(nyas_dur)/float(total_dur))
        print "fraction of duration which is nyas is :%f\n"%nyas_prob


        for filename in filenames:

            fname, ext = os.path.splitext(filename)

            phObj =  MS.PitchProcessing(pitchfile = fname+ self.pitchFileExt, tonicfile=fname + self.tonicFileExt)
            hopsize = phObj.phop

            time_last = phObj.timepitch.shape[0]*hopsize

            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            nyasSeg = nyasSeg[:,:2]
            nyasSeg = nyasSeg.astype(np.float)

            #because in the annotations there were two tired for annotation, all the nyas segments in thsi files might not be in order
            #sorting nyas annotations
            sort_arg = np.argsort(nyasSeg[:,0])
            nyasSeg = nyasSeg[sort_arg,:]

            trueboundaries = np.reshape(nyasSeg,nyasSeg.size)
            trueboundaries = list(set(trueboundaries.tolist()))
            trueboundaries = np.sort(trueboundaries).tolist()

            if baselineMethod==2:
                boundaries_random = np.arange(0,time_last,UNIFORM_BOUNDARY_GAP)
            elif baselineMethod==1:
                boundaries_random=[]
                last_boundary_time =0
                while last_boundary_time <=time_last:
                    boundaries_random.append(last_boundary_time)
                    last_boundary_time += inter_boundary_interval[np.random.randint(len(inter_boundary_interval))]
            elif baselineMethod==3:
                boundaries_random = trueboundaries

            boundaries_random = np.array(boundaries_random)
            guessboundaries = np.reshape(boundaries_random,boundaries_random.size)
            guessboundaries = list(set(guessboundaries.tolist()))
            guessboundaries = np.sort(guessboundaries).tolist()

            boundP, boundR, boundF, meangtt, meanttg = self.calculateBoundaryPRF(guessboundaries, trueboundaries, THRESHOLD_FINE)

            ###classifying boundaries with the probababity of nyas segments computed earlier
            guess_nyas_segments = []
            for i,boundary in enumerate(boundaries_random[:-1]):
                rand_num = np.random.rand()
                if rand_num<=nyas_prob:
                    guess_nyas_segments.append([boundaries_random[i],boundaries_random[i+1]])

            overlapP, overlapR, overlapF = self.calculateOverlapPRF(guess_nyas_segments, nyasSeg)

            self.stats.append([ boundP, boundR, boundF, meangtt, meanttg, overlapP, overlapR, overlapF])

        return self.stats


    def calculateBoundaryPRF(self, resBoundaries, gtBoundaries, THRESHOLD_FINE):

        #THRESHOLD_FINE=0.1

        if len(resBoundaries)==0:
            return 0,0,0,0,0

        # Evaluate boundaries (Precision)
        fineMatches=0
        for i in range(0,len(resBoundaries)):
            for j in range(0,len(gtBoundaries)):
                if abs(gtBoundaries[j]-resBoundaries[i])<THRESHOLD_FINE:
                    fineMatches+=1
                    break
        fP=float(fineMatches)/float(len(resBoundaries))
        # Evaluate boundaries (guess to true)
        gtt=[]
        for i in range(0,len(resBoundaries)):
            minTime=10000000
            for j in range(0,len(gtBoundaries)):
                dif=abs(gtBoundaries[j]-resBoundaries[i])
                if dif<minTime: minTime=dif
            gtt.append(minTime)
        # Evaluate boundaries (Recall)
        fineMatches=0
        for i in range(0,len(gtBoundaries)):
            for j in range(0,len(resBoundaries)):
                if abs(gtBoundaries[i]-resBoundaries[j])<THRESHOLD_FINE:
                    fineMatches+=1
                    break
        fR=float(fineMatches)/float(len(gtBoundaries))
        # Evaluate boundaries (true to guess)
        ttg=[]
        for i in range(0,len(gtBoundaries)):
            minTime=10000000
            for j in range(0,len(resBoundaries)):
                dif=abs(gtBoundaries[i]-resBoundaries[j])
                if dif<minTime: minTime=dif
            ttg.append(minTime)
        # Evaluate boundaries (F-measure)
        if fP>0 or fR>0: fF=2.0*fP*fR/(fP+fR)
        else: fF=0.0

        return fP,fR,fF, np.median(gtt), np.median(ttg)

    def calculateOverlapPRF(self, guessNyas, trueNyas):

        RESOLUTION = 0.1

        if len(guessNyas)==0:
            return 0,0,0

        min_time = np.min([np.min(guessNyas),np.min(guessNyas)])
        max_time = np.max([np.max(trueNyas),np.max(trueNyas)])

        vals = np.arange(min_time, max_time,RESOLUTION)

        if vals.size==0:
            return 0,0,0

        vals = np.append(vals, RESOLUTION+ vals[-1])

        guessNyas_flag = np.zeros(len(vals))
        trueNyas_flag = np.zeros(len(vals))

        for seg in guessNyas:
            ind_str = np.argmin(abs(vals-seg[0]))
            ind_end = np.argmin(abs(vals-seg[1]))
            guessNyas_flag[ind_str:ind_end+1] = 1

        for seg in trueNyas:
            ind_str = np.argmin(abs(vals-seg[0]))
            ind_end = np.argmin(abs(vals-seg[1]))
            trueNyas_flag[ind_str:ind_end+1] = 1

        N_ind_guess = np.where(guessNyas_flag==1)[0]
        N_ind_true = np.where(trueNyas_flag==1)[0]

        N_match_ind = list(set(N_ind_guess.tolist()) & set(N_ind_true.tolist()))

        Precision = len(N_match_ind)/float(N_ind_guess.shape[0])
        Recall = len(N_match_ind)/float(N_ind_true.shape[0])

        if Precision>0 or Recall>0: Fmeas=2.0*Precision*Recall/(Precision+Recall)
        else: Fmeas=0.0

        return Precision, Recall, Fmeas

    def extractMatchMTX(self, root_dir, segmentFileExt, matchMTXFile, foldInformationFileName, featureType):
        """
        same as featuer extraction but doing match matrix generation
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
        foldInfo={}
        segment_index_pointer=0
        for file_cnt, filename in enumerate(filenames):
            foldInfo[filename]={}
            fname, ext = os.path.splitext(filename)

            nfile = fname + self.nyasAnnotationFileSuffix
            nyasSeg = np.genfromtxt(nfile)  #this file contains only the information about valid nyas segments.
            nyasSeg = nyasSeg[:,:2]
            nyasSeg = nyasSeg.astype(np.float)

            #because in the annotations there were two tired for annotation, all the nyas segments in thsi files might not be in order
            #sorting nyas annotations
            sort_arg = np.argsort(nyasSeg[:,0])
            nyasSeg = nyasSeg[sort_arg,:]


            nyasproc = MS.NyasProcessing()
            ph_obj = MS.PitchProcessing(pitchfile = fname + '.essentia.pitch', tonicfile = fname +'.tonic')
            ph_obj.PitchHz2Cents()

            pitchSegments = np.loadtxt(fname + segmentFileExt)

            #Since all the segments are read, remove the trivial once which have mainly silence in them.
            pitchSegments = nyasproc.removeSegmentsWithSilence(ph_obj.timepitch, ph_obj.phop,pitchSegments)

            labels = np.array(nyasproc.obtainClassLabels(pitchSegments,fname+nyasAnnotationFileSuffix, ph_obj.phop ,ph_obj.pCents.shape[0]))

            segments = pitchSegments
            pitchSegments = pitchSegments/ph_obj.phop

            if file_cnt==0:
                pitchArray = ph_obj.pCents
                segmentsArray = pitchSegments
                labelsArray = labels
            else:
                sample_off = pitchArray.shape[0]
                pitchArray = np.append(pitchArray, ph_obj.pCents,axis=0)
                segmentsArray = np.append(segmentsArray,pitchSegments + sample_off,axis=0)
                labelsArray = np.append(labelsArray, labels, axis=0)

            n_folds_current = np.floor((nyasSeg.shape[0]/min_Num_Nyas)).astype(np.int16)
            remainder = nyasSeg.shape[0] - n_folds_current*min_Num_Nyas
            str_fold=0
            foldInfo[filename]['guessSeg']=[]
            foldInfo[filename]['trueSeg']=[]
            for i in range(n_folds_current):
                print "Hello", i
                ind_fold_more = np.where(segments[:,0]>=str_fold)[0]
                if not (i == n_folds_current-1):
                    ind_fold_less = np.where(segments[:,0]<nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1])[0]
                    ind_fold = np.array(list(set(list(ind_fold_more)) & set(list(ind_fold_less))))
                    foldInfo[filename]['trueSeg'].append(nyasSeg[i*min_Num_Nyas:i*min_Num_Nyas+min_Num_Nyas,:].tolist())

                else:
                    ind_fold = ind_fold_more
                    foldInfo[filename]['trueSeg'].append(nyasSeg[i*min_Num_Nyas:,:].tolist())

                if ind_fold.shape[0] == 0:
                    print "Something is terribly wrong"

                foldInfo[filename]['guessSeg'].append([segments[ind_fold,:].tolist(), (ind_fold+segment_index_pointer).tolist()])

                str_fold = nyasSeg[(i*min_Num_Nyas) +min_Num_Nyas-1,1]

            segment_index_pointer+=segments.shape[0]

        matchMTX = tss.computeMatchMatrix(pitchArray, segmentsArray, pitchArray, segmentsArray, featureType)

        #filling the other half of matrix
        for i in xrange(matchMTX.shape[0]):
            for j in range(i):
                matchMTX[i,j]=matchMTX[j,i]

        fid = open(matchMTXFile,'w')
        json.dump({'matchMTX':matchMTX.tolist(), 'labels':labelsArray.tolist()},fid, indent=4)
        fid.close()


        fid = open(foldInformationFileName,'w')
        json.dump(foldInfo,fid, indent=4)
        fid.close()



class classificationExperiment():

    expParam = {}

    def __init__(self):
        pass
    def featureSelectionManual(self, THRESHOLD_FINE):

        l1 = ['length']
        l2 = ['isflat']
        l3 = ['variance']
        l4 = l1+l2+l3
        l5 = l4 + ['mean']
        l6 = l5 + ['varPeakDist']
        l7 = l6 + ['meanPeakDist', 'meanPeakAmp', 'varPeakAmp','tCentroid']

        c1 = ['rel_len_longest']
        c2 = ['prev_variance']
        c3 = ['post_sil_dur']
        c4 = c1 + c2 + c3
        c5 = c4 + ['prev_length']
        c6 = c5 + ['prev_isflat']
        c7 = c6 + ['pre_sil_dur']
        c8 = c7 + ['rel_len_BP', 'rel_len_pre_segment', 'rel_len_post_segment', 'prev_mean', 'prev_tCentroid', 'prev_meanPeakDist', 'prev_varPeakDist', 'prev_meanPeakAmp', 'prev_varPeakAmp']

        lc1 = l4 + c4
        lc2 = l7 + c8

        featureSets = [l1,l2,l3,l4,l7,c1,c2,c3,c4,c8,lc1,lc2]
        #featureSets = [l1,l2,l3,l4,l5,l6,l7,c1,c2,c3,c4,c5,c6,c7,c8,lc1,lc2]

        classifierSets = [('tree',{'min_samples_split':10}), ('kNN','default'),('nbMulti',{'fit_prior':False}),('logReg',{'class_weight':'auto'}),('svm',{'class_weight':'auto'}),('randC','default')]

        featureFiles = ['OwnSegmentAllFeaturesFullDataset', 'KeoghSegment75AllFeaturesFullDataset', 'KeoghSegment50AllFeaturesFullDataset','KeoghSegment25AllFeaturesFullDataset','KeoghSegment10AllFeaturesFullDataset']


        out_dir = 'experimentResults_WITH_ARTIST_RAG_FILTERING_'+ str(np.floor(THRESHOLD_FINE*1000).astype(np.int16))

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)



        logfile= open(out_dir+'/ClassificationExperimentLogs.txt','ab')


        logfile.write("This is the mapping of feature sets , classifier sets and feature files:\n")
        for i, features in enumerate(featureSets):
            logfile.write(("feat"+str(i)+":"+ "\t%s"*len(features)+"\n")%tuple(features))
        for i, classifier in enumerate(classifierSets):
            logfile.write(("class"+str(i)+":"+ "\t%s\n")%classifier[0])
        for i, featfile in enumerate(featureFiles):
            logfile.write(("file"+str(i)+":"+ "\t%s\n")%featfile)

        logfile.write("\n")
        logfile.write("\n")
        logfile.write("\n")
        logfile.write("total number of experiments are %d\n"%(len(featureSets)*len(classifierSets)*len(featureFiles)))
        logfile.write("\n")
        logfile.write("\n")
        logfile.write("\n")

        nyasExp = nyasIdentification('/media/Data/Dropbox/Kaustuv_Annotations/DatasetNyas/')
        exp_index=0
        logfile.close()
        for file_cnt, featureFile in enumerate(featureFiles):
            logfile= open(out_dir+'/ClassificationExperimentLogs.txt','ab')
            logfile.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            logfile.write("\n")
            logfile.write("\n")
            logfile.write("\n")
            logfile.close()
            for feature_cnt, feature in enumerate(featureSets):
                for class_cnt, classifier in enumerate(classifierSets):
                    print featureFile, feature, classifier
                    print file_cnt, feature_cnt, class_cnt
                    logfile= open(out_dir+'/ClassificationExperimentLogs.txt','ab')
                    logfile.write("file"+str(file_cnt)+"\tfeat"+str(feature_cnt)+"\tclass"+str(class_cnt)+"\tExpIndex"+str(exp_index)+"\n")

                    nyasExp.performTrainTest(featureFile+'.arff', featureFile+'.json', feature, THRESHOLD_FINE, classifierInfo=classifier, DoArtistRagFiltering=1)

                    fid = open(out_dir+'/ExpData'+str(exp_index)+'.json','w')
                    json.dump(nyasExp.stats,fid, indent=4)
                    fid.close()

                    out = np.mean(np.array(nyasExp.stats),axis=0)
                    logfile.write(("%f\t"*len(out)+"\n")%tuple(out))

                    exp_index+=1
                    logfile.close()


    def computeNyasStatistics(self, FoldInfoFile):

        Info = json.load(open(FoldInfoFile))

        nyas_seg = []
        for key in Info:
            for fold in Info[key]['trueSeg']:
                nyas_seg.extend(fold)

        dur_Arr = []
        for nyas in nyas_seg:
            dur_Arr.append(nyas[1]-nyas[0])


        return np.sort(dur_Arr), (np.min(dur_Arr), np.max(dur_Arr), np.mean(dur_Arr), np.median(dur_Arr), np.var(dur_Arr))