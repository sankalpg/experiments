import numpy as np
import os, sys
import pickle
import numpy as np
import psycopg2 as psy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/machineLearning'))
import mlWrapper as ml

def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                names.append(path + "/" + f)
    return names

def get_all_bin_freqs(root_dir, feature_file_ext):
    
    filenames = GetFileNamesInDir(root_dir, feature_file_ext)
    
    bin_inds = []
    for filename in filenames:
        feature = pickle.load(open(filename,'r'))
        bin_inds.extend(feature.keys())
    
    return np.unique(np.array(bin_inds))

def generate_feature_index_mapping(bin_inds, feature_names):
    
    len_feature = len(bin_inds)*len(feature_names)
    cnt=0
    mapping = {}
    for b in bin_inds:
        if not mapping.has_key(b):
            mapping[b] = {}
        for f in feature_names:
            if not mapping[b].has_key(f):
                mapping[b][f] = cnt
                cnt+=1
    return mapping, cnt


def create_feature_per_file(filename, feat_bin_map, feat_dim, init_by = 0):
    
    feature_data = pickle.load(open(filename, 'r'))
    feature = init_by*np.ones(feat_dim)
    
    for k1 in feature_data.keys():
        for k2 in feature_data[k1].keys():
            try:
                feature[feat_bin_map[k1][k2]] = feature_data[k1][k2]
            except:
                pass
    return feature

def stripPrefix(audiofile,localPrefix ):
    
    if audiofile.count(localPrefix):
        audiofile_WOPre = audiofile.split(localPrefix)[1]
    else:
        print "please provide files with known prefixes (paths)"
        audiofile_WOPre = audiofile;
    return audiofile_WOPre


def filter_bins(bin_inds, bins_range):
    """
    To filter bins based on a range
    """

    ind1 = np.where(bin_inds>=bins_range[0])[0]
    ind2 = np.where(bin_inds<=bins_range[1])[0]
    ind = np.intersect1d(ind1,ind2)
    return bin_inds[ind]

def get_feature_per_collection(root_dir, myDatabase, init_by=0, skew_type=0, feature_file_ext= '.params', bins_range = []):
    """
    skew_type = 0 uses skew1, 1 uses skew2 and 2 uses skew1 and skew2 both
    """
    
    #fetch first the unique freq bins in the dataset
    bin_inds = get_all_bin_freqs(root_dir, feature_file_ext)

    #filtering based on some frequencies. NOTE: not all the experiments that Gopal do have bins octave folded. So in some we just have to select bins between 0-1200
    if len(bins_range) ==2:
        bin_inds = filter_bins(bin_inds, bins_range)
    
    #generate a mapping of feature to bin
    if skew_type == 0:
        feature_names = ['amplitude', 'position', 'variance', 'kurtosis', 'skew1', 'mean']
    elif skew_type == 1:
        feature_names = ['amplitude', 'position', 'variance', 'kurtosis', 'skew2', 'mean']
    elif skew_type == 2:
        feature_names = ['amplitude', 'position', 'variance', 'kurtosis', 'skew2', 'skew1', 'mean']
        
    mapping, feat_dim = generate_feature_index_mapping(bin_inds, feature_names)
    
    filenames = GetFileNamesInDir(root_dir, feature_file_ext)
    
    #create an ampty matrix of features
    features = np.zeros((len(filenames), feat_dim))
    
    cmd1 = "select mbid, raagaid from file where mbid = '%s'"
    
    try:
        con = psy.connect(database=myDatabase, user='sankalp') 
        cur = con.cursor()
    except:
        print "Error connecting to database"
        return -1
    mbid_list = []
    ragaid_list = []
    for ii, filename in enumerate(filenames):
        filen = stripPrefix(filename, root_dir)
        fname = filen.replace(feature_file_ext, '')
        fname  = os.path.basename(fname)
        cur.execute(cmd1%(fname))
        mbid, ragaid = cur.fetchall()[0]
        mbid_list.append(mbid)
        ragaid_list.append(ragaid)
        f_pf = create_feature_per_file(filename, mapping, feat_dim, init_by = init_by)
        features[ii,:] = f_pf
    return mbid_list, ragaid_list, features
   
def run_gopalas_Exp_40Raga_db_Histogram():
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/histogram_based/features/params_pickle'
    output_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/histogram_based/results'
    myDatabase = 'Raga_Rec_Carnatic_40Raga_Config0'
    skew_types = [2]
    init_bys = [-10000]
    classifiers = [('svm', "default"),
                    ('svm-lin', "default"),
                    ('sgd', "default"),
                    ('nb-multi', "default"),
                    ('nb-gauss', "default"),
                    ('nb-bern', "default"),
                    ('randForest', "default"),
                    ('logReg', "default"),
                    ('kNN', {'n_neighbors':1}),
                    ('kNN', {'n_neighbors':3}),
                    ('kNN', {'n_neighbors':5})]   
    n_fold = -1
    n_expts = 1
    
    fid = open(os.path.join(output_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*4%("Skew_type", "initialize_by", "classifier", "accuracy"))
    fid.write('\n')
    
    cnt=1
    
    for s in skew_types:
        for i in init_bys:
            for c in classifiers:
                out_dir = os.path.join(output_dir, 'Config_%s'%str(cnt))

                if os.path.isdir(out_dir):
                    results = pickle.load(open(os.path.join(out_dir, 'experiment_results.pkl'),'r'))
                    fid.write("%s\t"*4%(s,i,c,results['accuracy']))
                    fid.write("\n")

                else:

                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)                
                    mbids, ragaids, features = get_feature_per_collection(root_dir, myDatabase, init_by=i, skew_type=s, feature_file_ext= '-params.pickle')
                    print features.shape
                    mlObj  = ml.experimenter()
                    mlObj.setExperimentParams(nExp = n_expts, typeEval = ("LeaveOneOut",n_fold), nInstPerClass = -1, classifier = c, balanceClasses=1)
                    nans = np.isnan(features)
                    features[nans] = 0
                    infs = np.isinf(features)
                    features[infs] = 0
                    #making them all positives
                    min_val = np.min(features)
                    features = features - min_val
                    mlObj.setFeaturesAndClassLabels(features, np.array(ragaids))
                    mlObj.runExperiment()
                    fid.write("%s\t"*4%(s,i,c,mlObj.overallAccuracy))
                    fid.write("\n")
                    results = {'cm': mlObj.cMTXExp, 'gt_label': mlObj.classLabelsInt, 'pred_label':mlObj.decArray, 'mbids': mbids, 'mapping':mlObj.cNames, 'accuracy': mlObj.overallAccuracy}
                    pickle.dump(results, open(os.path.join(out_dir, 'experiment_results.pkl'),'w'))
                cnt+=1
    fid.close()

def run_gopalas_Exp_40Raga_db_ContextBased():
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based/features/grossparams_pickle'
    output_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based/results'
    myDatabase = 'Raga_Rec_Carnatic_40Raga_Config0'
    skew_types = [2]
    init_bys = [-10000]
    classifiers = [('svm', "default"),
                    ('svm-lin', "default"),
                    ('sgd', "default"),
                    ('nb-multi', "default"),
                    ('nb-gauss', "default"),
                    ('nb-bern', "default"),
                    ('randForest', "default"),
                    ('logReg', "default"),
                    ('kNN', {'n_neighbors':1}),
                    ('kNN', {'n_neighbors':3}),
                    ('kNN', {'n_neighbors':5})]    
    n_fold = -1
    n_expts = 1
    
    fid = open(os.path.join(output_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*4%("Skew_type", "initialize_by", "classifier", "accuracy"))
    fid.write('\n')
    
    cnt=1
    
    for s in skew_types:
        for i in init_bys:
            for c in classifiers:
                out_dir = os.path.join(output_dir, 'Config_%s'%str(cnt))
                
                if os.path.isdir(out_dir):
                    results = pickle.load(open(os.path.join(out_dir, 'experiment_results.pkl'),'r'))
                    fid.write("%s\t"*4%(s,i,c,results['accuracy']))
                    fid.write("\n")

                else:
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    
                    mbids, ragaids, features = get_feature_per_collection(root_dir, myDatabase, init_by=i, skew_type=s, feature_file_ext= '-grossparams.pickle')
                    print features.shape
                    mlObj  = ml.experimenter()
                    mlObj.setExperimentParams(nExp = n_expts, typeEval = ("LeaveOneOut",n_fold), nInstPerClass = -1, classifier = c, balanceClasses=1)
                    nans = np.isnan(features)
                    features[nans] = 0
                    infs = np.isinf(features)
                    features[infs] = 0
                    #making them all positives
                    min_val = np.min(features)
                    features = features - min_val
                    mlObj.setFeaturesAndClassLabels(features, np.array(ragaids))
                    mlObj.runExperiment()
                    fid.write("%s\t"*4%(s,i,c,mlObj.overallAccuracy))
                    fid.write("\n")
                    results = {'cm': mlObj.cMTXExp, 'gt_label': mlObj.classLabelsInt, 'pred_label':mlObj.decArray, 'mbids': mbids, 'mapping':mlObj.cNames, 'accuracy': mlObj.overallAccuracy}
                    pickle.dump(results, open(os.path.join(out_dir, 'experiment_results.pkl'),'w'))

                cnt+=1
    fid.close()    
        


def run_gopalas_Exp_40Raga_db_ContextBasedJNMR():
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based_JNMR/features/grossparams_pickle'
    output_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based_JNMR/results'
    myDatabase = 'Raga_Rec_Carnatic_40Raga_Config0'
    skew_types = [0]
    init_bys = [0]
    classifiers = [('svm', "default"),
                    ('svm-lin', "default"),
                    ('sgd', "default"),
                    ('nb-multi', "default"),
                    ('nb-gauss', "default"),
                    ('nb-bern', "default"),
                    ('randForest', "default"),
                    ('logReg', "default"),
                    ('kNN', {'n_neighbors':1}),
                    ('kNN', {'n_neighbors':3}),
                    ('kNN', {'n_neighbors':5})]    
    n_fold = -1
    n_expts = 1
    
    fid = open(os.path.join(output_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*4%("Skew_type", "initialize_by", "classifier", "accuracy"))
    fid.write('\n')
    
    cnt=1
    
    for s in skew_types:
        for i in init_bys:
            for c in classifiers:
                out_dir = os.path.join(output_dir, 'Config_%s'%str(cnt))
                
                if os.path.isdir(out_dir):
                    results = pickle.load(open(os.path.join(out_dir, 'experiment_results.pkl'),'r'))
                    fid.write("%s\t"*4%(s,i,c,results['accuracy']))
                    fid.write("\n")

                else:
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    
                    mbids, ragaids, features = get_feature_per_collection(root_dir, myDatabase, init_by=i, skew_type=s, feature_file_ext= '-grossparams.pickle')
                    print features.shape
                    mlObj  = ml.experimenter()
                    mlObj.setExperimentParams(nExp = n_expts, typeEval = ("LeaveOneOut",n_fold), nInstPerClass = -1, classifier = c, balanceClasses=1)
                    nans = np.isnan(features)
                    features[nans] = 0
                    infs = np.isinf(features)
                    features[infs] = 0
                    #making them all positives
                    min_val = np.min(features)
                    features = features - min_val
                    mlObj.setFeaturesAndClassLabels(features, np.array(ragaids))
                    mlObj.runExperiment()
                    fid.write("%s\t"*4%(s,i,c,mlObj.overallAccuracy))
                    fid.write("\n")
                    results = {'cm': mlObj.cMTXExp, 'gt_label': mlObj.classLabelsInt, 'pred_label':mlObj.decArray, 'mbids': mbids, 'mapping':mlObj.cNames, 'accuracy': mlObj.overallAccuracy}
                    pickle.dump(results, open(os.path.join(out_dir, 'experiment_results.pkl'),'w'))

                cnt+=1
    fid.close()    


def run_gopalas_Exp_40Raga_db_ContextBasedJNMRFiltered():
    root_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based_JNMR/features/grossparams_pickle'
    output_dir = '/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RagaRecognition/ThesisResults2016/Gopala/context_based_JNMR/results_0_1200_bins'
    myDatabase = 'Raga_Rec_Carnatic_40Raga_Config0'
    skew_types = [0]
    init_bys = [0]
    classifiers = [('svm', "default"),
                    ('svm-lin', "default"),
                    ('sgd', "default"),
                    ('nb-multi', "default"),
                    ('nb-gauss', "default"),
                    ('nb-bern', "default"),
                    ('randForest', "default"),
                    ('logReg', "default"),
                    ('kNN', {'n_neighbors':1}),
                    ('kNN', {'n_neighbors':3}),
                    ('kNN', {'n_neighbors':5})]    
    n_fold = -1
    n_expts = 1
    
    fid = open(os.path.join(output_dir, 'results_summary.txt'),'w')
    fid.write("%s\t"*4%("Skew_type", "initialize_by", "classifier", "accuracy"))
    fid.write('\n')
    
    cnt=1
    
    for s in skew_types:
        for i in init_bys:
            for c in classifiers:
                out_dir = os.path.join(output_dir, 'Config_%s'%str(cnt))
                
                if os.path.isdir(out_dir):
                    results = pickle.load(open(os.path.join(out_dir, 'experiment_results.pkl'),'r'))
                    fid.write("%s\t"*4%(s,i,c,results['accuracy']))
                    fid.write("\n")

                else:
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    
                    mbids, ragaids, features = get_feature_per_collection(root_dir, myDatabase, init_by=i, skew_type=s, feature_file_ext= '-grossparams.pickle', bins_range = [0, 1200])
                    print features.shape
                    mlObj  = ml.experimenter()
                    mlObj.setExperimentParams(nExp = n_expts, typeEval = ("LeaveOneOut",n_fold), nInstPerClass = -1, classifier = c, balanceClasses=1)
                    nans = np.isnan(features)
                    features[nans] = 0
                    infs = np.isinf(features)
                    features[infs] = 0
                    #making them all positives
                    min_val = np.min(features)
                    features = features - min_val
                    mlObj.setFeaturesAndClassLabels(features, np.array(ragaids))
                    mlObj.runExperiment()
                    fid.write("%s\t"*4%(s,i,c,mlObj.overallAccuracy))
                    fid.write("\n")
                    results = {'cm': mlObj.cMTXExp, 'gt_label': mlObj.classLabelsInt, 'pred_label':mlObj.decArray, 'mbids': mbids, 'mapping':mlObj.cNames, 'accuracy': mlObj.overallAccuracy}
                    pickle.dump(results, open(os.path.join(out_dir, 'experiment_results.pkl'),'w'))

                cnt+=1
    fid.close()    


if __name__ == "__main__":
    
    pass
    
    