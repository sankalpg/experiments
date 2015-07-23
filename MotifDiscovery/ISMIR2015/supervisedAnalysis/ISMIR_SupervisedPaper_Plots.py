import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import copy

sys.path.append('/home/sankalp/Work/Work_PhD/experiments/MotifDiscovery/RepAndDistComparison/')
import EvaluateSupervisedAnalysis as eval
   
def carnatic_complexity_example(plotName=-1):
    
    subSeqFileTN  = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/NCR100/r1/DB3/DB3.SubSeqsTN'
    patternInfoFile = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/NCR100/r1/DB3/DB3.SubSeqsInfo'
    phraseInd=[0, 38, 9207]
    plotName='CarnaticComplexityExample.pdf'
    subSeqLen = 800
    hopSize = 0.0029
    #These are the info for the selected patterns
    """[  5.92000000e-01   1.53900000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+03]
    [  176.930249     1.509297     4.           6.        1000.      ]
    [ 100.948753    1.254      11.         -1.         -1.      ]"""
    
    
    subData = np.fromfile(subSeqFileTN)
    if np.mod(len(subData),subSeqLen)!=0:
        print "Please provide a subsequence database and subSeqLen which make sense, total number of elements in the database should be multiple of subSeqLen"
        return -1
    subData = np.reshape(subData, (len(subData)/float(subSeqLen), subSeqLen))
    
    pattInfos = np.loadtxt(patternInfoFile)
    colors = ['b', 'r', 'm']
    markerArr = ['.', '*' , 'o']
    CategoryNames = ['$P_1$', '$P_2$', '$P_3$']
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.hold(True)
    
    offset = np.arange(10)*600
    pLeg=[]
    downSampleFactor = 5#needed for the markers to be clearly visible
    for ii, line in enumerate(pattInfos[phraseInd]):
        print line
        length = np.floor(line[1]/hopSize)
        d = subData[phraseInd[ii],:length] - np.mean(subData[phraseInd[ii],:length]) + offset[ii]
        d = d[::downSampleFactor]
        p, = plt.plot(hopSize*downSampleFactor*np.arange(len(d)), d,color = colors[ii], linewidth=2, marker= markerArr[ii], markersize = 5)
        pLeg.append(p)
    
    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    plt.xlabel("Time (s)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency (Cent)", fontsize = fsize, fontname=font)
    
    plt.xlim(np.array([0,600])*hopSize)
    plt.ylim(np.array([-600,1500]))
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2.5*float(yLim[1]-yLim[0])))
    
    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1 
    
    
def hindustani_flat_note_compression_example(plotName=-1):
    
    subSeqFileTN  = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/NCR100/r1/DB4/DB4.SubSeqsTN'
    patternInfoFile = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/NCR100/r1/DB4/DB4.SubSeqsInfo'
    
    phraseInd=[84, 277]
    plotName='Hindusani_flat_note_compression_example_reversed.pdf'
    subSeqLen = 1600
    hopSize = 0.005
    
    subData = np.fromfile(subSeqFileTN)
    if np.mod(len(subData),subSeqLen)!=0:
        print "Please provide a subsequence database and subSeqLen which make sense, total number of elements in the database should be multiple of subSeqLen"
        return -1
    subData = np.reshape(subData, (len(subData)/float(subSeqLen), subSeqLen))
    
    pattInfos = np.loadtxt(patternInfoFile)
    colors = ['b', 'r', 'm']
    markerArr = ['.', '*' , 'o']
    CategoryNames = ['$P_{1a}$', '$P_{2a}$', '$P_{1b}$', '$P_{2b}$']
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.hold(True)
    
    offset = [0,0]
    pLeg=[]
    downSampleFactor = 5#needed for the markers to be clearly visible
    for ii, line in enumerate(pattInfos[phraseInd]):
        print line
        length = np.floor(line[1]/hopSize)
        d = subData[phraseInd[ii],:length]
        d = d[::downSampleFactor]+ 600
        p, = plt.plot(hopSize*downSampleFactor*np.arange(len(d)), d,color = colors[ii], linewidth=2, marker= markerArr[ii], markersize = 5)
        pLeg.append(p)
        
        
    subSeqFileTN  = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/NCR100/r1/DB12/DB12.SubSeqsTN'
    patternInfoFile = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/NCR100/r1/DB12/DB12.SubSeqsInfo'
    
    subData = np.fromfile(subSeqFileTN)
    if np.mod(len(subData),subSeqLen)!=0:
        print "Please provide a subsequence database and subSeqLen which make sense, total number of elements in the database should be multiple of subSeqLen"
        return -1
    subData = np.reshape(subData, (len(subData)/float(subSeqLen), subSeqLen))
    pattInfos = np.loadtxt(patternInfoFile)
    
    colors = ['b', 'r', 'm']
    markerArr = ['o', 'p' , 'o']
    
    for ii, line in enumerate(pattInfos[phraseInd]):
        print line
        length = np.floor(line[1]/hopSize)
        d = subData[phraseInd[ii],:length]
        d = d[::downSampleFactor] 
        p, = plt.plot(hopSize*downSampleFactor*np.arange(len(d)), d,color = colors[ii], linewidth=2, marker= markerArr[ii], markersize = 5)
        pLeg.append(p)
        
    
    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    plt.xlabel("Time (s)", fontsize = fsize, fontname=font)
    plt.ylabel("Frequency (Cent)", fontsize = fsize, fontname=font)
    
    plt.xlim(np.array([0,1200])*hopSize)
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2.5*float(yLim[1]-yLim[0])))
    
    plt.legend(pLeg, [CategoryNames[pp] for pp in range(len(CategoryNames))], loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1

def plotPerCategoryMAPHindustaniBOXPLOTS(plotName=-1):
    
    outConfigs = [21,30]
    
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/results/NCR100/r1/out'
    config_file = 'configFiles_462'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/audioCollection/AllFiles.flist'
    
    perClassResults = []
    for ii, outConfig in enumerate(outConfigs):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit4')
        perClassResults.append({})
        perClassResults[ii]['mean']=[]
        perClassResults[ii]['std']=[]
        perClassResults[ii]['ap']=[]
        print np.mean(output[0])
        for k in output[1].keys():
            perClassResults[ii]['mean'].append(np.mean(output[1][k]))
            perClassResults[ii]['std'].append(np.std(output[1][k]))
            perClassResults[ii]['ap'].append(output[1][k])

    fig, ax = plt.subplots()
    n_groups = len(perClassResults[0]['mean'])
    index = np.arange(n_groups)
    bar_width = .25
    
    
    
    temp = []
    temp.append(perClassResults[0]['ap'][0])
    temp.append(perClassResults[1]['ap'][0])
    
    temp.append(perClassResults[0]['ap'][1])
    temp.append(perClassResults[1]['ap'][1])
                                
    temp.append(perClassResults[0]['ap'][2])
    temp.append(perClassResults[1]['ap'][2])
    
    temp.append(perClassResults[0]['ap'][3])
    temp.append(perClassResults[1]['ap'][3])
    
    temp.append(perClassResults[0]['ap'][4])
    temp.append(perClassResults[1]['ap'][4])
                                
    box = plt.boxplot(temp, patch_artist=True)
    
    colors = ['#B9B9B9', '#33CCFF', '#B9B9B9', '#33CCFF','#B9B9B9', '#33CCFF','#B9B9B9', '#33CCFF','#B9B9B9', '#33CCFF','#B9B9B9', '#33CCFF','#B9B9B9', '#33CCFF']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    fsize = 16
    fsize2 = 12
    font="Times New Roman"
    
    plt.xlabel('Phrase Category', fontsize = fsize, fontname=font, labelpad=25)
    plt.ylabel('Average precision', fontsize = fsize, fontname=font)
    #plt.title('Scores by group and gender')
    plt.xticks(1+np.arange(10), 5*['$M_{B}$', '$M_{DT}$'], fontsize = fsize, fontname=font)
    #plt.legend(loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #plt.tight_layout()
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1


def plotPerCategoryMAPCarnaticBOXPLOTS(plotName=-1):
    
    outConfigs = [5,17,23]
    
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/results/NCR100/r1/out'
    config_file = 'configFiles_524'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/audioCollection/AllFiles.flist'
    
    perClassResults = []
    ap_per_categories = []
    for ii, outConfig in enumerate(outConfigs):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit1')
        perClassResults.append({})
        perClassResults[ii]['mean']=[]
        perClassResults[ii]['std']=[]
        perClassResults[ii]['ap']=[]
        print np.mean(output[0])
        for k in output[1].keys():
            perClassResults[ii]['mean'].append(np.mean(output[1][k]))
            perClassResults[ii]['std'].append(np.std(output[1][k]))
            perClassResults[ii]['ap'].append(output[1][k])
    fig, ax = plt.subplots()
    n_groups = len(perClassResults[0]['mean'])
    index = np.arange(n_groups)
    bar_width = .25
    
    
    
    temp = []
    temp.append(perClassResults[0]['ap'][0])
    temp.append(perClassResults[1]['ap'][0])
    temp.append(perClassResults[2]['ap'][0])

    temp.append(perClassResults[0]['ap'][1])
    temp.append(perClassResults[1]['ap'][1])
    temp.append(perClassResults[2]['ap'][1])
                                
    temp.append(perClassResults[0]['ap'][2])
    temp.append(perClassResults[1]['ap'][2])
    temp.append(perClassResults[2]['ap'][2])

    temp.append(perClassResults[0]['ap'][3])
    temp.append(perClassResults[1]['ap'][3])
    temp.append(perClassResults[2]['ap'][3])

    temp.append(perClassResults[0]['ap'][4])
    temp.append(perClassResults[1]['ap'][4])
    temp.append(perClassResults[2]['ap'][4])
                                
    box = plt.boxplot(temp, patch_artist=True)
    
    colors = ['#B9B9B9', '#33CCFF', '#FF6666', '#B9B9B9', '#33CCFF', '#FF6666', '#B9B9B9', '#33CCFF', '#FF6666', '#B9B9B9', '#33CCFF', '#FF6666', '#B9B9B9', '#33CCFF', '#FF6666']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    fsize = 16
    fsize2 = 12
    font="Times New Roman"
    
    plt.xlabel('Phrase Category', fontsize = fsize, fontname=font, labelpad=25)
    plt.ylabel('Average precision', fontsize = fsize, fontname=font)
    #plt.title('Scores by group and gender')
    plt.xticks(1+np.arange(15), 5*['$M_{B}$', '$M_{DT}$', '$M_{CW2}$'], fontsize = fsize, fontname=font)
    #plt.legend(loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #plt.tight_layout()
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1



def plotPerCategoryMAPCarnatic(plotName=-1):
    
    outConfigs = [5,17,23]
    
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/results/NCR100/r1/out'
    config_file = 'configFiles_524'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/audioCollection/AllFiles.flist'
    
    perClassResults = []
    for ii, outConfig in enumerate(outConfigs):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit1')
        perClassResults.append({})
        perClassResults[ii]['mean']=[]
        perClassResults[ii]['std']=[]
        print np.mean(output[0])
        for k in output[1].keys():
            perClassResults[ii]['mean'].append(np.mean(output[1][k]))
            perClassResults[ii]['std'].append(np.std(output[1][k]))
    fig, ax = plt.subplots()
    n_groups = len(perClassResults[0]['mean'])
    index = np.arange(n_groups)
    bar_width = .25

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, perClassResults[0]['mean'], bar_width,
                    alpha=opacity,
                    color='#B9B9B9',
                    error_kw=error_config,
                    label='$M_{B}$',
                    hatch="/")
                    #yerr = perClassResults[0]['std'])

    rects2 = plt.bar(index+bar_width, perClassResults[1]['mean'], bar_width,
                    alpha=opacity,
                    color='#33CCFF',
                    error_kw=error_config,
                    label='$M_{DT}$',
                    hatch="\\")
                    #yerr = perClassResults[1]['std'])
    
    rects3 = plt.bar(index+bar_width+bar_width, perClassResults[2]['mean'], bar_width,
                    alpha=opacity,
                    color='#FF6666',
                    error_kw=error_config,
                    label='$M_{CW2}$',
                    hatch="x")
                    #yerr = perClassResults[2]['std'])

    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel('Phrase Category', fontsize = fsize, fontname=font)
    plt.ylabel('MAP', fontsize = fsize, fontname=font)
    #plt.title('Scores by group and gender')
    plt.xticks(index + 1.5*bar_width, ('$C_1$', '$C_2$', '$C_3$', '$C_4$', '$C_5$'), fontsize = fsize, fontname=font)
    plt.legend(loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #plt.tight_layout()
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1
    

def plotPerCategoryMAPHindustani(plotName=-1):
    
    outConfigs = [21,30]
    
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/results/NCR100/r1/out'
    config_file = 'configFiles_462'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/audioCollection/AllFiles.flist'
    
    perClassResults = []
    for ii, outConfig in enumerate(outConfigs):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit4')
        perClassResults.append({})
        perClassResults[ii]['mean']=[]
        perClassResults[ii]['std']=[]
        print np.mean(output[0])
        for k in output[1].keys():
            perClassResults[ii]['mean'].append(np.mean(output[1][k]))
            perClassResults[ii]['std'].append(np.std(output[1][k]))
    fig, ax = plt.subplots()
    n_groups = len(perClassResults[0]['mean'])
    index = np.arange(n_groups)
    bar_width = .25

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, perClassResults[0]['mean'], bar_width,
                    alpha=opacity,
                    color='#B9B9B9',
                    error_kw=error_config,
                    label='$M_{B}$',
                    hatch="/")

    rects2 = plt.bar(index+bar_width, perClassResults[1]['mean'], bar_width,
                    alpha=opacity,
                    color='#33CCFF',
                    error_kw=error_config,
                    label='$M_{DT}$',
                    hatch="\\")


    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel('Phrase Category', fontsize = fsize, fontname=font)
    plt.ylabel('MAP', fontsize = fsize, fontname=font)
    #plt.title('Scores by group and gender')
    plt.xticks(index + bar_width, ('$H_1$', '$H_2$', '$H_3$', '$H_4$', '$H_5$'), fontsize = fsize, fontname=font)
    plt.legend(loc ='upper left', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #plt.tight_layout()
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1


def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            #if filter in f.lower():
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                #print(path+"/"+f)
                #print(path)
                #ftxt.write(path + "/" + f + "\n")
                names.append(path + "/" + f)
                
    return names     

def generateTableNumbers():
    
    outConfigsHindustani = [21, 31, 36, 37, 38, 39]
    outConfigsCarnatic = [5, 17, 20, 23, 24, 25, 26, 30, 27, 28, 29, 31 ]
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/results/NCR100/r1/out'
    #config_file = 'configFiles_462'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/audioCollection/AllFiles.flist'
    
    MAP_Hindustani = []
    std_Hindustani = []
    for ii, outConfig in enumerate(outConfigsHindustani):
        #dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFilePath = GetFileNamesInDir(os.path.join(base_dir+str(outConfig), methoVariant), dbFileExt)[0]
        searchFilePath = GetFileNamesInDir(os.path.join(base_dir+str(outConfig), methoVariant), searchExt)[0]
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(searchFilePath, dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit4')
        MAP_Hindustani.append(np.mean(output[0]))
        std_Hindustani.append(np.std(output[0]))
        print "For Hindustani configuration %d, mean = %f, std = %f\n"%(outConfig, MAP_Hindustani[-1], std_Hindustani[-1])
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/results/NCR100/r1/out'
    #config_file = 'configFiles_524'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/audioCollection/AllFiles.flist'
    
    MAP_Carnatic = []
    std_Carnatic = []
    for ii, outConfig in enumerate(outConfigsCarnatic):
        #dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFilePath = GetFileNamesInDir(os.path.join(base_dir+str(outConfig), methoVariant), dbFileExt)[0]
        searchFilePath = GetFileNamesInDir(os.path.join(base_dir+str(outConfig), methoVariant), searchExt)[0]
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(searchFilePath, dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit1')
        MAP_Carnatic.append(np.mean(output[0]))
        std_Carnatic.append(np.std(output[0]))
        print "For Carnatic configuration %d, mean = %f, std = %f\n"%(outConfig, MAP_Carnatic[-1], std_Carnatic[-1]) 
        
   

def plotMAPDurationTruncation(plotName=-1):
    
    outConfigsHindustani = [29, 30, 31, 32, 33, 34, 35]
    outConfigsCarnatic = [13, 14, 15, 16, 17, 18 ,19]

    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/results/NCR100/r1/out'
    config_file = 'configFiles_462'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/hindustaniDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/hindustaniDB/audioCollection/AllFiles.flist'
    
    accuracyHindustani = []
    for ii, outConfig in enumerate(outConfigsHindustani):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit4')
        accuracyHindustani.append(np.mean(output[0]))
    
    base_dir = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/results/NCR100/r1/out'
    config_file = 'configFiles_524'
    searchExt = '.motifSearch.motifSearch'
    dbFileExt = '.flist'
    methoVariant = 'var1'
    SubSeqsInfoExt = '.SubSeqsInfo'
    
    local = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    server = '/homedtic/sgulati/motifDiscovery/dataset/PatternProcessing_DB/supervisedDBs/carnaticDB/subSeqDB/'
    fileList = '/media/Data/Datasets/PatternProcessing_DB/supervisedDBs/carnaticDB/audioCollection/AllFiles.flist'
    
    accuracyCarnatic = []
    for ii, outConfig in enumerate(outConfigsCarnatic):
        dbFilePath = os.path.join(base_dir+str(outConfig), methoVariant, config_file+dbFileExt)
        dbFileName = open(dbFilePath).readlines()[0].strip()
        output = eval.evaluateSupSearchNEWFORMAT(os.path.join(base_dir+str(outConfig), methoVariant, config_file+searchExt), dbFileName.replace(server,local)+SubSeqsInfoExt, fileList, anotExt='.anotEdit1')
        accuracyCarnatic.append(np.mean(output[0]))    
    
    fig, ax = plt.subplots()
    markerArr = ['s', 'o']
    pLeg = []
    p, = plt.plot(accuracyHindustani, linewidth=2, color= 'b', marker= markerArr[0], markersize = 8)
    pLeg.append(p)
    p, = plt.plot(accuracyCarnatic, linewidth=2, color = 'r', marker= markerArr[1], markersize = 8)
    pLeg.append(p)
    
    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    
    plt.xlabel('$\delta (s)$', fontsize = fsize, fontname=font)
    plt.ylabel('MAP', fontsize = fsize, fontname=font)
    
    plt.legend(pLeg, ['HMD',  'CMD'], loc ='top right', ncol = 1, fontsize = fsize2, scatterpoints=1, frameon=True, borderaxespad=0.1)
    plt.xticks(range(len(outConfigsHindustani)), ('0.1', '0.3', '0.5', '0.75', '1.0', '1.5', '2.0'), fontsize = fsize, fontname=font)
    
    #plt.ylim(np.array([.25,0.8]))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2.5*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    #plt.tight_layout()
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1
        
    
    
if __name__ == "__main__":
    

    carnatic_complexity_example(plotName='CarnaticComplexityExample.pdf')
    hindustani_flat_note_compression_example(plotName='Hindusani_flat_note_compression_example.pdf')
    plotPerCategoryMAPCarnatic(plotName='carnaticPerCategoryPerformance.pdf')
    plotPerCategoryMAPHindustani(plotName='hindustaniPerCategoryPerformance.pdf')
    plotMAPDurationTruncation(plotName='MAP_per_Duration_Truncation.pdf')
    