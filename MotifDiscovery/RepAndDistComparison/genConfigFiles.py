import os, sys

def genConfigFiles(baseName, outDir, methodVariant=1, musicType = ''):
    combinations= {
                    'distType':[0,1],#0 = Euc, 1 = DTW 
                    'DTWType':[0,1],#0 = DTW (no local constraint), 1 = DTW( local constraint)
                    'DTWBand': [0.1, 0.2, 0.9],
                    'TSRepType': [0,1], #0= raw ts, 1 = quantized ts
                    'quantSize': [12,24], #0= raw ts, 1 = quantized ts                  
                    'normType': [0,1,2,3,4,5,7], #0=
                    'sampleRate': [10,15,20,25,30],
                    'nInterpFac': [1, 5]
                    }
    
    methodVariants = ['var1', 'var2']
    
    params={
        'distType':combinations['distType'],
        'DTWBand': combinations['DTWBand'],
        'DTWType': combinations['DTWType'],
        'rankRefDistType': 0,
        'TSRepType': combinations['TSRepType'],
        'quantSize': combinations['quantSize'],
        'normType': combinations['normType'],
        'sampleRate': combinations['sampleRate'],
        'nInterpFac': combinations['nInterpFac'],
        'binsPOct': 1200,
        'minPossiblePitch': 60,
        'removeTaniSegs': -1,
        'varDur': -1,
        'threshold': -1,
        'flatThreshold': -1,
        'maxPauseDur': -1,
        'durMotif': -1,
        'blackDurFact': 1,
        'maxNMotifsPairs': -1,
        'dumpLogs': 1,
        'pitchHop': 0.01,
        'subSeqLen': 800,
        'methodVariant': -1,
        'complexityMeasure': -1
        #'distNormType': 2
        
        }
    params['methodVariant'] = methodVariant
    if musicType == 'hindustani':
      params['pitchHop'] = 0.005
      params['subSeqLen'] = 1600
    elif musicType == 'carnatic':
      params['pitchHop'] = 0.0029
      params['subSeqLen'] = 800
    else:
      print "please specify a valid music type"
      return -1

    outDir = os.path.join(outDir, methodVariants[methodVariant-1])
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    #opening summary file
    summary = open(os.path.join(outDir, baseName+ '_Summary.csv'),"w")
    keyArr = params.keys()
    
    for k in keyArr:
        summary.write("%s\t"%k)
    summary.write("\n")
    
    

    
    cnt = 1
    
    for dist in combinations['distType']:
        params['distType']=dist

        if dist==0:
            DTWType_temp=[-1]
            DTWBand_temp=[0.025]
        else:
            DTWType_temp=combinations['DTWType']
            DTWBand_temp=combinations['DTWBand']
        
        for t in DTWType_temp:
            params['DTWType']=t
            for b in DTWBand_temp:
                params['DTWBand']=b
                
                for n in combinations['normType']:
                    params['normType'] = n
                    
                    if n == 1 or n==7:
                        TSRepType_temp = combinations['TSRepType']
                    else:
                        TSRepType_temp = [0]
                    
                    for rep in TSRepType_temp:
                        params['TSRepType'] = rep
                        if rep==0:
                            quant_temp=[-1]
                        else:
                            quant_temp=combinations['quantSize']
                        
                        for q in quant_temp:
                            params['quantSize'] = q
                            
                            for s in combinations['sampleRate']:
                                params['sampleRate']=s
                                
                                for i in combinations['nInterpFac']:
                                    params['nInterpFac'] = i
                                    for k in keyArr:
                                        summary.write("%s\t"%str(params[k]))
                                    summary.write("\n")
                                    dumpConfigFile(params, os.path.join(outDir, baseName + '_'+ str(cnt) + '.txt'))
                                    cnt = cnt+1
                    
    summary.close()             
                
    print cnt
     


def genConfigFilesICASSP(baseName, outDir, methodVariant=1, musicType = ''):
    combinations= {
                    'distType':[0,1],#0 = Euc, 1 = DTW 
                    'DTWType':[0,1],#0 = DTW (no local constraint), 1 = DTW( local constraint)
                    'DTWBand': [0.05, 0.1, 0.9],
                    'TSRepType': [0,1], #0= raw ts, 1 = quantized ts
                    'quantSize': [12,24], #0= raw ts, 1 = quantized ts                  
                    'normType': [0,1,2,3,4,5], #0=
                    'sampleRate': [10,15,20,25,30],
                    'nInterpFac': [1, 5]
                    }
    
    methodVariants = ['var1', 'var2']
    
    params={
        'distType':combinations['distType'],
        'DTWBand': combinations['DTWBand'],
        'DTWType': combinations['DTWType'],
        'rankRefDistType': 0,
        'TSRepType': combinations['TSRepType'],
        'quantSize': combinations['quantSize'],
        'normType': combinations['normType'],
        'sampleRate': combinations['sampleRate'],
        'nInterpFac': combinations['nInterpFac'],
        'binsPOct': 1200,
        'minPossiblePitch': 60,
        'removeTaniSegs': -1,
        'varDur': -1,
        'threshold': -1,
        'flatThreshold': -1,
        'maxPauseDur': -1,
        'durMotif': -1,
        'blackDurFact': 1,
        'maxNMotifsPairs': -1,
        'dumpLogs': 1,
        'pitchHop': 0.01,
        'subSeqLen': 800,
        'methodVariant': -1
        
        }
    params['methodVariant'] = methodVariant
    if musicType == 'hindustani':
      params['pitchHop'] = 0.005
      params['subSeqLen'] = 1600
    elif musicType == 'carnatic':
      params['pitchHop'] = 0.0029
      params['subSeqLen'] = 800
    else:
      print "please specify a valid music type"
      return -1

    outDir = os.path.join(outDir, methodVariants[methodVariant-1])
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    #opening summary file
    summary = open(os.path.join(outDir, baseName+ '_Summary.csv'),"w")
    keyArr = params.keys()
    
    for k in keyArr:
        summary.write("%s\t"%k)
    summary.write("\n")
    
    

    
    cnt = 1
    
    for dist in combinations['distType']:
        params['distType']=dist

        if dist==0:
            DTWType_temp=[-1]
            DTWBand_temp=[0.025]
        else:
            DTWType_temp=combinations['DTWType']
            DTWBand_temp=combinations['DTWBand']
        
        for t in DTWType_temp:
            params['DTWType']=t
            for b in DTWBand_temp:
                params['DTWBand']=b
                
                for n in combinations['normType']:
                    params['normType'] = n
                    
                    if n == 1:
                        TSRepType_temp = combinations['TSRepType']
                    else:
                        TSRepType_temp = [0]
                    
                    for rep in TSRepType_temp:
                        params['TSRepType'] = rep
                        if rep==0:
                            quant_temp=[-1]
                        else:
                            quant_temp=combinations['quantSize']
                        
                        for q in quant_temp:
                            params['quantSize'] = q
                            
                            for s in combinations['sampleRate']:
                                params['sampleRate']=s
                                
                                for i in combinations['nInterpFac']:
                                    params['nInterpFac'] = i
                                    for k in keyArr:
                                        summary.write("%s\t"%str(params[k]))
                                    summary.write("\n")
                                    dumpConfigFile(params, os.path.join(outDir, baseName + '_'+ str(cnt) + '.txt'))
                                    cnt = cnt+1
                    
    summary.close()             
                
    print cnt
        
    
def dumpConfigFile(params, fileName):

        output = open(fileName,"w")
        
        for key in params.keys():
            output.write("%s: %s\n"%(key, str(params[key])))
            
            
        output.close()
        
def batchgenConfigFilesICASSP(resultDir, NCRIndex, baseName, runs=5, nVariants=2, outcomeID = [], outcomeBaseName = 'out', musicType = ''):
    
    for outid in outcomeID:
      for ii in range(runs):
        resultDirTemp = os.path.join(resultDir, NCRIndex, 'r'+str(ii+1), outcomeBaseName+str(outid))
        
        if not os.path.exists(resultDirTemp):
          os.makedirs(resultDirTemp)
        
        for jj in range(1, nVariants+1): # as we have only two variants
          genConfigFilesICASSP(baseName, resultDirTemp, methodVariant=jj, musicType = musicType)
          
          
def batchgenConfigFiles(resultDir, NCRIndex, baseName, runs=5, nVariants=2, outcomeID = [], outcomeBaseName = 'out', musicType = ''):
    
    for outid in outcomeID:
      for ii in range(runs):
        resultDirTemp = os.path.join(resultDir, NCRIndex, 'r'+str(ii+1), outcomeBaseName+str(outid))
        
        if not os.path.exists(resultDirTemp):
          os.makedirs(resultDirTemp)
        
        for jj in range(1, nVariants+1): # as we have only two variants
          genConfigFiles(baseName, resultDirTemp, methodVariant=jj, musicType = musicType)          
  
