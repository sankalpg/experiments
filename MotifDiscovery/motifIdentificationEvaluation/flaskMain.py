import flask, flask.views
import os, functools
#import fetchDataPSQL as psqlGet
from flask import request
import numpy as np

app = flask.Flask(__name__)
# Don't do this
app.secret_key = "sankalp"

nVersions =4
nPatternPerSeed = 10

class  Main(flask.views.MethodView):
    def get(self):
        #read the evaluation data
        EvalInfo = np.loadtxt('evaluation.txt')
        indIncomplete = np.where(EvalInfo[1:,:]==-1)[0]
        
        totalSearchPatterns = EvalInfo[1:,:].size
        searchPatternsDone = totalSearchPatterns - indIncomplete.size

        indIncomplete = np.where(EvalInfo[0,:]==-1)[0]
        totalSeedPatterns = EvalInfo[0,:].size
        seedPatternsDone = totalSeedPatterns - indIncomplete.size
        

        return flask.render_template('index.html', seedPatternsDone = seedPatternsDone, totalSeedPatterns = totalSeedPatterns, searchPatternsDone=searchPatternsDone, totalSearchPatterns=totalSearchPatterns)

class  Seed(flask.views.MethodView):
    def get(self):
        return flask.render_template('seed.html')


class  Version(flask.views.MethodView):
    def get(self):
        return flask.render_template('version.html')

class  Search(flask.views.MethodView):
    def get(self):
        return flask.render_template('search.html')
    
    
@app.route('/searchPage', methods=['GET', 'POST'])
def searchPage():
    
    seedInd = int(request.args.get('seedIndex'))-1

    versionInd = int(request.args.get('version'))-1
    
    if request.method=='POST':
        searchInd = int(request.args.get('searchIndex'))-1
        rating = int(request.form['rating'])
        EvalInfo = np.loadtxt('evaluation.txt')
        EvalInfo[1+(nPatternPerSeed*versionInd) + searchInd,seedInd]=rating
        np.savetxt('evaluation.txt',EvalInfo)
        
    #read the evaluation data
    patternInfo = np.loadtxt('pattern.txt')
    EvalInfo = np.loadtxt('evaluation.txt')
    
    doneArray = np.ones(nPatternPerSeed).astype(np.int)

    EvalInfo = EvalInfo[1+(versionInd*nPatternPerSeed):1+((versionInd+1)*nPatternPerSeed),seedInd]

    indIncomplete = np.where(EvalInfo==-1)[0]

    doneArray[indIncomplete] = 0
    
    return flask.render_template('search.html', searchPatterns = (np.arange(nPatternPerSeed)+1).tolist(), progress = doneArray, seedIndex=seedInd+1, version = versionInd+1)
    

@app.route('/versionPage', methods=['GET', 'POST'])
def versionPage():
    
    seedInd = int(request.args.get('seedIndex'))-1
    
    if request.method=='POST':
        rating = int(request.form['rating'])
        EvalInfo = np.loadtxt('evaluation.txt')
        EvalInfo[0,seedInd]=rating
        np.savetxt('evaluation.txt',EvalInfo)
        
    #read the evaluation data
    patternInfo = np.loadtxt('pattern.txt')
    EvalInfo = np.loadtxt('evaluation.txt')
    
    doneArray = np.zeros(nVersions).astype(np.int)
    for ii in range(nVersions):
        if np.min(EvalInfo[1+(ii*nPatternPerSeed):1+((ii+1)*nPatternPerSeed),seedInd])==-1:
            doneArray[ii]=0
        else:
            doneArray[ii]=1
            
    if EvalInfo[0,seedInd]==-1:
        seedDone=0
    else:
        seedDone=1
    
    return flask.render_template('version.html', versionNames = (np.arange(nVersions)+1).tolist(), progress = doneArray, seedIndex=seedInd+1, seedDone = seedDone)


@app.route('/seedPage', methods=['GET'])
def seedPage():
    
    #read the evaluation data
    patternInfo = np.loadtxt('pattern.txt')
    EvalInfo = np.loadtxt('evaluation.txt')
    
    
    #compute for which seed patterns evaluation is completely done
    doneArray = np.zeros((nVersions,patternInfo.shape[1])).astype(np.int)
    for ii in range(patternInfo.shape[1]):
        for jj in range(nVersions):
            if np.min(EvalInfo[1+(jj*nPatternPerSeed):1+((jj+1)*nPatternPerSeed),ii])==-1:
                doneArray[jj,ii]=0
            else:
                doneArray[jj,ii]=1
            
    overallDone = np.min(doneArray,axis=0)  
    return flask.render_template('seed.html', seedIndexes = (np.arange(patternInfo.shape[1])+1).tolist(), progress = overallDone)
   
	
app.add_url_rule('/', 
				view_func=Main.as_view('index'), 
				methods = ['GET'])
app.add_url_rule('/', 
                view_func=Seed.as_view('seed'), 
                methods = ['GET'])
app.add_url_rule('/', 
                view_func=Version.as_view('version'), 
                methods = ['GET', 'POST'])
app.add_url_rule('/', 
                view_func=Search.as_view('search'), 
                methods = ['GET', 'POST'])


app.debug = True

app.run()

