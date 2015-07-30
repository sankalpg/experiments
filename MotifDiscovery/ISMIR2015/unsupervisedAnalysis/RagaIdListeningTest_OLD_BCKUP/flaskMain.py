import flask, flask.views
import os, functools
from flask import request
import numpy as np

app = flask.Flask(__name__)
# Don't do this
app.secret_key = "sankalp"

nVersions =4
nPatternPerSeed = 10

experiment_root = 'raaga_id_exp'
responsefile = 'musician_response.txt'

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


class  Main(flask.views.MethodView):
    def get(self):
        mid = request.args.get('mid')
        
        print "Processing musician with id = %s"%mid
        
        #fetching how many responses he has already given
        m_dir = os.path.join(experiment_root, mid)
        
        if not os.path.isdir(m_dir):
            os.makedirs(m_dir)
        
        files = GetFileNamesInDir(m_dir, '.txt')
        nResp = len(files)
        return flask.render_template('index.html', n_responses = nResp, mid = mid)

class  Seed(flask.views.MethodView):
    def get(self):
        return flask.render_template('seed.html')


class  Version(flask.views.MethodView):
    def get(self):
        return flask.render_template('version.html')

class  Search(flask.views.MethodView):
    def get(self):
        return flask.render_template('search.html')

class  Evals(flask.views.MethodView):
    def get(self):
        return flask.render_template('evals.html')    
    
    
@app.route('/searchPage', methods=['GET', 'POST'])
def searchPage():
    
    seedInd = int(request.args.get('seedIndex'))-1

    versionInd = int(request.args.get('version'))-1
    
    if request.method=='POST':
        searchInd = int(request.args.get('searchIndex'))-1
        rating = int(request.form['rating'])
        print seedInd, versionInd, searchInd, rating
        EvalInfo = np.loadtxt(annotationFile)
        EvalInfo[2+(nPatternPerSeed*versionInd) + searchInd,seedInd]=rating
        np.savetxt(annotationFile,EvalInfo)
        
    #read the evaluation data
    patternInfo = np.loadtxt(patternInfoFile).astype(np.int)
    EvalInfo = np.loadtxt(annotationFile)
    
    doneArray = np.ones(nPatternPerSeed).astype(np.int)

    EvalInfo = EvalInfo[2+(versionInd*nPatternPerSeed):2+((versionInd+1)*nPatternPerSeed),seedInd]

    indIncomplete = np.where(EvalInfo==-1)[0]

    doneArray[indIncomplete] = 0
    
    searchPatternsIds = patternInfo[2+(versionInd*nPatternPerSeed):2+((versionInd+1)*nPatternPerSeed),seedInd]
    
    seedPatternId = patternInfo[1,seedInd]
    
    return flask.render_template('search.html', searchPatterns = (np.arange(nPatternPerSeed)+1).tolist(), progress = doneArray, seedIndex=seedInd+1, version = versionInd+1, searchPatternsIds =searchPatternsIds, seedPatternId=seedPatternId )
    

@app.route('/versionPage', methods=['GET', 'POST'])
def versionPage():
    
    seedInd = int(request.args.get('seedIndex'))-1
    
    if request.method=='POST':
        rating = int(request.form['rating'])
        EvalInfo = np.loadtxt(annotationFile)
        EvalInfo[1,seedInd]=rating
        np.savetxt(annotationFile,EvalInfo)
        
    #read the evaluation data
    patternInfo = np.loadtxt(patternInfoFile).astype(np.int)
    EvalInfo = np.loadtxt(annotationFile)
    
    doneArray = np.zeros(nVersions).astype(np.int)
    for ii in range(nVersions):
        if np.min(EvalInfo[2+(ii*nPatternPerSeed):2+((ii+1)*nPatternPerSeed),seedInd])==-1:
            doneArray[ii]=0
        else:
            doneArray[ii]=1
            
    if EvalInfo[1,seedInd]==-1:
        seedDone=0
    else:
        seedDone=1
        
    seedPattern = patternInfo[1,seedInd]
    seedPair = patternInfo[0,seedInd]
    
    return flask.render_template('version.html', versionNames = (np.arange(nVersions)+1).tolist(), progress = doneArray, seedIndex=seedInd+1, seedDone = seedDone, seedPattern = seedPattern, seedPair=seedPair)


@app.route('/seedPage', methods=['GET'])
def seedPage():
    
    #read the evaluation data
    patternInfo = np.loadtxt(patternInfoFile).astype(np.int)
    EvalInfo = np.loadtxt(annotationFile)
    
    
    #compute for which seed patterns evaluation is completely done
    doneArray = np.zeros((nVersions,patternInfo.shape[1])).astype(np.int)
    for ii in range(patternInfo.shape[1]):
        for jj in range(nVersions):
            if np.min(EvalInfo[2+(jj*nPatternPerSeed):2+((jj+1)*nPatternPerSeed),ii])==-1:
                doneArray[jj,ii]=0
            else:
                doneArray[jj,ii]=1
            
    overallDone = np.min(doneArray,axis=0)  
    return flask.render_template('seed.html', seedIndexes = (np.arange(patternInfo.shape[1])+1).tolist(), progress = overallDone)
   
@app.route('/evals', methods=['GET', 'POST'])
def evals():
    
    mid = request.args.get('mid')
    print mid, experiment_root
    
    #reading responses
    if request.method=='POST':
        r_index = int(request.args.get('r_index'))
        rating = int(request.form['rating'])
        fname = os.path.join(experiment_root, mid, str(r_index)+'.txt')
        np.savetxt(fname, np.array([rating]))
    
    #fetching how many responses he has already given
    eval_indexes = (np.arange(1, 100+1).tolist())
    m_dir = os.path.join(experiment_root, mid)    
    if not os.path.isdir(m_dir):
        print "This is a serious problem now"
    files = GetFileNamesInDir(m_dir, '.txt')
    for f in files:
        eval_indexes.pop(eval_indexes.index(int(os.path.basename(f).split('.')[0])))
        
    return flask.render_template('evals.html', eval_indexes = eval_indexes, mid=mid)
   
    
	
app.add_url_rule('/', 
                    view_func=Main.as_view('index'), 
                    methods = ['GET'])
app.add_url_rule('/', 
                view_func=Evals.as_view('eval'), 
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

