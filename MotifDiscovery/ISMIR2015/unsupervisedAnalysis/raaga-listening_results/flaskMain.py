import flask, flask.views
import os, functools
from flask import request
import numpy as np
import json
import copy
import eval_utils

app = flask.Flask(__name__, static_url_path= '/raaga-survey/static')
# Don't do this
app.secret_key = "sankalp"

response_dir = 'response'
raga_name_map_file = 'raaga_uuid_name_mapp.json'
phase_id_loc_map_file = 'phrase_id_loc_map.json'
musician_uuids_pids_names = 'musician_mid_names_pids.json'

raaga_name_map = json.load(open(raga_name_map_file))
phrase_id_loc_map = json.load(open(phase_id_loc_map_file))
mData = json.load(open(musician_uuids_pids_names))

n_phrases = len(phrase_id_loc_map.keys())


def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                names.append(os.path.join(path, f))
    return names


class  Main(flask.views.MethodView):
    def get(self):
        
        mid = request.args.get('mid')
        
        #TODO what if the user id doesn't match
        if mData.has_key(mid):
            name = mData[mid]['name']
            proceed = True
        else:
            name = 'Unknown user'
            proceed = False
        
        #if evaluators (mid) directory doesn't exist, create it
        m_dir = os.path.join(response_dir, mid)
        if not os.path.isdir(m_dir) and proceed:
            os.makedirs(m_dir)

        #fetching how many responses he has already given
        files = GetFileNamesInDir(m_dir, '.txt')
        nResp = len(files)
        
        return flask.render_template('index.html', n_responses = nResp, mid = mid, name = name, proceed = proceed)


class  Evals(flask.views.MethodView):
    def get(self):
        return flask.render_template('evals.html')   
    
class  End(flask.views.MethodView):
    def get(self):
        return flask.render_template('end.html')      
    
@app.route('/raaga-survey/evals', methods=['GET', 'POST'])
def evals():
    
    mid = request.args.get('mid')
    
    #reading responses
    if request.method=='POST':
        r_index = int(request.args.get('r_index'))
        rating = int(request.form['rating'])
        fname = os.path.join(response_dir, mid, str(r_index)+'.txt')        # response file
        np.savetxt(fname, np.array([rating]), fmt = '%d')
    
    #fetching how many responses he has already given
    eval_indexes = copy.copy(mData[mid]['pids'])
    
    #raaga_status = {}
    #initialize raga status
    #for r in eval_utils.raagas:
    #    raaga_status[raaga_name_map[r]] = 0
    
    m_dir = os.path.join(response_dir, mid)    
    if not os.path.isdir(m_dir):
        print "This is a serious problem with mid %s"%mid
    
    files = GetFileNamesInDir(m_dir, '.txt')
    for f in files:
        #pid = int(os.path.basename(f).split('.')[0])
        #raaga_status[raaga_name_map[phrase_id_loc_map[str(pid)]['raagaId']]]+=1
        eval_indexes.pop(eval_indexes.index(int(os.path.basename(f).split('.')[0])))
    
    eval_indexes = eval_indexes[:1] # showing only 1 line at a time
    f_locs = [phrase_id_loc_map[str(ii)]['path'] for ii in eval_indexes]
    r_names =[raaga_name_map[phrase_id_loc_map[str(ii)]['raagaId']] for ii in eval_indexes]
    
    if len(files) < 100:
        task_in_progress  = True
    else:
        task_in_progress  = False
    #return flask.render_template('evals.html', eval_indexes = eval_indexes, mid=mid, f_locs= f_locs, r_names = r_names, raaga_status=raaga_status, n_responses = len(files), task_in_progress = task_in_progress)
    if task_in_progress:
        return flask.render_template('evals.html', eval_indexes = eval_indexes, mid=mid, f_locs= f_locs, r_names = r_names, n_responses = len(files), task_in_progress = task_in_progress)
    else:
        return flask.render_template('end.html')
   
    
	
app.add_url_rule('/raaga-survey/', 
                    view_func=Main.as_view('index'), 
                    methods = ['GET'])
#app.add_url_rule('/', 
#                view_func=Evals.as_view('eval'), 
#                methods = ['GET', 'POST'])

#app.add_url_rule('/', 
#                view_func=End.as_view('end'), 
#                methods = ['GET'])

app.run(host='0.0.0.0')

