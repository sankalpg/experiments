import flask, flask.views
from flask import send_from_directory, Blueprint
import os, functools
import fetchDataPSQL as psqlGet
from flask import request

app = flask.Flask(__name__)
# Don't do this
app.secret_key = "sankalp"


class  Main(flask.views.MethodView):
    def get(self):
        mbids = psqlGet.getAllMBIDs()
        return flask.render_template('index.html', mbids = mbids)

class  Seed(flask.views.MethodView):
    def get(self):
        return flask.render_template('seed.html')

class  Search(flask.views.MethodView):
    def get(self):
        return flask.render_template('search.html')    

@app.route('/base/<path:filename>')
def base_static(filename):
    print filename
    return send_from_directory('/media/Data/Datasets/MotifDiscovery_Dataset/CompMusic/', filename)

blueprint = Blueprint('site', __name__, static_url_path='/static/audio', static_folder='/media/Data/Datasets/MotifDiscovery_Dataset/CompMusic')


@app.route('/searchPage', methods=['GET'])  
def searchPage():
    error = None
    patternID = request.args.get('patternID', '')
    mbid = request.args.get('mbid', '')
    print patternID
    print mbid
    searchData = psqlGet.getSearchedPatterns(patternID)
    return flask.render_template('search.html', searchData=searchData, mbid =mbid)

@app.route('/seedPage', methods=['GET'])
def seedPage():
    error = None
    mbid = request.args.get('mbid', '')
    data = psqlGet.getSeeds4MBID(mbid)
    filename = psqlGet.getFilenameFromMBID(mbid)
    seedData = []
    for ii in xrange(0,len(data),2):
        
        seedData.append((data[ii][0], data[ii][2], data[ii][3], data[ii+1][0], data[ii+1][2], data[ii+1][3]))
        
    return flask.render_template('seed.html', seedData=seedData, mbid=mbid, filename=filename)
   
	
app.add_url_rule('/', 
				view_func=Main.as_view('index'), 
				methods = ['GET'])
app.add_url_rule('/', 
                view_func=Seed.as_view('seed'), 
                methods = ['GET'])
app.add_url_rule('/', 
                view_func=Search.as_view('search'), 
                methods = ['GET'])

app.debug = True
app.register_blueprint(blueprint)
app.run()

