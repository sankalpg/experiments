from __future__ import unicode_literals
from flask import Flask, request, jsonify, current_app
import requests
import psycopg2 as psy
from functools import wraps
import sys
import os.path
import json
from flask.ext.cors import CORS
from compmusic import dunya as dn

auth_token = "60312f59428916bb854adaa208f55eb35c3f2f07"

app = Flask(__name__)
CORS(app)

con = psy.connect(database='ICASSP2016_10RAGA_2S', user='sankalp') 
cur = con.cursor()

@app.route('/')
def index():
    return "raga phrase demo"

def support_jsonp(f):
    """Wraps JSONified output for JSONP"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            content = str(callback) + '(' + str(f(*args, **kwargs)) + ')'
            return current_app.response_class(content, mimetype='application/javascript')
        else:
            return f(*args, **kwargs)

    return decorated_function


@app.route('/get_phrase_data', methods=['GET', 'POST'])
@support_jsonp
def get_phrase_data():
    nid = int(request.args.get('nid'))
    cmd = "select file.mbid, file.raagaid, file.tonic, pattern.start_time, pattern.end_time from pattern join file on (file.id = pattern.file_id) where pattern.id = %d"
    cur.execute(cmd%(nid))
    mbid, raaga, tonic, start, end = cur.fetchone()
    out = {'mbid':mbid, 'ragaid': raaga, 'start':start, 'end':end, 'tonic': tonic}
    
    return jsonify(**out)

@app.route('/get_rec_data', methods=['GET', 'POST'])
@support_jsonp
def get_rec_data():
    mbid = request.args.get('mbid')
    url = "http://dunya.compmusic.upf.edu/api/carnatic/recording/" + mbid
    data = requests.get(url, headers = {"Authorization":"Token " + auth_token})
    return jsonify(**(data.json()))


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host= '0.0.0.0', debug = True)    
