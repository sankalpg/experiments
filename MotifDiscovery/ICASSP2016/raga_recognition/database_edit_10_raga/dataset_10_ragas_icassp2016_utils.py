from __future__ import unicode_literals
import codecs
import json, os, sys
import numpy as np
import compmusic
import json
from compmusic import dunya as dn
from compmusic.dunya import hindustani as hn
from compmusic.dunya import carnatic as ca
from compmusic.dunya import docserver as ds
from compmusic import musicbrainz
import fixpath
from mutagen import easyid3
import shutil


dn.set_token("60312f59428916bb854adaa208f55eb35c3f2f07")



lines = open('/home/sankalp/Work/Work_PhD/library_pythonnew/utils/CarnaticInfo/carnaticMBIDLocationMapping_3_March_2015.txt','r').readlines()
location = {}
for line in lines:
	splitLine = line.split()
	location[splitLine[1].strip()] = splitLine[0].strip()
	
def getDataPerMBID(MBID, path, features=[{'name':'mp3', 'extension':'.mp3'}], collection = 'carnatic'):
	
	if collection == 'hindustani':
		tradition = hn
	elif collection == 'carnatic':
		tradition = ca
		
	for ftr in features:
		if ftr['name']=='mp3':
			#src = location[MBID]
			#cmd = "scp -r kora:'%s' '%s'"%(src, path)
			#os.system(cmd)
			tradition.download_mp3(MBID, path)


def download_audio_files(mbid_raga_file, out_dir, logfile, collection = 'carnatic'):
	
	if collection == 'hindustani':
		tradition = hn
	elif collection == 'carnatic':
		tradition = ca
		
	#lines = open(mbid_raga_file,'r').readlines()
	#raga_data = {}
	#for ii, line in enumerate(lines):
		#sline = line.split('\t')
		#mbid = sline[0].strip()
		#raga_id = sline[1].strip()
		#try:
			#rec_data = tradition.get_recording(mbid)
			#con_data = tradition.get_concert(rec_data['concert'][0]['mbid'])
			#if not raga_data.has_key(raga_id):
				#raga_data[raga_id] = {}
			#if len(con_data['concert_artists'])>0:
				#raga_data[raga_id][mbid] = {'concert': con_data['title'], 'title':rec_data['title'], 'artist': con_data['concert_artists'][0]['name']}
			#else:
				#raga_data[raga_id][mbid] = {'concert': con_data['title'], 'title':rec_data['title'], 'artist': 'Unknown'}
		#except:
			#print mbid
	
	#json.dump(raga_data, open('raga_data.json','w'))
	
	raga_data = json.load(open('raga_data.json','r'))
	fid = open(logfile,'w')
	for raga in raga_data.keys():
		for mbid in raga_data[raga].keys():
			#print "Processing", mbid, raga_data[raga][mbid]
			outPath = fixpath.fixpath(os.path.join(out_dir,raga, raga_data[raga][mbid]['artist'],raga_data[raga][mbid]['concert'], raga_data[raga][mbid]['title']))
			#outPath = fixpath.fixpath(os.path.join(out_dir,raga, raga_data[raga][mbid]['artist'],raga_data[raga][mbid]['concert']))
			if not os.path.exists(outPath):
				os.makedirs(outPath)
			try:
				#if not os.path.isfile(outPath+'/'+location[mbid].split('/')[-1]):
				getDataPerMBID(mbid, outPath)
				fid.write("%s\t%s\t%d\n"%(mbid,outPath, 1))
			except:
				print mbid
				fid.write("%s\t%s\t%d\n"%(mbid,outPath, 0))
			
			#has_src = 0
			#try:
				#src = location[mbid]
				#has_src = 1
			#except:
				#print "CAnnot find file for this MBID: %s\n"%mbid
				
			#if has_src ==1 and os.path.isfile(outPath+'/'+src.split('/')[-1]):
				#fid.write("%s\t%s\t%d\n"%(mbid,outPath+'/'+src.split('/')[-1], 1))
			#else:
				#fid.write("%s\t%s\t%d\n"%(mbid,outPath+'/'+	src.split('/')[-1], 0))
	#fid.close()	

def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            #if filter in f.lower():
            if f.endswith(filter):
                #print(path+"/"+f)
                #print(path)
                #ftxt.write(path + "/" + f + "\n")
                names.append(path + "/" + f)
                
    return names

def get_mbid_from_mp3(mp3_file):
    """
    fetch MBID form an mp3 file
    """
    try:
        mbid = easyid3.ID3(mp3_file)['UFID:http://musicbrainz.org'].data
    except:
        print "problem reading mbid for file %s\n" % mp3_file
        raise
    return mbid

def fix_file_names(root_dir, raga_data):
	
	raga_data = json.load(open('raga_data.json','r'))
	
	mbid_title = {}
	
	for raga in raga_data.keys():
		for mbid in raga_data[raga].keys():
			mbid_title[mbid] = raga_data[raga][mbid]['title']
	
	filenames = GetFileNamesInDir(root_dir, '.mp3')
	
	for filename in filenames:
		dir_name = os.path.dirname(filename)
		mbid = get_mbid_from_mp3(filename)
		try:
			os.system("mv '%s' '%s'"%(filename, os.path.join(dir_name, fixpath.fixfile(mbid_title[mbid])+'.mp3')))
		except:
			print filename
			
def copy_data(src_dir, dst_dir, log_file, exts = ['.pitch', '.tonic', '.tonicFine', '.taniSegKNN', '.pitchSilIntrpPP']):
	
	fid = open(log_file, 'w')
	
	filenames_src = GetFileNamesInDir(src_dir, '.mp3')
	filenames_dst = GetFileNamesInDir(dst_dir, '.mp3')
	
	mbid_src_loc = {}
	for filename in filenames_src:
		mbid_src = get_mbid_from_mp3(filename)
		mbid_src_loc[mbid_src] = filename
	
	for filename in filenames_dst:
		mbid_dst = get_mbid_from_mp3(filename)
		if mbid_src_loc.has_key(mbid_dst):
			fname, ext = os.path.splitext(filename)
			fname_src, ext = os.path.splitext(mbid_src_loc[mbid_dst])
			for e in exts:
				shutil.copy(fname_src+e, fname +e )
		else:
			fid.write("%s"%filename)
	fid.close()
				

def map_mbid_title_vignesh(mbid_raga_file, outfile):
	
	lines = open(mbid_raga_file,'r').readlines()
	fid = open(outfile,'w')
	raga_data = json.load(open('raga_data.json','r'))
	for line in lines:
		mbid = line.split('\t')[0].strip()
		raga = line.split('\t')[1].strip()
		try:
			fid.write("%s\t%s"%(mbid, raga_data[raga][mbid]['title']))
		except:
			fid.write("%s\t%s"%(mbid, 'Unknown'))
		fid.write('\n')
	fid.close()
	
		
	