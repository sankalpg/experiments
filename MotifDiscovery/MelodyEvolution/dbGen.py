import codecs
import json, os, sys
import shutil
import numpy as np
import compmusic
import json
from mutagen import easyid3
from compmusic import dunya as dn
from compmusic.dunya import hindustani as hn
from compmusic.dunya import carnatic as ca
from compmusic.dunya import docserver as ds
from compmusic import musicbrainz
import fixpath
from mutagen import easyid3
import shutil
import numpy
import psycopg2 as psy
try:
    from mutagen.mp3 import MP3
except:
    pass

dn.set_token("60312f59428916bb854adaa208f55eb35c3f2f07")

def getStatsFromSelectionFile(filename):
    """
    This function get stats from metadata file where kaustuv annotated the selected files
    """
    lines = codecs.open(filename, 'r', encoding='utf-8').readlines()
    mbid_raga_anot = []
    for line in lines:
        sline = line.split('\t')
        mbid_raga_anot.append((sline[0].strip(), sline[8].strip(), sline[15].strip(), sline[9].strip()))
    
    raga_detail = {}
    for row in mbid_raga_anot:
        if not raga_detail.has_key(row[1]):
            raga_detail[row[1]] = {'cnt':0, 'mbids':[], 'name':row[3]}
        if row[2] == '1':
            raga_detail[row[1]]['mbids'].append(row[0])
            raga_detail[row[1]]['cnt']+=1
    
    raga_cnt = []
    for raga in raga_detail.keys():
        raga_cnt.append([raga, raga_detail[raga]['cnt']])
        
    cnt_arr = [r[1] for r in raga_cnt]
    ind_sort = np.argsort(np.array(cnt_arr))
    ind_sort = ind_sort[::-1]
    sort_ragas = [raga_cnt[ii][0] for ii in ind_sort]
    
    for ii, raga in enumerate(sort_ragas):
        print ii+1, raga_detail[raga]['name'], raga_detail[raga]['cnt']
    
    return mbid_raga_anot

def downloadAudio():
    
    recording = get_recording(recordingid)
    release = get_release(recording["release"][0]["mbid"])
    title = recording["title"]
    artists = " and ".join([a["name"] for a in release["release_artists"]])
    
    name = "%s - %s.mp3" % (artists, title)
    path = os.path.join(location, name)
    

    
def downloadData(tradition, mbid, root_dir, ragaid, artist, release, title, features = ['.mp3', '.pitch', '.tonic'], over_write_files =0 ):
    print root_dir, fixpath.fixpath(artist), fixpath.fixpath(release), fixpath.fixfile(title)
    dirname = os.path.join(root_dir, ragaid, fixpath.fixpath(artist), fixpath.fixpath(release))
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    
    
    print "downloading " + mbid
    for ext in features:
        filename = os.path.join(root_dir, ragaid, fixpath.fixpath(artist), fixpath.fixpath(release), fixpath.fixfile(title))+'_'+mbid + ext
        if os.path.isfile(filename) and over_write_files == 0 and os.path.getsize(filename) > 0:
            continue
        if ext == '.mp3': #which means downloading download_audio
            contents = ds.get_mp3(mbid)
            open(filename, "wb").write(contents)
        elif ext == '.pitch':
            content = ds.file_for_document(mbid, 'pitch', 'pitch', version='noguessunv')
            content = json.loads(content)
            content = np.array(content)
            np.savetxt(filename, content, fmt= '%.7f', delimiter='\t')
        elif ext == '.tonic':
            content = ds.file_for_document(mbid, 'ctonic', 'tonic', version='0.3')
            fid2 = open(filename, 'w')
            fid2.write(content)
            fid2.close()
        else:
            print "This feature extension is not processed %s"%ext
        
    return True
    
    

def buildRagaDb(metadata_file, annotation_file, dst_dir, report_file, music_tradition='', over_write_files = 0, features = ['.mp3', '.pitch', '.tonic']):
    """
    This function creates a directory structure for raga dataset and downloads features and files
    """
    
    if music_tradition == 'hindustani':
        tradition = hn
    elif music_tradition == 'carnatic':
        tradition = ca
        
    report = open(report_file,'w')
        
    metadata = json.load(open(metadata_file, 'r'))
    
    lines = codecs.open(annotation_file, 'r', encoding='utf-8').readlines()
    mbid_raga_anot = []
    for line in lines:
        sline = line.split('\t')
        mbid_raga_anot.append((sline[0].strip(), sline[8].strip(), sline[15].strip(), sline[9].strip()))
    
    raga_detail = {}
    for row in mbid_raga_anot:
        if row[2] == '1':
            if not raga_detail.has_key(row[1]):
                raga_detail[row[1]] = {'cnt':0, 'mbids':[], 'name':row[3]}
            raga_detail[row[1]]['mbids'].append(row[0])
            raga_detail[row[1]]['cnt']+=1
    
    for raga in raga_detail.keys():
        for ii, mbid in enumerate(raga_detail[raga]['mbids']):
            release = metadata[mbid]['recording']['release'][0]['title']
            artist = metadata[mbid]['concert']['release_artists'][0]['name']
            title = metadata[mbid]['recording']['title']
            
            try:
                downloadData(tradition, mbid, dst_dir, raga, artist, release, title, features = features, over_write_files = over_write_files)
                done +=1
                report.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(raga, mbid, artist, release, title, "Yes"))
            except:
                report.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(raga, mbid, artist, release, title, "No"))
                pass
            
    report.close()
        
        
    

def get_filenames_in_dir(dir_name, ext=".mp3"):
    """
    This function gives full pathnames of all the files of a specified extention in dir_name directory
    """
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if f.lower().endswith(ext):
                names.append(os.path.join(path, f))
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

def renameDataFiles(root_dir, metadata_file):
    
    filenames = get_filenames_in_dir(root_dir, '.mp3')
    features = ['.mp3', '.pitch', '.tonic']
    metadata = json.load(open(metadata_file, 'r'))
    for filename in filenames:
        mbid = get_mbid_from_mp3(filename)
        title = fixpath.fixfile(metadata[mbid]['recording']['title'])
        dirname = os.path.dirname(filename)
        fname, ext = os.path.splitext(filename)
        print "processing filename " + filename + "  new title  " + title
        for feat in features:
            try:
                shutil.move(fname+feat, os.path.join(dirname, title)+'_'+mbid+feat)
            except:
                pass

def removeExtraFiles(root_dir, metadata_file):
    
    filenames = get_filenames_in_dir(root_dir, '.mp3')
    metadata = json.load(open(metadata_file, 'r'))
    cnt_del = 0
    for filename in filenames:
        mbid = get_mbid_from_mp3(filename)
        title = fixpath.fixfile(metadata[mbid]['recording']['title'])
        del_filename = os.path.join(os.path.dirname(filename), title+'.mp3')
        if filename == del_filename:
            cnt_del+=1
            print "Deleting file %s"%(del_filename)
            os.remove(del_filename)
    
        #for feat in features:
            #try:
                #shutil.move(fname+feat, os.path.join(dirname, title)+'_'+mbid+feat)
            #except:
                #pass




    ## based on the music tradition select appropriate items
    #if music_tradition == 'hindustani':
        #tradition = hi
    #elif music_tradition == 'carnatic':
        #tradition = ca
    #else:
        #print "Please specify a valid music tradition"
        #exit()

    ## fetching all the mp3 file names in the src file
    #filenames = get_filenames_in_dir(src_dir, '.mp3')

    ## loop over all the file names to copy them to the dst_dir
    #for filename in filenames:
        #fname = os.path.split(filename)[1]  # get the file name
        #fname, ext = os.path.splitext(fname)  # get the file name - extension
        #mbid = get_mbid_from_mp3(filename)  # mbid from the file
        #con_path, con_name, rec_path, rec_name = get_recording_path(mbid, music_tradition, token)
        #print("Downloading/moving recording %s with mbid %s"%(rec_name, mbid))
        #local_rec_path = os.path.join(dst_dir, rec_path)
        #if not os.path.isdir(local_rec_path):  # if the directory is not there, create one
            #os.makedirs(local_rec_path)

        ## finally lets copy the file!!
        #if download_audio:
            #tradition.download_mp3(mbid, os.path.join(local_rec_path))
        #else:
            #shutil.copy(filename, os.path.join(local_rec_path, rec_name + ext))