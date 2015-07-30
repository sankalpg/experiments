#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2 as psy
import sys, os
import numpy as np
import pickle

try:
    from mutagen.mp3 import MP3
except:
    pass



myUser = 'sankalp'
myDatabase = 'motifDB_CONF1'

root_path = '/media/Data/Datasets/MotifDiscovery_Dataset/CompMusic/'


def computeDutationSong(audiofile): 
    
    filename, ext = os.path.splitext(audiofile)

    if  ext=='.mp3':
        audio = MP3(audiofile)
        duration = audio.info.length
    elif ext=='.wav':
        duration = ES.MetadataReader(filename = audiofile)()[7]
            
    return duration


def generateClipsOfEvalData(patternInfoFile, outputPath):
    
    cmd1 = "select filename from file where id=%d"
    cmd2 = "select file_id from pattern where id=%d"
    cmd3 = "select start_time, end_time from pattern where id=%d"
    
    #reading patternInfoFIle to get ids of all the patterns
    patternInfo = np.loadtxt(patternInfoFile).astype(np.int)
    patternInfo = np.ndarray.flatten(patternInfo)
    
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        print len(patternInfo)
        for pattern in patternInfo:
            cur.execute(cmd2%pattern)
            file_id = int(cur.fetchone()[0])
            cur.execute(cmd1%file_id)
            filename = root_path + cur.fetchone()[0]
            cur.execute(cmd3%pattern)
            timeInfo = cur.fetchone()
            start = float(timeInfo[0])
            end = float(timeInfo[1])
            
            ### TODO this is a hack because there was some duplication in the mbids but not in audio. There fore we get mismatch eventually in audio and features. TO do it correct way remove these duplications from the data. As a hack I just replace the audiofile name string to make it work temporarily
            if filename.count("_(1)"):
                filename = filename.replace("_(1)","")
            if filename.count("_(2)"):
                filename = filename.replace("_(2)","")
            if filename.count("_(3)"):
                filename = filename.replace("_(3)","")
            if filename.count("_(4)"):
                filename = filename.replace("_(4)","")
            
            #check if the start and end are within the expected length of the file
            duration = computeDutationSong(filename)
            if start < duration and end < duration:
                try:
                    clipAudio(outputPath, filename, start, end, str(pattern))
                except:
                    fid = open("errorLogs.txt", 'a')
                    fid.write("############ BUG2###########\n")
                    fid.write("File %s\n"%filename)
                    fid.close()
            else:
                fid = open("errorLogs.txt", 'a')
                fid.write("############ BUG1###########\n")
                fid.write("File %s\n"%filename)
                fid.close()
                
            
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
    
    return 1

def clipAudio(path, filename, start, end, pattern):
    
    outfile = path + str(pattern) + '.mp3'
    
    cmd = "sox \"%s\" \"%s\" trim %s =%s"%(filename, outfile, convertFormat(start), convertFormat(end))
    os.system(cmd)
    
    return 1
    
    
    
def convertFormat(sec):
        
    hours = int(np.floor(sec/3600))
    minutes = int(np.floor((sec - (hours*3600))/60))
    
    seconds = sec - ( hours*3600 + minutes*60)
    
    return str(hours) + ':' + str(minutes) + ':' + str(seconds)


def checkClipsAvailability(patternInfoFile, outputPath):
    
    patternInfo = np.loadtxt(patternInfoFile).astype(np.int)
    patternInfo = np.unique(np.ndarray.flatten(patternInfo))
    
    for pattern in patternInfo:
        filename = outputPath + str(pattern) + '.mp3'
        if not os.path.exists(filename):
            print filename
    
    