#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2 as psy
import sys, os
from mutagen import easyid3
import numpy as np

myUser = 'sankalp'
myDatabase = 'motifDB_CONF1'

def getAllMBIDs():
    
    cmd1 = "select MBID from file where hasseed=1"
    mbidList = [] 
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1)
        mbids = cur.fetchall()
        mbidList = [x[0] for x in mbids]
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
        
    return mbidList

def getFilenameFromMBID(MBID):
    
    cmd1 = "select filename from file where mbid='%s'"
    
    filename=""
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1%str(MBID))
        filename, = cur.fetchone()
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
        
    return filename
    
    
    
def getSeeds4MBID(MBID):
    
    cmd1 = "select id from file where mbid ='%s' and hasseed=1"
    cmd2 = "select * from pattern where file_id=%d and isseed=1"
    
    seedData=[]
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1%str(MBID))
        file_id, = cur.fetchone()
        cur.execute(cmd2%file_id)
        seedData = cur.fetchall()
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
        
    return seedData

def getSearchedPatterns(patternID):
    
    cmd1 = "select * from match where source_id =%d and version=0"
    
    patternID = int(patternID)
    searchedData=[]
    try:
        con = psy.connect(database=myDatabase, user=myUser) 
        cur = con.cursor()
        cur.execute(cmd1%patternID)
        searchedData = cur.fetchall()
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
        
    return searchedData

