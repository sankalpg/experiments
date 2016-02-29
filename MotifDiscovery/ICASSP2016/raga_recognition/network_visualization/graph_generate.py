import numpy as np
import sys, os
import json
import matplotlib.pyplot as plt
import time 
import networkx as nx
import psycopg2 as psy


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../library_pythonnew/networkAnalysis'))

DEBUG = False

import networkProcessing as net_pro
import communityCharacterization as comm_char
import constructNetwork as cons_net
import ragaRecognition as raga_recog

def generate_demo_network_raga_recognition(network_file, community_file, output_network, colors = cons_net.colors, colorify = True, mydatabase = ''):
    """
    This function generates a network used as a demo for demonstrating relations between phrases.
    The configuration used to generate this network should ideally be the one that is used for the
    raga recognition task reported in the paper.
    """
    
    #loading the network
    full_net = nx.read_pajek(network_file)
    
    #loading community data
    comm_data = json.load(open(community_file,'r'))
    
    #loading all the phrase data
    comm_char.fetch_phrase_attributes(comm_data, database = mydatabase, user= 'sankalp')
    
    #getting all the communities from which we dont want any node in the graph in the demo
    #obtaining gamaka communities
    gamaka_comms = comm_char.find_gamaka_communities(comm_data)[0]
    
    #obtaining communities with only phrases from one mbid
    one_mbid_comms = comm_char.get_comm_1MBID(comm_data)
    
    #collect phrases which should be removed from the graph
    phrases = []
    for c in gamaka_comms:
        for n in comm_data[c]:
            phrases.append(int(n['nId']))
    for c in one_mbid_comms:
        for n in comm_data[c]:
            phrases.append(int(n['nId']))
    
    print len(phrases)
    
    #removing the unwanted phrases
    full_net = raga_recog.remove_nodes_graph(full_net, phrases)
    
    # colorify the nodes according to raga labels
    if colorify:
        cmd1 = "select raagaId from file where id = (select file_id from pattern where id =%d)"
        con = psy.connect(database='ICASSP2016_10RAGA_2S', user='sankalp') 
        cur = con.cursor()
        for n in full_net.nodes():
            cur.execute(cmd1%(int(n)))
            ragaId = cur.fetchone()[0]
            full_net.node[n]['color'] = ragaId

    #saving the network
    nx.write_gexf(full_net, output_network)
    
    
def generate_artificially_connected_network(network_file, community_file, output_network, colorify = True, mydatabase = ''):
    """  
    Since isolated communities belonging to different ragas are scattered and jumbled up, we attempt to connect them artificaially
    so that they are all grouped together.
    """
    
     #loading the network
    full_net = nx.read_pajek(network_file)
    
    #loading community data
    comm_data = json.load(open(community_file,'r'))
    
    #loading all the phrase data
    comm_char.fetch_phrase_attributes(comm_data, database = mydatabase, user= 'sankalp')
    
    #getting all the communities from which we dont want any node in the graph in the demo
    #obtaining gamaka communities
    gamaka_comms = comm_char.find_gamaka_communities(comm_data)[0]
    
    #obtaining communities with only phrases from one mbid
    one_mbid_comms = comm_char.get_comm_1MBID(comm_data)
    
    print len(full_net.nodes()), len(full_net.edges())
    #collect phrases which should be removed from the graph
    phrases = []
    for c in gamaka_comms:
        for n in comm_data[c]:
            phrases.append(int(n['nId']))
    for c in one_mbid_comms:
        for n in comm_data[c]:
            phrases.append(int(n['nId']))
    
    print len(phrases)
    
    #removing the unwanted phrases
    full_net = raga_recog.remove_nodes_graph(full_net, phrases)
    print len(full_net.nodes()), len(full_net.edges())
    
    #lets remove these phrases from the comm_data as well
    for g in gamaka_comms:
        comm_data.pop(g)
    for o in one_mbid_comms:
        comm_data.pop(o)
    
    #obtaining the raga labels for each community (majority voting ofcourse)
    comm_raga = {}
    raga_comm = {}
    node_cnt = 0
    for comId in comm_data.keys():
        ragaIds = [r['ragaId']  for r in comm_data[comId]]
        
        raga_hist, raga_names = comm_char.get_histogram_sorted(ragaIds)
        comm_raga[comId] = raga_names[0]
        if not raga_comm.has_key(raga_names[0]):
            raga_comm[raga_names[0]] = []
        raga_comm[raga_names[0]].append(comId)

    edge_list = []
    for comId in comm_data.keys():
        raga = comm_raga[comId]
        node_cnt+= len(comm_data[comId])
        for comms_in_raga in raga_comm[raga]:
            if comms_in_raga == comId:
                continue
            #full_net.add_edge(comm_data[comId][0]['nId'], comm_data[comms_in_raga][0]['nId'], weight=0.0)
            edge_list.append((str(comm_data[comId][0]['nId']), str(comm_data[comms_in_raga][0]['nId']), 0.000000001))

    print node_cnt
    print len(full_net.nodes()), len(full_net.edges())
    json.dump(full_net.nodes(), open('pehle.json','w'))
    full_net.add_weighted_edges_from(edge_list)
    json.dump(full_net.nodes(), open('baad.json','w'))
    print len(full_net.nodes()), len(full_net.edges())
    
    # colorify the nodes according to raga labels
    if colorify:
        cmd1 = "select raagaId from file where id = (select file_id from pattern where id =%d)"
        con = psy.connect(database='ICASSP2016_10RAGA_2S', user='sankalp') 
        cur = con.cursor()
        for n in full_net.nodes():
            cur.execute(cmd1%(int(n)))
            ragaId = cur.fetchone()[0]
            full_net.node[n]['color'] = ragaId


    
    
    #saving the network
    nx.write_gexf(full_net, output_network)
    
def convertFormat(sec):
        
    hours = int(np.floor(sec/3600))
    minutes = int(np.floor((sec - (hours*3600))/60))
    
    seconds = sec - ( hours*3600 + minutes*60)
    
    return str(hours) + ':' + str(minutes) + ':' + str(seconds)
    
    
def clipAudio(path, filename, start, end, pattern):
    
    outfile = os.path.join(path,str(pattern) + '.mp3')
    
    cmd = "sox \"%s\" \"%s\" trim %s =%s"%(filename, outfile, convertFormat(start), convertFormat(end))
    os.system(cmd)
    
    return 1    
    
def dump_melodic_phrases_in_network(network_file, output_dir, myDatabase, base_name):
    """
    This function dumps all the mp3 files for the patterns in the 'network' (gexf file)
    """
    
    cmd1 = "select file.filename, pattern.start_time, pattern.end_time from pattern join file on (pattern.file_id = file.id) where pattern.id = %d"
    
    #reading the network
    full_net = nx.read_gexf(network_file)
    
    labels = nx.get_node_attributes(full_net, 'label')
    
    patterns = full_net.nodes()
    
    try:
        con = psy.connect(database=myDatabase, user='sankalp') 
        cur = con.cursor()
        for ii, pattern in enumerate(patterns):
            pattern = labels[pattern]
            cur.execute(cmd1%int(pattern))
            filename, start, end = cur.fetchone()
            clipAudio(output_dir, os.path.join(base_name, filename), start, end, int(pattern))
        
    except psy.DatabaseError, e:
        print 'Error %s' % e
        if con:
            con.rollback()
            con.close()
        sys.exit(1)
    
    if con:
        con.close()
    
    return 1
    