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

def generate_demo_network_raga_recognition(network_file, community_file, output_network, colors = cons_net.colors, colorify = True):
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
    comm_char.fetch_phrase_attributes(comm_data, database = 'ISMIR2015_10RAGA_TONICNORM', user= 'sankalp')
    
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
        con = psy.connect(database='ISMIR2015_10RAGA_TONICNORM', user='sankalp') 
        cur = con.cursor()
        for n in full_net.nodes():
            cur.execute(cmd1%(int(n)))
            ragaId = cur.fetchone()[0]
            full_net.node[n]['color'] = colors[ragaId]

    #saving the network
    nx.write_gexf(full_net, output_network)
    