import numpy as np
import sys, os
import json
import matplotlib.pyplot as plt
import time 
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis'))


import networkProcessing as net_pro
import communityCharacterization as comm_char
import constructNetwork as cons_net
import ragaRecognition as raga_rec



"""
This file contains functions to analyse the raga communities. So the motivation for that is:
1) To determine how representative a community is, how much coverage does detected raga communities 
have, have they covered all the recordings of the collection. How many phrases are discovered per 
community and how does it vary over community rank etc.
2) This would give some insights needed for raga recognition task. 
3) This would also give some insights if the criterion for choosing raga communities is good or not

What is to be done:
1) #number of unique recordings in the communities, across communities of the same raga
2) Distribution of nodes in communities of differenr ragas.
3) #number of unique compositions in the communities
"""

def plot_item_distribution_per_community(fileListFile, thresholdBin, pattDistExt, myDatabase = '', myUser = '', force_build_network =0, top_N_com = 10, weighted_dist = 1):
    """
    This function determines the distribution of items (mbid or raga_id or comp id) across communities. 
    The communities selected for this analysis is determined by filtering.
    """
    
    #construncting graph
    t1 = time.time()
    wghtd_graph_filename = 'graph_temp'+'_'+str(thresholdBin)
    if force_build_network or not os.path.isfile(wghtd_graph_filename):
        cons_net.constructNetwork_Weighted_NetworkX(fileListFile, wghtd_graph_filename , thresholdBin, pattDistExt, 0 , -1)
    
    full_net = nx.read_pajek(wghtd_graph_filename)
    t2 = time.time()
    print "time taken = %f"%(t2-t1)
    
    
    comm_filename = 'comm'+'_'+str(thresholdBin)+'.community'
    net_pro.detectCommunitiesInNewtworkNX(wghtd_graph_filename, comm_filename)
    
    comm_rank_filename  = 'comm'+'_'+str(thresholdBin)+'.communityRank'
    comm_char.rankCommunities(comm_filename, comm_rank_filename, myDatabase = myDatabase, myUser = myUser)
    
    raga_comms  = comm_char.select_topN_community_per_raaga(comm_rank_filename, top_N_com)
    
    comm_data = json.load(open(comm_filename,'r'))
    comm_char.fetch_phrase_attributes(comm_data, database = myDatabase, user= myUser)
    
    #computing the unique mbids, raga_ids in the dataset
    raga_mbid = raga_rec.get_mbids_raagaIds_for_collection(myDatabase, myUser)
    raga_list = np.array([r[0] for r in raga_mbid])
    u_ragas = np.unique(raga_list)
    u_mbids = []
    for raga in u_ragas:
        ind_raga = np.where(raga_list == raga )[0]
        u_mbids.extend(np.unique([raga_mbid[r][1] for r in ind_raga]).tolist())
    
    print len(u_mbids)
    print len(u_ragas)
    u_mbids = np.array(u_mbids)
    
    dist_mbid = np.zeros((len(u_mbids), top_N_com*len(u_ragas)))
    dist_raga = np.zeros((len(u_ragas), top_N_com*len(u_ragas)))
    
    cnt_col = 0
    for raga in u_ragas:
        for raga_comm in raga_comms[raga]:
            comId = raga_comm['comId']
            for node in comm_data[str(comId)]:
                ind_mbid = np.where(u_mbids == node['mbid'])[0]
                ind_raga = np.where(u_ragas == node['ragaId'])[0]
                
                if weighted_dist == 1:
                    dist_mbid[ind_mbid, cnt_col]+=1
                    dist_raga[ind_raga, cnt_col]+=1
                else:
                    dist_mbid[ind_mbid, cnt_col]=1
                    dist_raga[ind_raga, cnt_col]=1
            cnt_col+=1
    
    np.save(wghtd_graph_filename+'.mbid_dist', dist_mbid)
    np.save(wghtd_graph_filename+'.raga_dist', dist_raga)
    
    #mbid_per_comm = np.sum(dist_mbid, axis =0)
    #raga_per_comm = np.sum(dist_raga, axis =0)
    
    
    
    
    
    
    
