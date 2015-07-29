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

def plot_item_distribution_per_community(fileListFile, out_dir, thresholdBin, pattDistExt, myDatabase = '', myUser = '', force_build_network =0, top_N_com = 10, comm_types = 'raga', network_wght_type = -1):
    """
    This function determines the distribution of items (mbid or raga_id or comp id) across communities. 
    The communities selected for this analysis is determined by filtering.
    
    comm_types: determine what type of communities are used for the analysis of the distribution ('raga' or 'all' or 'compositions')
    """
    
    #construncting graph
    t1 = time.time()
    wghtd_graph_filename = 'graph_temp'+'_'+str(thresholdBin)
    if force_build_network or not os.path.isfile(wghtd_graph_filename):
        cons_net.constructNetwork_Weighted_NetworkX(fileListFile, wghtd_graph_filename , thresholdBin, pattDistExt, network_wght_type , -1) #we dont apply any disparity filtering here!!!!!!!
        #TODO disparity filtering didn't prove much useful in the ISMIR 2015 analysis. Maybe worth a shot again...try it later point in time
    
    full_net = nx.read_pajek(wghtd_graph_filename)
    t2 = time.time()
    print "time taken = %f"%(t2-t1)
    
    #detecting communities 
    comm_filename = 'comm'+'_'+str(thresholdBin)+'.community'
    net_pro.detectCommunitiesInNewtworkNX(wghtd_graph_filename, comm_filename)
    
    #Ranking communities based on which we can later decide raga communities
    comm_rank_filename  = 'comm'+'_'+str(thresholdBin)+'.communityRank'
    comm_char.rankCommunities(comm_filename, comm_rank_filename, myDatabase = myDatabase, myUser = myUser)
    
    #fetching all the relevant attributes for each phrase and all the communities
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
    u_mbids = np.array(u_mbids)
    
    #determining the comm ids to be used for the distribution anlaysis.
    comIds = []
    if comm_types == 'raga':
        #selecting top N communities per raga
        raga_comms  = comm_char.select_topN_community_per_raaga(comm_rank_filename, top_N_com)
        for raga in u_ragas:
            for raga_comm in raga_comms[raga]:
                comIds.append(raga_comm['comId'])
    
    #Initializing arrays to store the dists    
    mbid_vs_comm = np.zeros((len(u_mbids), top_N_com*len(u_ragas)))
    mbid_vs_comm_wght = np.zeros((len(u_mbids), top_N_com*len(u_ragas)))    #weighted
    
    ragaid_vs_comm = np.zeros((len(u_ragas), top_N_com*len(u_ragas)))
    ragaid_vs_comm_whtd = np.zeros((len(u_ragas), top_N_com*len(u_ragas)))  #weighted
    
    
    cnt_col = 0
    mbid_centroid_vs_comm = []
    ragaid_centroid_vs_comm = []
    for comId in comIds:
        mbids_in_comm = []
        ragaids_in_comm = []
        for node in comm_data[str(comId)]:
            ind_mbid = np.where(u_mbids == node['mbid'])[0]
            ind_raga = np.where(u_ragas == node['ragaId'])[0]
            
            mbids_in_comm.append(node['mbid'])
            ragaids_in_comm.append(node['ragaId'])
            
            mbid_vs_comm[ind_mbid, cnt_col]=1
            mbid_vs_comm_wght[ind_mbid, cnt_col]+=1
            ragaid_vs_comm[ind_raga, cnt_col]=1            
            ragaid_vs_comm_whtd[ind_raga, cnt_col]+=1            
        cnt_col+=1
        mbid_centroid_vs_comm.append(comm_char.fileCentroid(comm_char.get_histogram_sorted(mbids_in_comm)[0]))
        ragaid_centroid_vs_comm.append(comm_char.fileCentroid(comm_char.get_histogram_sorted(ragaids_in_comm)[0]))
    
    N_mbid_vs_comm = np.sum(mbid_vs_comm,axis=0)
    N_ragaid_vs_comm = np.sum(ragaid_vs_comm,axis=0)
    
    comms_vs_mbid = np.sum(mbid_vs_comm,axis=1)
    
    N_mbid_comm_distribution, n_comms = comm_char.get_histogram_sorted(comms_vs_mbid)
    
    output_dir = os.path.join(out_dir, 'network_wght_%d_Tshld_%d_NoDispFilt_NComms_%d'%(network_wght_type, thresholdBin, top_N_com), comm_types+'_comms_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #saving the plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fsize = 22
    fsize2 = 16
    font="Times New Roman"
    
    
    plt.imshow(mbid_vs_comm, interpolation="nearest")
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("MBID", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'mbid_vs_communities_unwghtd.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()
    
    plt.imshow(mbid_vs_comm_wght, interpolation="nearest")
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("MBID", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'mbid_vs_communities_wghtd.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()

    plt.imshow(ragaid_vs_comm, interpolation="nearest")
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("RAGAID", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'ragaid_vs_communities_unwghtd.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear() 
    
    plt.imshow(ragaid_vs_comm_whtd, interpolation="nearest")
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("RAGAID", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'ragaid_vs_communities_wghtd.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()
    
    plt.plot(N_mbid_vs_comm, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# of MBIDs", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_MBIDs_vs_communities.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()
    
    plt.plot(N_ragaid_vs_comm, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# of RAGAs", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_ragas_vs_communities.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()    
    
    
    plt.plot(mbid_centroid_vs_comm, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("Centroid of file dist", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'Centroid_files_vs_communities.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()
    
    plt.plot(ragaid_centroid_vs_comm, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("Centroid of raga dist", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'Centroid_ragas_vs_communities.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()     
    
    plt.plot(comms_vs_mbid, '*')
    plt.xlabel("MBID", fontsize = fsize, fontname=font)
    plt.ylabel("# Communities", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_comms_vs_mbids.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()  
    
    plt.plot(n_comms, N_mbid_comm_distribution, '*')
    plt.xlabel("# Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# MBIDS", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_MBIDS_vs_N_comms.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()        
    

    
    #plt.plot(mbid_centroid_vs_comm)
    #plt.show()
    #plt.plot(ragaid_centroid_vs_comm)
    #plt.show()
    
    
    #np.save(wghtd_graph_filename+'.mbid_dist', mbid_vs_comm)
    #np.save(wghtd_graph_filename+'.raga_dist', ragaid_vs_comm)
    
    #mbid_per_comm = np.sum(mbid_vs_comm, axis =0)
    #raga_per_comm = np.sum(ragaid_vs_comm, axis =0)
    
    
    
    
    
    
    
