import numpy as np
import sys, os
import json
import matplotlib.pyplot as plt
import time 
import networkx as nx


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../library_pythonnew/networkAnalysis'))

DEBUG = False

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

"""

def plot_item_distribution_per_community(fileListFile, out_dir, thresholdBin, pattDistExt, myDatabase = '', myUser = '', force_build_network =0, top_N_com = 10, comm_types = 'raga', network_wght_type = -1):
    """
    This function determines the distribution of items (mbid or raga_id or comp id) across communities. 
    The communities selected for this analysis is determined by comm_types.
    
    comm_types: determine what type of communities are used for the analysis of the distribution ('raga' or 'all' or 'compositions')
    """
    
    #construncting graph
    t1 = time.time()
    base_name = 'Thsldbin_%d_pattDistExt_%s_WghtType_%d'%(thresholdBin, pattDistExt, network_wght_type)
    wghtd_graph_filename = os.path.join(out_dir, base_name+'.edges')
    if force_build_network or not os.path.isfile(wghtd_graph_filename):
        cons_net.constructNetwork_Weighted_NetworkX(fileListFile, wghtd_graph_filename , thresholdBin, pattDistExt, network_wght_type , -1) #we dont apply any disparity filtering here!!!!!!!
        #TODO disparity filtering didn't prove much useful in the ISMIR 2015 analysis. Maybe worth a shot again...try it later point in time
    
    full_net = nx.read_pajek(wghtd_graph_filename)
    t2 = time.time()
    print "time taken = %f"%(t2-t1)
    
    #detecting communities 
    comm_filename = os.path.join(out_dir, base_name+'.community')
    net_pro.detectCommunitiesInNewtworkNX(wghtd_graph_filename, comm_filename)
    
    #Ranking communities based on which we can later decide raga communities
    comm_rank_filename  = os.path.join(out_dir, base_name+'.communityRank')
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
    mbid_vs_comm = np.zeros((len(u_mbids), len(comIds)))
    mbid_vs_comm_wght = np.zeros((len(u_mbids), len(comIds)))    #weighted
    
    ragaid_vs_comm = np.zeros((len(u_ragas), len(comIds)))
    ragaid_vs_comm_whtd = np.zeros((len(u_ragas), len(comIds)))  #weighted
    
    
    cnt_col = 0
    mbid_centroid_vs_comm = []
    ragaid_centroid_vs_comm = []
    nodes_in_comm = []
    for comId in comIds:
        mbids_in_comm = []
        ragaids_in_comm = []
        nodes_in_comm.append(len(comm_data[str(comId)]))
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
        mbid_centroid_vs_comm.append(comm_char.compute_centroid(comm_char.get_histogram_sorted(mbids_in_comm)[0]))
        ragaid_centroid_vs_comm.append(comm_char.compute_centroid(comm_char.get_histogram_sorted(ragaids_in_comm)[0]))
    
    N_umbid_vs_comms = np.sum(mbid_vs_comm,axis=0)  #how many unique mbids are there in each communities
    
    N_umbids_comms_distribution, n_umbids = comm_char.get_histogram_sorted(N_umbid_vs_comms)   # how many unique mbids have appeared in how many communities.
    
    N_uragaid_vs_comms = np.sum(ragaid_vs_comm,axis=0)
    
    N_ucomms_vs_mbids = np.sum(mbid_vs_comm,axis=1) # one mbid has appeared in how many communities    
    N_mbid_comm_distribution, n_comms = comm_char.get_histogram_sorted(N_ucomms_vs_mbids)   # how many mbids have appeared in how many communities.
    
    #computing mbids which either doesn't appear in any of the communities or appear in communities which only have one mbid. Basically trying to find out the mbids which do no combine with any other mbids to for mcommunities.
    ind_mbid_no_comm = np.where(N_ucomms_vs_mbids==0)[0]
    if DEBUG:
        print "Number of files with no communties %d"%(len(ind_mbid_no_comm))
        print ind_mbid_no_comm
        
    ind_mbid_no_comm = ind_mbid_no_comm.tolist()
    
    ind_comm_one_mbid = np.where(N_umbid_vs_comms==1)[0]
    if DEBUG:
        print "There are %d number of communities with only one file in them"%(len(ind_comm_one_mbid))

    for ind in ind_comm_one_mbid:
        ind_mbid = np.where(mbid_vs_comm[:,ind]==1)[0]
        if len(ind_mbid)>1:
            print "There is a serious problem here"
        ind_comms = np.where(mbid_vs_comm[ind_mbid[0],:]==1)[0]
        if len(np.union1d(ind_comms, ind_comm_one_mbid)) > len(ind_comm_one_mbid):
            continue        
        ind_mbid_no_comm.append(ind_mbid[0])
    ind_mbid_no_comm = np.unique(np.array(ind_mbid_no_comm)).tolist()
    
    print "There are %d number of files which do not contribute to any community or are found in communities with only one files"%(len(ind_mbid_no_comm))
    
    output_dir = os.path.join(out_dir, base_name+'_NComms_%d'%(top_N_com), comm_types+'_comms_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #saving list of mbids which are found in none of the communities or in communities with only one mbid
    fname = os.path.join(output_dir, 'problematic_mbids.json')
    mbid_problematic = [u_mbids[r] for r in ind_mbid_no_comm]
    json.dump(mbid_problematic, open(fname,'w'))
    
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
    
    plt.plot(N_umbid_vs_comms, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# of MBIDs", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_MBIDs_vs_communities.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()
    
    plt.plot(N_uragaid_vs_comms, '*')
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
    
    plt.plot(N_ucomms_vs_mbids, '*')
    plt.xlabel("MBID", fontsize = fsize, fontname=font)
    plt.ylabel("# Communities", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_N_ucomms_vs_mbidss.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()  
    
    plt.plot(n_comms, N_mbid_comm_distribution, '*')
    plt.xlabel("# Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# MBIDS", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_MBIDS_vs_N_comms.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()    
    
    plt.plot(n_umbids, N_umbids_comms_distribution, '*')
    plt.xlabel("# unique MBIDS", fontsize = fsize, fontname=font)
    plt.ylabel("# Communities", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_Comms_vs_N_UMBIDs.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()   
    
    plt.plot(nodes_in_comm, '*')
    plt.xlabel("Communities", fontsize = fsize, fontname=font)
    plt.ylabel("# nodes", fontsize = fsize, fontname=font, labelpad=fsize2)
    plotName = os.path.join(output_dir, 'N_nodes_vs_Comms.pdf')
    fig.savefig(plotName, bbox_inches='tight')
    fig.clear()      
    
    
    
if __name__ == "__main__":
        
        distance_bins = [8, 10, 12, 14, 16]
        pattDistExts = ['.pattDistances1']
        top_N_coms = [10, 20, 30, 40, 50]
        comm_types = ['raga']
        network_wght_types = [-1] 
        
        fileListFile = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic10RagasISMIR2015DB/__dbInfo__/Carnatic10RagasISMIR2015DB.flist'
        out_dir = 'community_analysis_data/ISMIR2015_10RAGA_TONICNORM'
        myDatabase = 'ISMIR2015_10RAGA_TONICNORM'
        myUser = 'sankalp'
        
        for dbin in distance_bins:
            for ext in pattDistExts:
                for topN in top_N_coms:
                    for comm_type in comm_types:
                        for wght_type in network_wght_types:
                            print dbin, ext, topN, comm_type, wght_type
                            plot_item_distribution_per_community(fileListFile, out_dir, dbin, ext, myDatabase = myDatabase, myUser = myUser, force_build_network =0, top_N_com = topN, comm_types = comm_type, network_wght_type = wght_type)

        
        
