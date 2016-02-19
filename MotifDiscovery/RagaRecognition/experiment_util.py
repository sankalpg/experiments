import os, sys
import numpy as np
import pickle
import json
import codecs
import itertools as iter
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../library_pythonnew/networkAnalysis/'))

import ragaRecognition as rr



local_database_path_hindustani = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/hindustaniDB/Hindustani30Ragas/'
local_database_path_carnatic = '/media/Data/Datasets/PatternProcessing_DB/unsupervisedDBs/carnaticDB/Carnatic40RagaICASSP2016'

def quantizeOverlap(sets):
	overlap = np.zeros((len(sets), len(sets)))

	for ii in range(len(sets)):
		for jj in range(len(sets)):
			overlap[ii,jj] = len(np.intersect1d(np.array(sets[ii]), np.array(sets[jj])))

	return overlap

def generateSubsetDatasets(root_dir, file_list_file, database = '', user = '', tradition = 'hindustani'):
	"""
	This function generates n_datasets number of sub datasets with sub_classes number of classes each out of proivided dataset
	This is a completely hardcoded function as we need to run it only once (in a lifetime hopefully)
	:root_dir: Directory to store output
	:sub_classes: number of classes in each sub-dataset
	:n_datasets: number of datasets to be generated
	:file_list_file: this is the root file, a comprehensive list of all the files in the database
	:database: psql database from which raga labels are to be fetched
	:user: psql user of the database
	"""
	#raga and mbid mapping
	raga_mbid = rr.get_mbids_raagaIds_for_collection(file_list_file, database, user)
	raga2mbid = {}
	mbid2raga = {}
	for r in raga_mbid:
		if not raga2mbid.has_key(r[0]):
			raga2mbid[r[0]] = []
		raga2mbid[r[0]].append(r[1])
		mbid2raga[r[1]] = r[0]
	ragas = raga2mbid.keys()
	#reading all files in the file_list
	lines = codecs.open(file_list_file, 'r', encoding = 'utf-8').readlines()
	mbid2file = {}
	for line in lines:
		mbid = rr.get_mbid_from_mp3(line.strip()+'.mp3')
		mbid2file[mbid] = {'filepath':line.strip()}
		line = line.strip().replace(local_database_path_carnatic, '')
		line = line.strip().replace(local_database_path_hindustani, '')
		mbid2file[mbid].update({'rel_filepath':line.strip()})

	
	#since we want maximally different set of 	n_datasets. We do a card coded way, which ensures that.
	if tradition == 'hindustani':
		n_datasets = 50
		set1_size = 2
		set2_size = 3
		n_set1 = 6
		n_set2 = 6
		sub_classes  =[5 , 10, 15, 20, 25, 30]
	if tradition == 'carnatic':
		n_datasets = 50
		set1_size = 2
		set2_size = 3
		n_set1 = 8
		n_set2 = 8
		sub_classes  =[5 , 10, 20, 30, 40]

	"We apply awesome logic to make sure that we have maxaimally different set of combinations! HINT: if there are all combinations in list order such that diff in adjacent is one (the typical way to generate) then the ones with maximal diff are on the binary divisions"
	fractions = [0, 1, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
	set1 = []
	set2 = []
	cnt = 0
	for ii in range(0, set1_size*n_set1, set1_size):
		set1.append((ragas[ii], ragas[ii+1]))
	for ii in range(set1_size*n_set1, len(ragas), set2_size):
		set2.append((ragas[ii], ragas[ii+1], ragas[ii+2]))

	set1_inds = range(len(set1))
	set2_inds = range(len(set2))
	for n_class in sub_classes:
		sets = []
		n_combinations = n_class/(set1_size + set2_size)
		comb1_array = list(iter.combinations(set1_inds, n_combinations))
		n_points_first = int(np.floor(np.sqrt(n_datasets)))
		for f in fractions[:n_points_first+1]:
			comb1  = comb1_array[int(np.ceil(f*(len(comb1_array)-1)))]
			comb1 = list(comb1)
			first_group = []
			for c1 in comb1:
				first_group.extend(set1[c1])
			
			comb2_array = list(iter.combinations(set2_inds, n_combinations))
			n_points_second = int(np.floor(np.sqrt(n_datasets)))
			for f in fractions[:n_points_second+1]:
				comb2  = comb2_array[int(np.ceil(f*(len(comb2_array)-1)))]
				comb2 = list(comb2)
				second_group = []
				for c2 in comb2:
					second_group.extend(set2[c2])
				sets.append(np.append(first_group, second_group))

		# overlap = quantizeOverlap(sets)
		# return overlap
		#make the directory
		dir_name = os.path.join(root_dir, '%d_classes'%n_class)
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)

		#dumping files
		for ii in range(min(len(sets), n_datasets)):
			uuid_file = str(uuid.uuid4())
			filename_list = os.path.join(dir_name, str(n_class) + '_' + uuid_file + '.filelist')
			filename_json = os.path.join(dir_name, str(n_class) + '_' + uuid_file + '.json')
			fid = codecs.open(filename_list, 'w', encoding = 'utf-8')
			json_list = []
			print "n_class: %d, Dataset index: %d, num_ragas: %d"%(n_class, ii, len(sets[ii]))
			for ragaid in sets[ii]:
				for mbid in raga2mbid[ragaid]:
					fid.write("%s\n"%mbid2file[mbid]['filepath'])
					json_list.append((mbid, ragaid, mbid2file[mbid]['rel_filepath']))

			fid.close()
			json.dump(json_list, codecs.open(filename_json, 'w', encoding = 'utf-8'))




def generate_FILE_MBID_RAGAlist(filelist, database, user, outputfile):

	raga_mbid = rr.get_mbids_raagaIds_for_collection(filelist, database, user)

	fid = open(outputfile, 'w')

	lines = open(filelist, 'r').readlines()
	for ii, line in enumerate(lines):
		file_path = line.strip()
		fid.write("%s\t%s\t%s\n"% (file_path, raga_mbid[ii][1], raga_mbid[ii][0]))

	fid.close()






