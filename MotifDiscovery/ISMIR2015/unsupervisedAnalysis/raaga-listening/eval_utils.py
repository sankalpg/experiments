import numpy as np
import json
import os
import uuid

raagas = [
'09c179f3-8b19-4792-a852-e9fa0090e409',
'77bc3583-d798-4077-85a3-bd08bc177881',
'85ccf631-4cdf-4f6c-a841-0edfcf4255d1',
'123b09bd-9901-4e64-a65a-10b02c9e0597',
'700e1d92-7094-4c21-8a3b-d7b60c550edf',
'a9413dff-91d1-4e29-ad92-c04019dce5b8',
'aa5f376f-06cd-4a69-9cc9-b1077104e0b0',
'bf4662d4-25c3-4cad-9249-4de2dc513c06',
'd5285bf4-c3c5-454e-a659-fec30075990b',
'f0866e71-33b2-47db-ab75-0808c41f2401'
]

def fetchRagaNames(raagas, outFile):

	dunya.set_token('60312f59428916bb854adaa208f55eb35c3f2f07')
	names = {}
	for raaga in raagas:
		r_info = ca.get_raaga(raaga)
		names[raaga]= r_info['common_name'].title()

	json.dump(names, open(outFile,'w'))
	return names

def generate_phraseID_2_location_map(root_dir, output_file):
    """
    Map a location (pre-desired) to a phrase id. Because we want to present the first phrase in the
    ordered list, this function does that for the entire collection.
    """

    n_raagas = 10
    n_communities = 10
    n_phrase_per_com = 1

    id_loc_map = {}
    phrase_id = 1
    for raaga in raagas:
        for cInd in range(1,n_communities+1):
            for pInd in range(n_phrase_per_com):
                comm_name = 'phrase_%d_'%cInd
                phrase_name = '%d_'%pInd
                for (dirs, path, files) in os.walk(root_dir):
                    if dirs.count(raaga) == 1 and dirs.count(comm_name) == 1:
                        for f in files:
                            if f.startswith(phrase_name):
                                id_loc_map[phrase_id] = {'path':os.path.join(dirs.replace(root_dir,''), f), 'raagaId':raaga}
                                phrase_id+=1

    json.dump(id_loc_map, open(output_file,'w'))


musicians = [
'Vignesh',
'Vidya',
'Vikram',
'Ananya',
'Dharini',
'Bharat',
'Archana',
'Shraddha',
'Gokul',
'Jayant',
'Lavanya',
'Srividya',
'Vidhya',
'Aarathi',
'Sridharan',
'Kaushik',
'Ajay S',
'Sankalp',
'Xavier',
'Joan',
'Kaustuv K.',
'User1',
'User2',
'User3',
'User4',
'User5'
    ]

def generate_UUIDS_randomization_for_musicians(names, outfile):

    mData = {}
    for name in names:
        uuid_local = uuid.uuid1().hex
        mData[uuid_local] = {}
        mData[uuid_local]['name'] = name
        ids = range(1,101)
        np.random.shuffle(ids)
        mData[uuid_local]['pids'] = ids

    json.dump(mData,open(outfile, 'w'))




