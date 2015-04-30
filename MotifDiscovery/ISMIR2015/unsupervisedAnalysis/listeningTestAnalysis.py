import numpy as np
import json
import os, sys
import matplotlib.pyplot as plt

users = [
'f38adb44e36f11e4a6ba902b349f69fb',
'f38adb45e36f11e4a6ba902b349f69fb',
'f38adb46e36f11e4a6ba902b349f69fb',
'f38adb47e36f11e4a6ba902b349f69fb',
'f38adb48e36f11e4a6ba902b349f69fb',
'f38adb49e36f11e4a6ba902b349f69fb',
'f38adb4ae36f11e4a6ba902b349f69fb',
'f38adb4ce36f11e4a6ba902b349f69fb',
'f38adb4ee36f11e4a6ba902b349f69fb',
'f38adb4fe36f11e4a6ba902b349f69fb',
'f38adb50e36f11e4a6ba902b349f69fb',
'f38adb51e36f11e4a6ba902b349f69fb',
'f38adb52e36f11e4a6ba902b349f69fb',
'f38adb53e36f11e4a6ba902b349f69fb',
'f38adb55e36f11e4a6ba902b349f69fb',
'f38adb56e36f11e4a6ba902b349f69fb',
'f38adb57e36f11e4a6ba902b349f69fb',
'f38adb59e36f11e4a6ba902b349f69fb',
'f38adb5ae36f11e4a6ba902b349f69fb',
'f38adb5ce36f11e4a6ba902b349f69fb',
'f38adb5de36f11e4a6ba902b349f69fb'
]

def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                names.append(path + "/" + f)
    return names

def computePerRagaMOS(root_dir, pattern_id_map, users_ids):

	#selecting only the users who have all 100 files saved
	user_ratings = {}
	for user in users_ids:
		files = GetFileNamesInDir(os.path.join(root_dir,user), filter = '.txt')
		if len(files) == 100:
			print user
			user_ratings[user] = {}
			for ii in range(1,101):
				rating = np.loadtxt(os.path.join(root_dir,user,str(ii)+'.txt'))
				user_ratings[user][ii] = rating

	pattern_id_map = json.load(open(pattern_id_map))

	users = user_ratings.keys()
	perPhraseRatings = {}

	for ii in range(1,101):
		perPhraseRatings[ii] = []
		for user in users:
			perPhraseRatings[ii].append(user_ratings[user][ii])

	#per phrase MOS
	per_phrase_mos = []
	for ii in range(1,101):
		per_phrase_mos.append(np.mean(perPhraseRatings[ii]))

	plt.plot(per_phrase_mos)
	plt.show()
	"""
	for ii in range(100):
		print ii+1, per_phrase_mos[ii]

	for ii in range(10):
		print ii+1, np.mean(per_phrase_mos[ii:(ii*10)+5])
	print np.mean(per_phrase_mos)
	"""