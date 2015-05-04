import numpy as np
import json
import os, sys
import matplotlib.pyplot as plt
import shutil

users = [
'f38adb44e36f11e4a6ba902b349f69fb',
'f38adb45e36f11e4a6ba902b349f69fb',
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

users_all = [
'f38adb44e36f11e4a6ba902b349f69fb',
'f38adb45e36f11e4a6ba902b349f69fb',
'f38adb47e36f11e4a6ba902b349f69fb',
'f38adb46e36f11e4a6ba902b349f69fb',
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

#NOTE THAT WE HAVE REMOVED VIKRAM () from the evaluations owing to him being a outlier. 
#The ratings he gave are way way lesser than other musicians. Probably he had hard time understanding what is to be evaluated.


rag_map_full = {u'09c179f3-8b19-4792-a852-e9fa0090e409': r'$K\bar{a}pi$',
 u'123b09bd-9901-4e64-a65a-10b02c9e0597': r'$Bhairavi$',
 u'700e1d92-7094-4c21-8a3b-d7b60c550edf': r'$Beh\bar{a}g$',
 u'77bc3583-d798-4077-85a3-bd08bc177881': r'$Hamsadhvani$',
 u'85ccf631-4cdf-4f6c-a841-0edfcf4255d1': r'$K\bar{a}mavardhini$',
 u'a9413dff-91d1-4e29-ad92-c04019dce5b8': r'$T\bar{o}{d}\d{i}$',
 u'aa5f376f-06cd-4a69-9cc9-b1077104e0b0': r'$B\bar{e}ga{d}\d{a}$',
 u'bf4662d4-25c3-4cad-9249-4de2dc513c06': r'$Kaly\bar{a}{n}\d{i}$',
 u'd5285bf4-c3c5-454e-a659-fec30075990b': r'$Darb\bar{a}r$',
 u'f0866e71-33b2-47db-ab75-0808c41f2401': r'$K\bar{a}mb\bar{o}ji$'}

rag_map = {
 u'09c179f3-8b19-4792-a852-e9fa0090e409': 'Kp',
 u'123b09bd-9901-4e64-a65a-10b02c9e0597': 'Bh',
 u'700e1d92-7094-4c21-8a3b-d7b60c550edf': 'Bg',
 u'77bc3583-d798-4077-85a3-bd08bc177881': 'Hd',
 u'85ccf631-4cdf-4f6c-a841-0edfcf4255d1': 'Kv',
 u'a9413dff-91d1-4e29-ad92-c04019dce5b8': 'Td',
 u'aa5f376f-06cd-4a69-9cc9-b1077104e0b0': 'Bd',
 u'bf4662d4-25c3-4cad-9249-4de2dc513c06': 'Kl',
 u'd5285bf4-c3c5-454e-a659-fec30075990b': 'Db',
 u'f0866e71-33b2-47db-ab75-0808c41f2401': 'Kb'}  
       

def renameAndCopyPhrases(audio_dir, output_dir,pattern_id_map):
    
    patten_map = json.load(open(pattern_id_map))
    
    for k in patten_map.keys():
        shutil.copy(os.path.join(audio_dir,patten_map[k]['path']),os.path.join(output_dir,k+'.mp3'))
    

def FleisskappaAgreement(mtx):
    """
    This function computes Fleiss kappa Agreement (http://en.wikipedia.org/wiki/Fleiss%27_kappa), 
    which is kind of kappa agreement for the case of multiple raters. 
    The input to this function has a specific format.
    The rows of the matrix is different subjects (different items which were rated)
    The columns are the raters
    And the values at every row, column is the rating which is assigned.
    So we follow the formula provided in the wikipedia page and compute the agreement
    
    """
    
    #creating a matrix of subject(rows) Vs categories or rating(cols) where value per pixel is number of raters gave that category or rating
    
    # finding unique ratings
    categories = np.unique(mtx.flatten())
    N = mtx.shape[0]
    n = mtx.shape[1]
    k = len(categories)
    
    mtx_cat = np.zeros((N,len(categories)))
    
    for ii in range(N):
        for jj, c in enumerate(categories):
            mtx_cat[ii,jj] = len(np.where(mtx[ii,:]==c)[0])
    
    print mtx_cat
    
    PI = []
    PJ = []
    
    for ii, c in enumerate(categories):
        PJ.append(np.sum(mtx_cat[:,ii])/float(N*n))
    
    for ii in range(N):
        PI.append((np.sum(np.power(mtx_cat[ii,:],2))-n)/float(n*(n-1)))
    
    p_bar = np.mean(PI)
    p_bar_e = np.sum(np.power(PJ,2))
    
    agreement = (p_bar - p_bar_e)/(1-p_bar_e)
    
    return agreement
    #print PI, PJ, p_bar, p_bar_e, agreement
    
        
    
    
    
def GetFileNamesInDir(dir_name, filter=".wav"):
    names = []
    for (path, dirs, files) in os.walk(dir_name):
        for f in files:
            if filter.split('.')[-1].lower() == f.split('.')[-1].lower():
                names.append(path + "/" + f)
    return names


def plotAvgUserResponse(root_dir, user_map, user_ids, plotName = -1):
    
    #selecting only the users who have all 100 files saved
    user_ratings = {}
    for user in user_ids:
        files = GetFileNamesInDir(os.path.join(root_dir,user), filter = '.txt')
        if len(files) == 100:
            print user
            user_ratings[user] = {}
            for ii in range(1,101):
                rating = np.loadtxt(os.path.join(root_dir,user,str(ii)+'.txt'))
                user_ratings[user][ii] = rating
    
    
    user_names = json.load(open(user_map))
    
    ratings = []
    names = []
    for u in user_ratings:
        r = []
        for ii in range(1,101):
            r.append(user_ratings[u][ii])
        ratings.append(np.mean(r))
        names.append(user_names[u]['name'])
    
    fig, ax = plt.subplots()
    fsize = 8
    fsize2 = 16
    font="Times New Roman"
    plt.plot(ratings)
    plt.xlabel('Musicians', fontsize = fsize, fontname=font)
    plt.ylabel('Feedback', fontsize = fsize, fontname=font)
    
    plt.xticks(range(len(ratings)), names, fontsize = fsize, fontname=font)
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1

def computePerRagaOverallStats(root_dir, pattern_id_map, users_ids):

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


    users = user_ratings.keys()
    perPhrasePerUserRatings = []
    for ii in range(1,101):
        per_phrase = []
        for user in users:
            per_phrase.append(user_ratings[user][ii])
        perPhrasePerUserRatings.append(per_phrase)
    
    opacity = 1
    error_config = {'ecolor': '0.3'}
    
    perPhrasePerUserRatings = np.array(perPhrasePerUserRatings)
    
    pattern_ids = json.load(open(pattern_id_map))
    names = []
    fig, ax = plt.subplots()
    per_raga_agreement = []
    per_raga_MOS = []
    per_raga_MOS_std = []
    for ii in range(10):
        names.append(rag_map_full[pattern_ids[str(ii*10+1)]['raagaId']])        
        per_raga_MOS.append(np.mean(perPhrasePerUserRatings[ii*10:(ii*10)+10,:]))        
        per_raga_MOS_std.append(np.std(np.mean(perPhrasePerUserRatings[ii*10:(ii*10)+10,:],axis=1)))        
        per_raga_agreement.append(FleisskappaAgreement(perPhrasePerUserRatings[ii*10:(ii*10)+10,:]))        
        print "##################################"
        print names[-1]
        print "Agreement = %f"%per_raga_agreement[-1]
        print "MOS = %f"%per_raga_MOS[-1]
        print "STD = %f"%per_raga_MOS_std[-1]
        print "##################################"
    
    MOS = np.mean(perPhrasePerUserRatings)  
    STD = np.std(np.mean(perPhrasePerUserRatings,axis=1))
    agreement = FleisskappaAgreement(perPhrasePerUserRatings)
    
    print "##################################"
    print "Overall"
    print "Agreement = %f"%agreement
    print "MOS = %f"%MOS
    print "STD = %f"%STD
    print "##################################"    
    
    return 1

def plotAverageRatingHistogram(root_dir, pattern_id_map, users_ids, plotName=-1):

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


    users = user_ratings.keys()
    perPhrasePerUserRatings = {}
    perPhraseMOS = []
    for ii in range(1,101):
        perPhrasePerUserRatings[ii] = []
        for user in users:
            perPhrasePerUserRatings[ii].append(user_ratings[user][ii])
        
        perPhraseMOS.append(np.mean(perPhrasePerUserRatings[ii]))
    
    fig, ax = plt.subplots()
    
    ratings = [0, .1, .2, .3, .4, .5, .6, .7,.8,.9,1]
    hist = []
    perPhraseMOS = np.array(perPhraseMOS)
    for r in ratings:
        hist.append(len(np.where(perPhraseMOS==r)[0]))
    
    opacity = 1
    error_config = {'ecolor': '0.3'}
    
    index = np.arange(len(hist))
    plt.bar(index , hist, 0.8, alpha=opacity, color='#33CCFF', error_kw=error_config)

    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    plt.xlabel('$\mu_p$', fontsize = fsize, fontname=font)
    plt.ylabel('Count', fontsize = fsize, fontname=font)
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.xticks(index +0.4, ratings, fontsize = fsize, fontname=font)
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    

    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1


def computePerRagaMOS(root_dir, pattern_id_map, users_ids, plotName=-1):

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


    users = user_ratings.keys()
    perPhrasePerUserRatings = {}
    perPhraseMOS = []
    for ii in range(1,101):
        perPhrasePerUserRatings[ii] = []
        for user in users:
            perPhrasePerUserRatings[ii].append(user_ratings[user][ii])
        
        perPhraseMOS.append(np.mean(perPhrasePerUserRatings[ii]))
    
    opacity = 1
    error_config = {'ecolor': '0.3'}
    
    pattern_ids = json.load(open(pattern_id_map))
    names = []
    fig, ax = plt.subplots()
    per_raga_rating = []
    per_raga_std = []
    for ii in range(10):
        names.append(rag_map[pattern_ids[str(ii*10+1)]['raagaId']])
        MOS = perPhraseMOS[ii*10:(ii*10)+10]
        per_raga_rating.append(np.mean(MOS))
        per_raga_std.append(np.std(MOS))
        #index = ii + 0.08*np.arange(10)
    
    index = np.arange(10)
    plt.bar(index, per_raga_rating, 0.8,
                alpha=opacity,
                color='#33CCFF',
                error_kw=error_config,
                yerr = per_raga_std)
    
    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    plt.xlabel(r'$R\bar{a}gs$', fontsize = fsize, fontname=font)
    plt.ylabel('$\mu_p$', fontsize = fsize, fontname=font)
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
    
    plt.xticks(np.arange(10)+0.4, names, fontsize = fsize, fontname=font)
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1

def computePerRagaMOS_perPhrase(root_dir, pattern_id_map, users_ids, plotName=-1):

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


    users = user_ratings.keys()
    perPhrasePerUserRatings = {}
    perPhraseMOS = []
    for ii in range(1,101):
        perPhrasePerUserRatings[ii] = []
        for user in users:
            perPhrasePerUserRatings[ii].append(user_ratings[user][ii])
        
        perPhraseMOS.append(np.mean(perPhrasePerUserRatings[ii]))
    
    opacity = 1
    error_config = {'ecolor': '0.3'}
    
    pattern_ids = json.load(open(pattern_id_map))
    names = []
    fig, ax = plt.subplots()
    for ii in range(10):
        names.append(rag_map[pattern_ids[str(ii*10+1)]['raagaId']])
        MOS = perPhraseMOS[ii*10:(ii*10)+10]
        index = ii + 0.06*np.arange(10)
        plt.bar(index, MOS, 0.06,
                alpha=opacity,
                color='#808080',
                edgecolor = "#191919",
                error_kw=error_config)
    
    fsize = 20
    fsize2 = 16
    font="Times New Roman"
    plt.xlabel(r'$R\bar{a}gs$', fontsize = fsize, fontname=font)
    plt.ylabel('$\mu_p$', fontsize = fsize, fontname=font)
    
    plt.ylim([0.18,1.0])
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()    
    ax.set_aspect((xLim[1]-xLim[0])/(2*float(yLim[1]-yLim[0])))
    plt.tick_params(axis='both', which='major', labelsize=fsize2)
        
    
    plt.xticks(np.arange(10)+0.3, names, fontsize = fsize, fontname=font)
    if isinstance(plotName, int):
        plt.show()
    elif isinstance(plotName, str):
        fig.savefig(plotName, bbox_inches='tight')

    return 1



    
    """    
    #per phrase MOS
    per_phrase_mos = []
    for ii in range(1,101):
    per_phrase_mos.append(np.mean(perPhraseRatings[ii]))

    plt.plot(per_phrase_mos)
    plt.show()

    for ii in range(100):
    print ii+1, per_phrase_mos[ii]

    for ii in range(10):
    print ii+1, np.mean(per_phrase_mos[ii:(ii*10)+5])
    print np.mean(per_phrase_mos)
	"""
