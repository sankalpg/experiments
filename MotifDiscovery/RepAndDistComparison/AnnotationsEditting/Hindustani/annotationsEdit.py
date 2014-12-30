import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../library_pythonnew/batchProcessing/'))
import batchProcessing as BP

def find_nearest_element_ind(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx


def findAnotsWithSil(root_dir, outputFile,  anotExt = '.anot', pitchExt = '.pitch', silVal = 0):

	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	fid = open(outputFile, 'w')

	for filename in filenames:
		fname, ext = os.path.splitext(filename)
		timePitch = np.loadtxt(fname + pitchExt)
		anots = np.loadtxt(filename)

		for ii, anot in enumerate(anots):
			indStart = find_nearest_element_ind(timePitch[:,0], anot[0])
			indEnd = find_nearest_element_ind(timePitch[:,0], anot[1])
			if len(np.where(timePitch[indStart: indEnd,1]<=silVal)[0]) > 0 :
				fid.write("%s\t%d\t%d\t%d\n"%(filename, ii, anot[0], anot[1]))

	fid.close()

def findSmallAnots(root_dir, outputFile, anotExt = '.anot', Duration = 0.5):
	
	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	fid = open(outputFile, 'w')

	for filename in filenames:
		anots = np.loadtxt(filename)
		for ii, anot in enumerate(anots):
			if anot[1]-anot[0] <= Duration:
				fid.write("%s\t%d\t%f\t%f\n"%(filename, ii, anot[0], anot[1]))

	fid.close()		

def convertSonicAnotsToMine(root_dir, anotExt = '.anot', outExt = '.anot'):

	filenames = BP.GetFileNamesInDir(root_dir, anotExt)

	for filename in filenames:
		fname, ext = os.path.splitext(filename)
		anots = np.loadtxt(filename)
		anots[:,1] = anots[:,0] + anots[:,1]
		np.savetxt(fname + outExt, anots)

def renameAnotFiles(root_dir, inpExt = '.anot', outExt = '.anotEdit1'):

	filenames = BP.GetFileNamesInDir(root_dir, inpExt)

	for filename in filenames:
		fname, ext = os.path.splitext(filename)
		os.rename(filename, fname+outExt)

def formatAnots(root_dir, inpExt = '.anot'):

	filenames = BP.GetFileNamesInDir(root_dir, inpExt)

	for filename in filenames:
		anots = np.loadtxt(filename)
		np.savetxt(filename, anots, fmt='%.3f\t%.3f\t%d')

