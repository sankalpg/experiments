import numpy as np

sampleRate = {	'10.0':"S_100",
				'15.0':"S_67",
				'20.0':"S_50",
				'25.0':"S_40",
				'30.0':"S_33"}

scaling = {	'1.0':'U_off',
			'5.0':'U_on'}

normalization = {	'0.0':'off',
					'1.0':'tonic',
					'2.0':'z',
					'3.0':'mean',
					'4.0':'median',
					'5.0':'MAD'}
quantization = {	'-1.0':'',
					'12.0':'Q12',
					'24.0':'Q24'}
distances = {	'0.0':'_euc',
				'1.0':'_DTW'}	
DTWType = {	'-1.0':'',
			'0.0':'_L0',
			'1.0':'_L1'}								
constraint = {	'0.025':'',
				'0.05':'_G5',
				'0.1':'_G10',
				'0.9':'_G90'}

def convertResutlsToHumanReadable(csvFile,accuracyFile, outFile):
	data = np.loadtxt(csvFile)
	acc = np.transpose(np.array([np.loadtxt(accuracyFile)]))
	print data.shape
	data = np.hstack((data,acc))
	print data.shape
	sortInd = np.argsort(data[:,-1])
	sortInd = sortInd[::-1]
	data = data[sortInd,:]
	fid = open(outFile, 'w')
	fid.write("MAP\tSrate\tNorm\tTScale\tDist\n")
	for ii in range(data.shape[0]):
		scalVal = data[ii,0]
		distVal = data[ii,5]
		quantVal = data[ii,10]
		dtwtVal = data[ii,12]
		bandVal = data[ii,15]
		normVal = data[ii,17]
		samVal = data[ii,20]

		Srate = sampleRate[str(samVal)]
		Norm = 'N_'+normalization[str(normVal)]+quantization[str(quantVal)]
		TScale = scaling[str(scalVal)]
		Dist = 'D'+distances[str(distVal)] + DTWType[str(dtwtVal)] + constraint[str(bandVal)]
		fid.write(str(round(data[ii,-1],4)) + '\t' + Srate + '\t' + Norm+ '\t' + TScale+ '\t' +Dist + '\n')

	fid.close()