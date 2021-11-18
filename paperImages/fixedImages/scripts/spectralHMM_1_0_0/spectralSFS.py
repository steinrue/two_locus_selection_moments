import subprocess
import sys
import numpy
import os
import pandas
from matplotlib import gridspec
import matplotlib.pyplot as plt



# parameters
sampleSize = 15
generationTime = 10000000
yearsPerGen = 1
Nref = 10000
mutationProb = 1e-6

# filenames
plotFile = "sfs.pdf"
sfsFilename = "tmp.sfs"
spectralOutFile = "tmp.spectral"


def generateSFS (sampleSize, timeInYears, ofs):

	# (0, 3, 1); (500, 3, 1)

	for i in range(sampleSize+1):
		for j in range(sampleSize+1):
			ofs.write (f"(0, {sampleSize}, {i}); ({timeInYears}, {sampleSize}, {j})\n")


def prepareSFSFile():

	# write the sfsFile
	print ("[WRITING_SFS]")
	print (sfsFilename)

	ofs = open (sfsFilename, "w")

	generateSFS (sampleSize, generationTime*yearsPerGen, ofs)

	ofs.close()

	print ("[DONE]")


def runSpectralHMM():

	# run the java code to get probabilities

	print ("[RUNNING_SPECTRAL_HMM]")

	cmdLine = " ".join(["python",
		"runSpectralHMM.py",
		"--multiplex",
		"--inputFile", sfsFilename,
		"--mutToBenef", str(mutationProb),
		"--mutFromBenef", str(mutationProb),
		"--effPopSize", str(Nref),
		"--yearsPerGen", str(yearsPerGen),
		"--mutDriftBalance",
		# "--initFrequency", "0.1",
		"--initTime", "-0.00000001",
		"--hetF", "0",
		"--homF", "0",
		"--precision", "40",
		"--matrixCutoff", "150",
		"--maxM", "140",
		"--maxN", "130",
		">", spectralOutFile])

	print (cmdLine)
	# p = subprocess.Popen (cmdLine, stdout=subprocess.PIPE)
	os.system (cmdLine)

	print ("[DONE]")


def plotResults():

	plt.figure(figsize=(12,12))

	# reading in stuff

	print ("[PLOTTING_SFS]")

	daPanda = pandas.read_csv (spectralOutFile, delimiter='\t', comment='#', header=None)

	values = numpy.array(daPanda[3])

	print (f"total sum: {numpy.sum(values)}")

	assert (len(values) == (sampleSize+1)*(sampleSize+1))
	sfs = numpy.reshape (values, (sampleSize+1,sampleSize+1)) 

	# them marginals
	marginalZero = sfs.mean(0)
	marginalOne = sfs.mean(1)

	# print (f"(0,0)\t{sfs[0,0]}")
	# print (f"(0,1)\t{sfs[0,1]}")
	# print (f"(1,0)\t{sfs[1,0]}")

	# maybe logify
	sfs = numpy.log (sfs)
	marginalZero = numpy.log (marginalZero)
	marginalOne = numpy.log (marginalOne)

	# # make a heatmap
	# im = plt.imshow (sfs, cmap='jet', interpolation='nearest', origin='lower')
	# plt.colorbar (im)

	# other thing
	gs = gridspec.GridSpec (2, 2, width_ratios=[1,3], height_ratios=[1,3])
	ax = plt.subplot (gs[1,1])
	axl = plt.subplot (gs[1,0], sharey=ax)
	axb = plt.subplot (gs[0,1], sharex=ax)
	myBar = plt.subplot (gs[0,0],)

	im = ax.imshow (sfs, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')

	plt.colorbar (im, ax=myBar)

	axl.plot (marginalOne, range(sfs.shape[0]))
	axb.plot (range(sfs.shape[1]), marginalZero)

	# plot it
	# plt.show()
	plt.savefig (plotFile)
	print (plotFile)

	print ("[DONE]")


def computeStuff():

	prepareSFSFile()

	runSpectralHMM()

	plotResults()

	# clean up
	os.system (f"rm {spectralOutFile}")
	os.system (f"rm {sfsFilename}")


def main():

	computeStuff()


if __name__ == "__main__":
	main()