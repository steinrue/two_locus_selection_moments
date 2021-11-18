// Copyright (C) 2012  Matthias Steinrücken, Anand Bhaskar, Yun S. Song
//
// This file is part of spectralHMM.
//
// spectralHMM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// spectralHMM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with spectralHMM.  If not, see <http://www.gnu.org/licenses/>.
//
// email: steinrue@stat.berkeley.edu

spectralHMM, version 1.0.0


1. SUMMARY:

spectralHMM is a software program for computing likelihoods of time series allele frequencies data under given population genetic parameters (strength of selection in an arbitrary diploid model, mutation rates, population size, allele age). It assumes a model of a single di-allelic locus, with one wild type allele, and one derived allele that has the given selective advantage. These likelihoods can be used to infer the respective parameters. spectralHMM is an implementation of the method described in

Matthias Steinrücken, Anand Bhaskar, Yun S. Song (2014). A novel spectral method for inferring general selection from time series genetic data. Annals of Applied Statistics, in press. Preprint: http://arxiv.org/abs/1310.1068


2. LICENSES:

The source code is released under the GNU General Public License, version 3. The full text of the license can be found in LICENSE_GPLv3.txt, which should have been included with this README.


3. REQUIREMENTS:

SpectralHMM is implemented in java, so a recent version of the java virtual machine (and possibly the java compiler) is required. The program was developed and tested under JDK/JSE 1.6. Furthermore, the scripts to build and run the program need python (any version greater than 2.5 should work).

Furthermore, the following libraries (jar-files) are required:
- JSAP-2.1.jar
	Download from http://sourceforge.net/projects/jsap/files/jsap/2.1/ (alt: http://www.martiansoftware.com/jsap/).
- arpack_combined_all.jar
	Download from at http://en.sourceforge.jp/projects/sfnet_f2j/releases/ .
- lapack_simple.jar
	Download jlapack-0.8.tgz from http://www.netlib.org/java/f2j/ and unpack. lapack_simple.jar can be found in this archive.

These jar files (or a symbolic link to the files) have to be put into <main-dir>/spectralHMM_lib/ (here <main-dir> denotes the top-level directory). If it is necessary to specify custom paths to these libraries, then the build instructions and the scripts build.py and runSpectralHMM.py have to be changed accordingly.


4. BUILD:

Download the file spectralHMM_1_0_0.tar.gz and unpack it.

4.1 tarball contains jar-file:

File spectralHMM.jar exists. No build necessary. Continue reading section 5. USAGE.

4.2 tarball does not contain jar-file (but should contain sourcecode):

In the <main-dir> (the top level directory) execute the command:

> python build.py

to compile the sourcecode and create the jar-file:

spectralHMM.jar


5. USAGE:

In the <main-dir> (the top level directory), execute the python script with the command

> python runSpectralHMM.py <arguments>

to print the usage and see the available command line arguments, execute the command

> python runSpectralHMM.py --help

For some example calls, see the next section 6. EXAMPLES.

Note: Depending on the command line arguments, the program might require a substantial amount of memory. If the program fails due to insufficient memory, you can provide the '-Xmx' flag to have the JVM use more memory. For example:

> python runSpectralHMM.py -Xmx10g <arguments>

will allow the program to use 10 GB.


6. EXAMPLES:

Note that all population genetic parameters are UNSCALED and the population size is given in terms of diploid individuals. Furthermore, all times area specified in years. To gauge the right precision and cutoff values it is helpful to run the analysis (for the extremal selection coefficients desired) several times with different values, to ensure a stable result.

6.1

The file examples/single_withoutInitTime contains a single temporal dataset and instructions on the data format. The command

> python runSpectralHMM.py --inputFile examples/single_withoutInitTime --mutToBenef 1e-6 --mutFromBenef 1e-6 --effPopSize 10000 --yearsPerGen 5 --initFrequency 0.1 --initTime -20000 --hetF 0.000625 --homF 0.00125 --precision 40 --matrixCutoff 150 --maxM 140 --maxN 130

computes the likelihood of the data given (--inputFile) in the file under the following parameters: The per generation mutation probability from the wild type to the selected allele (--mutToBenef) is 1e-6, and the probability of the reverse event (--mutFromBenef) is also 1e-6. The (diploid) effective population size is 10000, and one generation corresponds to 5 years (--yearsPerGen). The initial frequency (--initFrequency 0.1) of derived advantageous alleles is 0.1 at time (--initTime) -20000 years. The fitness of a heterozygous (--hetF) individual is 0.000625 more than an individual homozygous for the wild type (reference fitness 0), and the fitness of an individual homozygous (--homF) for the derived allele is 0.00125. The computations are performed with a precision (--precision) of 10^(-40), and the size of the matrix whose eigenvalues yield the coefficients for the eigenfunctions (--matrixCutoff) is set to 150. Finally, 141 terms of the infinite sum that approximates the eigenfunctions (--maxM 140) are used, and 131 coefficients of the spectral expansion of the transition density (--maxN 130) are used.

Every line of the output that starts with a '#' denotes logging information. The relevant result is not preceded by a '#' and has 3 values on one line: The heterozygous fitness, the homozygous fitness, and the likelihood.

6.2

The file examples/multi_withoutInitTime contains several temporal dataset, one on each line of the input, and instructions on the data format. The command

> python runSpectralHMM.py --multiplex --inputFile examples/multi_withoutInitTime --mutToBenef 1e-6 --mutFromBenef 1e-6 --effPopSize 10000 --yearsPerGen 5 --mutDriftBalance --initTime -20000 --hetF [0.000625:0.0001:0.0008] --homF [0.00125:0.001:0.003] --precision 40 --matrixCutoff 150 --maxM 140 --maxN 130 --condOnLastSegregating 

computes the likelihood of the datasets given (--inputFile) in the file. In addition the --multiplex has to be specified to indicate multiple datasets in the input file. Most of the population genetic parameters, the precision and the cutoffs are specified as in the previous example. One difference is that instead of an initial frequency, it is specified that at time (--initTime) -20000 the allele frequency is drawn from the stationary distribution of the neutral model (--mutDriftBalance). Instead of a single parameter for the selective advantages (--hetF and --homF), now ranges are specified in the format [start:step:stop]. Finally, this command calculates the likelihood of the data conditional on the event that at least one derived allele is observed at the last sampling time point (--condOnLastSegregating).

Every line of the output that starts with a '#' denotes logging information. The relevant results are not preceded by a '#'. Each result line has 4 values: A running index for the dataset, the heterozygous fitness, the homozygous fitness, and the likelihood.


6.3

The file examples/multi_withInitTime contains several temporal dataset, one on each line of the input, and instructions on the data format. Here an initial time for each dataset is also given in the input file and does not have to be specified on the command line. The command

> python runSpectralHMM.py --multiplex --inputFile examples/multi_withInitTime --mutToBenef 1e-6 --mutFromBenef 1e-6 --effPopSize 10000 --yearsPerGen 5 --initFrequency 0.1 --selection [0.00125:0.001:0.003] --dominance [0.45:0.05:0.55] --precision 40 --matrixCutoff 150 --maxM 140 --maxN 130 --condOnLastSegregating --ignoreBinomials

computes the likelihood of the datasets given (--inputFile) in the file. The parameters are mostly identical to the previous example. However, in this command, the diploid fitness values are given as a selection strength (--selection) and a dominance parameter (--dominance). Also, the likelihood is computed without the binomial factors (--ignoreBinomials) that would be required in the true probability model. Ignoring these factors does not change the shape of the likelihood surface.

Every line of the output that starts with a '#' denotes logging information. The relevant results are not preceded by a '#'. Each result line has 4 values: A running index for the dataset, the selection parameter, the dominance, and the likelihood.


7. CONTACT:

Please contact steinrue@stat.berkeley.edu with bugs, comments, or questions regarding the software.


8. HISTORY:

1.0.0: Initial release.
