# two_locus_selection_moments

This repository contains the code associated with *"A numerical framework for genetic hitchhiking in populations of variable size"* by Eric Friedlander and Matthias Steinrücken. I brief guide to this repository is as follows:

* ```momlink``` - This folder contains the code implementing the method outlined in the manuscript.
    * ```ode.py``` contains code which defined the ```TwoLocMomOde``` object which computes the time derivative of the moment ODEs and stores the state of the moments. This object is the main way to interfact with the moments. 
    * ```integrator.py``` implements a Runger-Kutta 4(5) ODE solver which is modified to ensure the solution is a probability measure. 
    * ```help_funcs.py``` contains helpful functions and objects that do things like interpolation and downsampling of moments. 
    * ```demography.py``` contains code necessary for defining and working with populations of variable size. 
* ```paperImages``` - This folder contains the code necessary to generate the data and figures in the manuscript. 
    * ```auxImages``` contains the scripts ```betaPlots.py``` and ```demographicModel.py``` which generate Figures 1 and 2, respectively. 
    * ```fixedImages``` contains the code necessary to generate Figures 3-6, D.1, and the data used in  Tables 1 and 2. In order to generate these just run the ```snakemake.snakefile``` script. For completeness, the script will generate several figures note included in the manuscript and will also create many intermediate file in the ```data``` and ```ode_output``` folders. The code to recreate Tables 1 and 2 is included in a Jupyter notebook stored in the ```analysis folder``` of the main repo.  This will generate all Figures and all of the necessary data. ```run_snakemake.sh``` contains code for running the code in parallel on a cluster which is highly recommended. This will need to be adjusted (along with the ```cluster/cluster.json``` depending on the cluster being used). In order to run this code one will need to download and install [Snakemake](https://snakemake.readthedocs.io/en/stable/) and [simuPOP](http://simupop.sourceforge.net/). 
    * ```stationaryImages``` contains the code to generate Figure 7, 8, D.2-D5. The images can be generated by running the snakemake file in the same way as in the ```fixed Images``` folder. 
    * ```slimImages``` contains the code to generate Figure 9. The images can be generated by running the snakemake file in the same way as in the ```fixed Images``` folder. This time one will need to have [SLiM](https://messerlab.org/slim/) installed in order to run the code. 
* ```analysis``` - This folder contains the Jupyter notebook ```performanceAnalysis.ipybn``` necessary to generate the Tables 1 and 2. In order to run the notebook make sure that ```odePerformance_all.csv``` is in the folder. This file will be in ```paperImages/fixedImages/data``` after running the corresponding snakemake script. 
* ```JackKnife.nb``` - This is a Mathematica notebook which defines moments in terms of a second order polynomial. This is necessary for implementing the JackKnife method proposed in Jouganous et al. (2017) and Ragsdale and Gravel (2019). 
