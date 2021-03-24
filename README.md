# two_locus_selection_moments

This repository contains the code associated with *"A numerical framework for genetic hitchhiking in populations of variable size"* by Eric Friedlander and Matthias Steinrücken. I brief guide to this repository is as follows:

* ```momlink``` - This folder contains the code implementing the method outlined in the manuscript. ```ode.py``` contains code which defined the ```TwoLocMomOde``` object which computes the time derivative of the moment ODEs and stores the state of the moments. This object is the main way to interfact with the moments. ```integrator.py``` implements a Runger-Kutta 4(5) ODE solver which is modified to ensure the solution is a probability measure. ```help_funcs.py``` contains helpful functions and objects that do things like interpolation and downsampling of moments. ```demography.py``` contains code necessary for defining and working with populations of variable size.
