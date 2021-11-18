#!/usr/bin/env python

import os
import sys

scriptDir = os.path.dirname(os.path.realpath(__file__))

# set the classpath right
libDir = os.path.join (scriptDir, "spectralHMM_lib")
jarFile = os.path.join (scriptDir, "spectralHMM.jar")
CLASSPATH = '"%s:%s"' % (os.path.join (libDir, "*"), jarFile)

# check whether the libraries are in place
requiredLibs = ["arpack_combined_all.jar", "JSAP-2.1.jar", "lapack_simple.jar"]
for lib in requiredLibs:
	if (not os.path.isfile (os.path.join (libDir, lib))):
		print ("[LIBRARY] %s is missing" % lib)
		exit(-1)

# see about whether we have some jvm arguments
JVMargs = []
args = []
preArgs = sys.argv[1:]
for thisArg in preArgs:
	if (thisArg.startswith ("-X")):
		JVMargs.append (thisArg)
	else:
		args.append (thisArg)
		

# run it
argString = " ".join (args)
JVMargString = " ".join (JVMargs)
spectralHMMcmd = "java %s -classpath %s edu.berkeley.spectralHMM.oneD.SelectionHMM %s" % (JVMargString, CLASSPATH, argString)
os.system (spectralHMMcmd)