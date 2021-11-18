import sys
import numpy as np
import os
import allel
import glob
import pandas
import subprocess
import pickle as pkl
import json
import click


@click.command()
@click.option('-demog', help='Demography', type=str)
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-init', help='initial frequency', type=str)
@click.option('-out', '--out_file', help='Filename to output data dictionary to', type=str)
def main(demog, m, s, init, out_file):

    

    # Extract initial haplotype frequencies
    init = json.loads(init)

    
    multiplexFile = "expected_traj.multi"
    outFile = "expected_traj.spec"

    # and then run SpectralHMM
    if demog == 'constant2':
        Ne = 2000
        gens = 3000
        momentGens = np.arange (0, gens, 10)
    elif demog == 'constant10':
        Ne = 10000
        gens = 6000
        momentGens = np.arange (0, gens, 10)
    initFreq = init[0] + init[1]
    initTime = -2
    dps = 75
    matrixCutoff = 200
    m = str(m)
    fixedSel = s
    cmd = f"python scripts/spectralHMM_1_0_0/runSpectralHMM.py --multiplex --inputFile {multiplexFile} --mutToBenef {m} --mutFromBenef {m} --effPopSize {Ne} --yearsPerGen 1 --initFrequency {initFreq} --initTime {initTime} --selection {fixedSel} --dominance 0.5 --precision {dps} --matrixCutoff {matrixCutoff} --maxM {matrixCutoff-10} --maxN {matrixCutoff-20} > {outFile}"

    subprocess.run (cmd.split(" "))
    # read output
    realOutFile = outFile
    themData = pandas.read_csv (realOutFile, delimiter="\t", header=None, comment="#")

    out_dict = {'generations' : momentGens, 'a_traj' : themData[3]}

    pkl.dump(out_dict,  open( out_file, "wb" ) )


if __name__ == '__main__':
    main()