#!python3

import gzip as gz
import os
import sys
import numpy as np
import pickle as pkl
import itertools
import random

seeed=4711
random.seed(seeed)

# Set parameters
pop_size = 10000
sim_size=1000
bot_gen = 6000
exp_gen = 9000
bot_size = 2000
gens = 10000
growth_rate = np.log(1.0025)
reps = 1000
num_loci = 41
m=[1.25*10**-8]
order = 31

# Generate demographies
full_dem = 'demographies/constBotExpDem.txt'
with open(full_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(pop_size))
    dem_file.write('\n')
    dem_file.writelines('setSize, {}, {}'.format(bot_gen, bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(exp_gen, growth_rate))

bott_dem = 'demographies/botExpDem.txt'
with open(bott_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(pop_size))
    dem_file.write('\n')
    dem_file.writelines('setSize, 1, {}'.format(bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(exp_gen-bot_gen, growth_rate))

exp_dem = 'demographies/expDem.txt'
with open(exp_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(0, growth_rate))

# Define wildcards
demographies = ['full', 'bottle', 'growth']
sigs = [1, 50, 100]
s = [sig/(4*pop_size) for sig in sigs]
init = [.01, .03, .05]
windows = [10**5]
r = [10**-8]
min_step = [10**-4]

# Define output files
prefix='scripts/'
sims_out = 'data/simuPopStationary_dem_{demographies}_m_{m}_s_{s}_r_{r}_window_{windows}_i_{init}.p'
ode_out = 'ode_output/odeOutput_dem_{demographies}_m_{m}_s_{s}_r_{r}_window_{windows}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.p'
hz_fn = 'images/hzPlot_dem_{demographies}_m_{m}_s_{s}_r_{r}_window_{windows}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
d2_fn = 'images/d2Plot_dem_{demographies}_m_{m}_s_{s}_r_{r}_window_{windows}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
hzgrid_fn = 'images/hzGridPlot_m_{m}_r_{r}_window_{windows}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
d2grid_fn = 'images/d2GridPlot_m_{m}_r_{r}_window_{windows}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'


rule all:
    input: 
        expand(sims_out, s=s, r=r, windows=windows, init=init, m=m, demographies=demographies),
        expand(ode_out, s=s, r=r, windows=windows, init=init, min_step=min_step, m=m, demographies=demographies),
        expand(hzgrid_fn, r=r, windows=windows, init=init, min_step=min_step, m=m),
        expand(d2grid_fn, r=r, windows=windows, init=init, min_step=min_step, m=m)
        
# Run simupop simulations       
rule gen_sims:
    params:
        N = str(pop_size),
        num_loci = str(num_loci),
        bot_gen = str(bot_gen),
        exp_gen = str(exp_gen),
        bot_size = str(bot_size),
        growth_rate = str(growth_rate),
        gens = str(gens),
        reps=str(reps),
        seed = str(seeed),
        sim_size = str(sim_size)
    output: 
        out_file = sims_out
    run:
        shell('module load miniconda3/4.3.21; python3 ' + prefix + '''simupopSimsStationary.py -n {params.sim_size} -truen {params.N} -num_loci {params.num_loci} -reps {params.reps} -s {wildcards.s} -m {wildcards.m} -r {wildcards.r} -init {wildcards.init}  -window {wildcards.windows} -seed {params.seed} -demo {wildcards.demographies} -out_file {output.out_file}''') 

# Get output from moment odes
rule gen_odes:
    params: 
        order = str(order),
        gens = str(gens),
        num_loci = str(num_loci)
    output: 
        out_file = ode_out
    run:
        shell('python3 ' + prefix + '''odeStationary.py -ord {params.order} -demog {wildcards.demographies} -m {wildcards.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -out_file {output.out_file} -min_step {wildcards.min_step} -num_loci {params.num_loci} -window {wildcards.windows}
        ''')

# Generate plots
rule gen_plots:
    params:
        s_vals = str(s).replace(' ', '')
    input: 
        expand('data/simuPopStationary_dem_{demographies}_m_{{m}}_s_{s}_r_{{r}}_window_{{windows}}_i_{{init}}.p', demographies=demographies, s=s),
        expand('ode_output/odeOutput_dem_{demographies}_m_{{m}}_s_{s}_r_{{r}}_window_{{windows}}_i_{{init}}_minstep_{{min_step}}_loglin_renorm_parsimonious-no.p', demographies=demographies, s=s)
    output:
        hzgrid_fn = hzgrid_fn,
        d2grid_fn = d2grid_fn
    run:
        shell('python3 ' + prefix + '''generateStationaryPlots.py -hzgridout {output.hzgrid_fn} -d2gridout {output.d2grid_fn} -m {wildcards.m} -r {wildcards.r} -window {wildcards.windows} -init {wildcards.init} -s_vals {params.s_vals}''')