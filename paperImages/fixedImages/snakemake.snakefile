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

# Initialize parameters
pop_size = 10000
sim_size = 1000
bot_gen = 6000
exp_gen = 9000
bot_size = 2000
gens = 10000
growth_rate = np.log(1.0025)
reps = 1000
m = 1.25*10**-8
order = 31

# Initialize demographies
full_dem = 'demographies/constBotExpDem.txt'
with open(full_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(pop_size))
    dem_file.write('\n')
    dem_file.writelines('setSize, {}, {}'.format(bot_gen, bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(exp_gen, growth_rate))

bott_dem = 'demographies/botExpDem.txt'
with open(bott_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(exp_gen-bot_gen, growth_rate))

exp_dem = 'demographies/expDem.txt'
with open(exp_dem, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(0, growth_rate))

# Define wildcards
demographies = ['full', 'bottle', 'growth']
sigs = [0, .01 , .1, 1, 10, 100]
s = [sig/(4*pop_size) for sig in sigs]
init = ['[0.01,0.0,0.49,0.5]', '[0.03,0.0,0.47,0.5]', '[0.05,0.0,0.45,0.5]', '[0.1,0.0,0.4,0.5]']
r = [10**-8, 10**-2]
it = ['loglin', 'lin', 'project', 'jackknife', 'jackknife-constrained']
renorm = ['renorm', 'renorm-no']
parsimonious = ['parsimonious', 'parsimonious-no']
min_step = [10**-4]
prefix = 'scripts/'
sims_out = 'data/simuPopFixed_dem_{demographies}_s_{s}_r_{r}_i_{init}.p'
ode_out = 'ode_output/odeOutput_dem_{demographies}_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.p'
a_fn = 'images/aTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.pdf'
b_fn = 'images/bTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.pdf'
ab_fn = 'images/abTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.pdf'
ld_fn = 'images/ldTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.pdf'
perf_fn = 'data/odePerformance_dem_{demographies}_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.csv'
total_perf_fn = 'data/odePerformance_all.csv'


rule all:
    input: 
        expand(sims_out, s=s, r=r, init=init, demographies=demographies),
        expand(ode_out, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious, demographies=demographies),
        expand(a_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious),
        expand(b_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious),
        expand(ab_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious),
        expand(ld_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious),
        expand(perf_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious, demographies=demographies),
        expand(total_perf_fn)

 # Rule to generate simuPoP simulations       
rule gen_sims:
    params:
        N = str(pop_size),
        bot_gen = str(bot_gen),
        exp_gen = str(exp_gen),
        bot_size = str(bot_size),
        growth_rate = str(growth_rate),
        gens = str(gens),
        reps=str(reps),
        m=str(m),
        seed = str(seeed),
        sim_size = str(sim_size)
    output: 
        out_file = sims_out
    run:
        shell('module load miniconda3/4.3.21; python3 ' + prefix + '''simupopSimsFixed.py -n {params.sim_size} -truen {params.N} -reps {params.reps} -s {wildcards.s} -m {params.m} -r {wildcards.r} -init {wildcards.init} -seed {params.seed} -demo {wildcards.demographies} -out_file {output.out_file}''')

# Rule to compute compute moment odes
rule gen_odes:
    params: 
        order = str(order),
        m = str(m),
        gens = str(gens)
    output: 
        out_file = ode_out
    run:
        shell('python3 ' + prefix + '''odeFixed.py -ord {params.order} -it {wildcards.it} -{wildcards.renorm} -{wildcards.parsimonious} -demog {wildcards.demographies} -m {params.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -out_file {output.out_file} -min_step {wildcards.min_step}
        ''')

# Compare ODEs to Simulations
rule evaluate_performance:
    input: 
        ode_in = ode_out,
        simupop_in = sims_out
    output:
        perf_fn = perf_fn
    run:
        shell('python3 ' + prefix + '''evaluatePerformance.py -oin {input.ode_in} -sin {input.simupop_in} -out {output.perf_fn}''')     

# Generate Plots
rule gen_plots:
    input: 
        expand('ode_output/odeOutput_dem_{demographies}_s_{{s}}_r_{{r}}_i_{{init}}_minstep_{{min_step}}_loglin_renorm_parsimonious-no.p', demographies=demographies),
        expand('data/simuPopFixed_dem_{demographies}_s_{{s}}_r_{{r}}_i_{{init}}.p', demographies=demographies)
    output:
        a_fn = a_fn,
        b_fn = b_fn,
        ab_fn = ab_fn,
        ld_fn = ld_fn
    run:
        shell('python3 ' + prefix + '''generateFixedPlots.py -oin ode_output/odeOutput_dem_demographies_s_{wildcards.s}_r_{wildcards.r}_i_{wildcards.init}_minstep_{wildcards.min_step}_loglin_renorm_parsimonious-no.p -sin data/simuPopFixed_dem_demographies_s_{wildcards.s}_r_{wildcards.r}_i_{wildcards.init}.p -ain {output.a_fn} -bin {output.b_fn} -abin {output.ab_fn} -ldin {output.ld_fn}''')

rule aggregate_performance:
    input:
        expand(perf_fn, demographies=demographies, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious)
    output:
        out = total_perf_fn
    run:
        shell('cat data/odePerformance* > {output.out}')


