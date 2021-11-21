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


# ---- Base Rule ----------- #

pop_size = 10000
sim_size=1000
bot_gen = 6000
exp_gen = 9000
bot_size = 2000
gens = 10000
growth_rate = np.log(1.0025)
reps = 1000
num_loci = [10**6+1, 10**6//2+1, 10**5+1]

saves = [1000*i for i in range(11)]
m = [1.25*10**-8]
order = 31
orders = [31, 51, 71, 101]
demog_fn = 'constBotExpDem.txt'
with open(demog_fn, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(pop_size))
    dem_file.write('\n')
    dem_file.writelines('setSize, {}, {}'.format(bot_gen, bot_size))
    dem_file.write('\n')
    dem_file.write('expGrow, {}, {}'.format(exp_gen, growth_rate))


sigs = [1, 50, 100]
s = [sig/(4*pop_size) for sig in sigs]
init = [.05]
r = [10**-8]
it = ['loglin']
renorm = ['renorm']
clip_ind = ['clip']
parsimonious = ['parsimonious-no']
folded = ['folded', 'unfolded']
includeFixed = ['includefixed', 'excludefixed']
min_step = [10**-4]
prefix='scripts/'
sims_out = 'data/slim_m_{m}_s_{s}_r_{r}_i_{init}_win_{num_loci}.p'
ode_out = 'ode_output/odeOutput_m_{m}_s_{s}_r_{r}_i_{init}_win_{num_loci}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}.p'
ode_out_rags = 'ode_output/odeOutputRags_m_{m}_s_{s}_r_{r}_i_{init}_win_{num_loci}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}.p'
sfs_out = 'images/sfs_m_{m}_s_{s}_r_{r}_i_{init}_win_{num_loci}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}.pdf'
grid_out = 'images/sfsgrid_m_{m}_r_{r}_i_{init}_order_{order}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}_{includeFixed}.pdf'
grid_log_out = 'images/sfsloggrid_m_{m}_r_{r}_i_{init}_order_{order}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}_excludefixed.pdf'
sfs_out_rags = 'images/sfsrags_m_{m}_s_{s}_r_{r}_i_{init}_win_{num_loci}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}.pdf'
grid_out_rags = 'images/sfsgridrags_m_{m}_r_{r}_i_{init}_order_{order}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}_{includeFixed}.pdf'
grid_log_out_rags = 'images/sfsloggridrags_m_{m}_r_{r}_i_{init}_order_{order}_minstep_{min_step}_{it}_{renorm}_{clip_ind}_{parsimonious}_{folded}_excludefixed.pdf'


rule all:
    input: 
        expand(sims_out, s=s, r=r, init=init, m=m, num_loci=num_loci),
        expand(ode_out, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, num_loci=num_loci, folded=folded),
        expand(ode_out_rags, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, num_loci=num_loci, folded=folded),
        expand(grid_out, r=r, init=init, min_step=min_step, it=it, order=orders, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, folded=folded, includeFixed=includeFixed),
        expand(grid_out_rags, r=r, init=init, min_step=min_step, it=it, order=orders, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, folded=folded, includeFixed=includeFixed),
        expand(grid_log_out, r=r, init=init, min_step=min_step, it=it, order=orders, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, folded=folded),
        expand(grid_log_out_rags, r=r, init=init, min_step=min_step, it=it, order=orders, renorm=renorm, clip_ind=clip_ind, parsimonious=parsimonious, m=m, folded=folded)

        
rule gen_sims:
    params:
        N = str(pop_size),
        bot_gen = str(bot_gen),
        exp_gen = str(exp_gen),
        bot_size = str(bot_size),
        growth_rate = str(growth_rate),
        gens = str(gens),
        reps=str(reps),
        seed = str(seeed),
        sim_size = str(sim_size),
        saves = str(saves).replace(' ', ''),
        order = str(order)
    output: 
        out_file = sims_out
    run:
        shell('module purge; module load gcc/6.2.0; module load python/3.6.0; module load gsl/2.3; module load slim/3.2.1; python3 ' + prefix + '''slimSims.py -n {params.sim_size} -truen {params.N} -nl {wildcards.num_loci} -reps {params.reps} -s {wildcards.s} -m {wildcards.m} -r {wildcards.r} -init {wildcards.init} -bot_gen {params.bot_gen} -exp_gen {params.exp_gen} -bot_size {params.bot_size} -growth_rate {params.growth_rate} -gens {params.gens} -seed {params.seed} -saves {params.saves} -order {params.order} -out_file {output.out_file}''') 




rule gen_odes:
    params: 
        order = str(order),
        demog_fn = demog_fn,
        gens = str(saves).replace(' ', ''),
        num_loci = str(num_loci)
    output: 
        out_file = ode_out
    run:
        shell('python3 ' + prefix + '''odeChrome.py -ord {params.order} -it {wildcards.it} -{wildcards.renorm} -{wildcards.clip_ind} -{wildcards.parsimonious} -demog_fn {params.demog_fn} -m {wildcards.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -gens {params.gens} -out_file {output.out_file} -min_step {wildcards.min_step} -num_loci {wildcards.num_loci} -{wildcards.folded}
        ''')

rule gen_odes_rags:
    params: 
        order = str(order),
        demog_fn = demog_fn,
        gens = str(saves).replace(' ', ''),
        num_loci = str(num_loci)
    output: 
        out_file = ode_out_rags
    run:
        shell('python3 ' + prefix + '''odeChromeRagsdale.py -ord {params.order} -it {wildcards.it} -{wildcards.renorm} -{wildcards.clip_ind} -{wildcards.parsimonious} -demog_fn {params.demog_fn} -m {wildcards.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -gens {params.gens} -out_file {output.out_file} -min_step {wildcards.min_step} -num_loci {wildcards.num_loci} -{wildcards.folded}
        ''')
        
rule gen_plots:
    params:
        s_vals = str(s).replace(' ', ''),
        window_vals = str(num_loci).replace(' ', '')
    input: 
        expand('data/slim_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}.p', s=s, num_loci=num_loci),
        expand('ode_output/odeOutput_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{{folded}}.p', s=s, num_loci=num_loci)
    output:
        expand('images/sfsgrid_m_{{m}}_r_{{r}}_i_{{init}}_order_{order}_minstep_{{min_step}}_{{it}}_{{renorm}}_{{clip_ind}}_{{parsimonious}}_{{folded}}_{{includeFixed}}.pdf',  order=orders)
    run:
        shell('python3 ' + prefix + '''sfsPlots.py -m {wildcards.m} -r {wildcards.r} -init {wildcards.init} -s_vals {params.s_vals} -window_vals {params.window_vals} -{wildcards.folded} -{wildcards.includeFixed} -reg''')

rule gen_log_plots:
    params:
        s_vals = str(s).replace(' ', ''),
        window_vals = str(num_loci).replace(' ', '')
    input: 
        expand('data/slim_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}.p', s=s, num_loci=num_loci),
        expand('ode_output/odeOutput_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{{folded}}.p', s=s, num_loci=num_loci)
    output:
        expand('images/sfsloggrid_m_{{m}}_r_{{r}}_i_{{init}}_order_{order}_minstep_{{min_step}}_{{it}}_{{renorm}}_{{clip_ind}}_{{parsimonious}}_{{folded}}_excludefixed.pdf',  order=orders)
    run:
        shell('python3 ' + prefix + '''sfsLogPlots.py -m {wildcards.m} -r {wildcards.r} -init {wildcards.init} -s_vals {params.s_vals} -window_vals {params.window_vals} -{wildcards.folded} -reg''')

rule gen_plots_rags:
    params:
        s_vals = str(s).replace(' ', ''),
        window_vals = str(num_loci).replace(' ', '')
    input: 
        expand('data/slim_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}.p', s=s, num_loci=num_loci),
        expand('ode_output/odeOutputRags_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{{folded}}.p', s=s, num_loci=num_loci)
    output:
        expand('images/sfsgridrags_m_{{m}}_r_{{r}}_i_{{init}}_order_{order}_minstep_{{min_step}}_{{it}}_{{renorm}}_{{clip_ind}}_{{parsimonious}}_{{folded}}_{{includeFixed}}.pdf',  order=orders)
    run:
        shell('python3 ' + prefix + '''sfsPlots.py -m {wildcards.m} -r {wildcards.r} -init {wildcards.init} -s_vals {params.s_vals} -window_vals {params.window_vals} -{wildcards.folded}  -{wildcards.includeFixed} -rags''')

rule gen_log_plots_rags:
    params:
        s_vals = str(s).replace(' ', ''),
        window_vals = str(num_loci).replace(' ', '')
    input: 
        expand('data/slim_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}.p', s=s, num_loci=num_loci),
        expand('ode_output/odeOutputRags_m_{{m}}_s_{s}_r_{{r}}_i_{{init}}_win_{num_loci}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{{folded}}.p', s=s, num_loci=num_loci)
    output:
        expand('images/sfsloggridrags_m_{{m}}_r_{{r}}_i_{{init}}_order_{order}_minstep_{{min_step}}_{{it}}_{{renorm}}_{{clip_ind}}_{{parsimonious}}_{{folded}}_excludefixed.pdf',  order=orders)
    run:
        shell('python3 ' + prefix + '''sfsLogPlots.py -m {wildcards.m} -r {wildcards.r} -init {wildcards.init} -s_vals {params.s_vals} -window_vals {params.window_vals} -{wildcards.folded} -rags''')


