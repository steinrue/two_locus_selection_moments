#!python3

import gzip as gz
import os
import sys
import numpy as np
import pickle as pkl
import itertools
import random
import json

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
order_test = [2*i for i in range(1, 25)]

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

const_dem2000 = 'demographies/constDem2000.txt'
with open(const_dem2000, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(2000))

const_dem10000 = 'demographies/constDem10000.txt'
with open(const_dem10000, 'w+') as dem_file:
    dem_file.writelines('setSize, 0, {}'.format(10000))




# write a multiplex file that corresponds to expected value over time
# taking one focal allele out of sample size 1 should be expected value
# het is 1 out of 2 (even the factor should work)
multiplexFile2 = "expected_traj2000.multi"
ofs = open (multiplexFile2, "w")
momentGens = np.arange (0, 3000, 10)
for g in momentGens:
    # put one sample at this gens
    ofs.write (f"({g}, 1, 1)\n")
ofs.close()

multiplexFile10 = "expected_traj10000.multi"
ofs = open (multiplexFile10, "w")
momentGens = np.arange (0, 6000, 10)
for g in momentGens:
    # put one sample at this gens
    ofs.write (f"({g}, 1, 1)\n")
ofs.close()

# Define wildcards
demographies = ['full', 'bottle', 'growth', 'constant2', 'constant10']
const_demographies = ['constant2', 'constant10']
sigs = [0, .01 , .1, 1, 10, 100]
s = [sig/(4*pop_size) for sig in sigs]
init = ['[0.01,0.0,0.49,0.5]', '[0.03,0.0,0.47,0.5]', '[0.05,0.0,0.45,0.5]', '[0.1,0.0,0.4,0.5]']
r = [10**-8, 10**-2]
it = ['loglin', 'lin', 'project', 'jackknife', 'jackknife-constrained']
renorm = ['renorm', 'renorm-no']
parsimonious = ['parsimonious', 'parsimonious-no']
d_size = [32, 64, 128]
min_step = [10**-4]
prefix = 'scripts/'
sims_out = 'data/simuPopFixed_dem_{demographies}_s_{s}_r_{r}_i_{init}.p'
ode_out = 'ode_output/odeOutput_dem_{demographies}_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.p'
ode_out_orders = 'ode_output/odeOrderOutput_dem_full_s_{s}_r_{r}_i_{init}_minstep_{min_step}_order_{order_test}_dsize_{d_size}.p'
exact_a_traj_fn10 = 'trajectories/aTraj_constant10_s_{s}_i_{init}.txt',
exact_a_traj_fn2 = 'trajectories/aTraj_constant2_s_{s}_i_{init}.txt',
exact_a_traj_fn = 'trajectories/aTraj_{const_demographies}_s_{{s}}_i_{{init}}.txt',
a_fn_ll = 'images/aTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
b_fn_ll = 'images/bTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
ab_fn_ll = 'images/abTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
ld_fn_ll = 'images/ldTraj_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
a_exact_fn_ll = 'images/aExactTraj_{const_demographies}_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
perf_fn = 'data/odePerformance_dem_{demographies}_s_{s}_r_{r}_i_{init}_minstep_{min_step}_{it}_{renorm}_{parsimonious}.csv'
perf_fn_orders = 'data/odeOrderPerformance_dem_full_s_{s}_r_{r}_i_{init}_minstep_{min_step}_order_{order_test}_dsize_{d_size}.csv'
total_perf_fn = 'data/odePerformance_all.csv'


rule all:
    input: 
        expand(sims_out, s=s, r=r, init=init, demographies=demographies),
        expand(ode_out, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious, demographies=demographies),
        expand(ode_out_orders, s=s, r=r, init=init, min_step=min_step, order_test=order_test, d_size=d_size),
        expand(exact_a_traj_fn2, s=s, init=init),
        expand(exact_a_traj_fn10, s=s, init=init),
        expand(a_fn_ll, s=s, r=r, init=init, min_step=min_step),
        expand(b_fn_ll, s=s, r=r, init=init, min_step=min_step),
        expand(ab_fn_ll, s=s, r=r, init=init, min_step=min_step),
        expand(ld_fn_ll, s=s, r=r, init=init, min_step=min_step),
        expand(a_exact_fn_ll, s=s, r=r, init=init, min_step=min_step, const_demographies=const_demographies),
        expand(perf_fn, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious, demographies=demographies),
        expand(perf_fn_orders, s=s, r=r, init=init, min_step=min_step, order_test=order_test, d_size=d_size),
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

# Rule to compute compute moment odes for evaluating performance
rule gen_odes_performance:
    params: 
        order = str(order),
        m = str(m),
        gens = str(gens)
    output: 
        out_file = ode_out
    run:
        shell('python3 ' + prefix + '''odeFixed.py -ord {params.order} -it {wildcards.it} -{wildcards.renorm} -{wildcards.parsimonious} -demog {wildcards.demographies} -m {params.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -out_file {output.out_file} -min_step {wildcards.min_step}
        ''')

# Rule to compute compute moment odes for evaluating different orders
rule gen_odes_order:
    params:
        m = str(m),
        gens = str(gens)
    output: 
        out_file = ode_out_orders
    run:
        shell('python3 ' + prefix + '''odeFixed.py -ord {wildcards.order_test} -it loglin -renorm -parsimonious-no -demog full -m {params.m} -s {wildcards.s} -r {wildcards.r} -init {wildcards.init} -out_file {output.out_file} -min_step {wildcards.min_step} -d_size {wildcards.d_size}
        ''')

# Compare ODEs to Simulations across parameters
rule evaluate_performance_params:
    input: 
        ode_in = ode_out,
        simupop_in = sims_out
    output:
        perf_fn = perf_fn
    run:
        shell('python3 ' + prefix + '''evaluatePerformance.py -oin {input.ode_in} -sin {input.simupop_in} -out {output.perf_fn}''')     


# Compare ODEs to Simulations across orders
rule evaluate_performance_orders:
    input: 
        ode_in = ode_out_orders,
        simupop_in = 'data/simuPopFixed_dem_full_s_{s}_r_{r}_i_{init}.p'
    output:
        perf_fn = perf_fn_orders
    run:
        shell('python3 ' + prefix + '''evaluatePerformance.py -oin {input.ode_in} -sin {input.simupop_in} -out {output.perf_fn}''')    

# Generate exact trajectories
rule compute_exact_trajectories:
    params:
        m = str(m),
        multiplexFile2 = multiplexFile2,
        multiplexFile10 = multiplexFile10,
        dps = str(75),
        matrixCutoff = str(200),
        maxM = str(200-10),
        maxN = str(200-20)
    output:
        traj_fn2 = exact_a_traj_fn2,
        traj_fn10 = exact_a_traj_fn10
    run:
        initFreq = str(json.loads(wildcards.init)[0])
        shell('module load java-jdk/1.6.0_45; python3 scripts/spectralHMM_1_0_0/runSpectralHMM.py --multiplex --inputFile {params.multiplexFile2} --mutToBenef {params.m} --mutFromBenef {params.m} --effPopSize 2000 --yearsPerGen 1 --initFrequency {initFreq} --initTime -2 --selection {wildcards.s} --dominance 0.5 --precision {params.dps} --matrixCutoff {params.matrixCutoff} --maxM {params.maxM} --maxN {params.maxN} >> {output.traj_fn2}')  
        shell('module load java-jdk/1.6.0_45; python3 scripts/spectralHMM_1_0_0/runSpectralHMM.py --multiplex --inputFile {params.multiplexFile10} --mutToBenef {params.m} --mutFromBenef {params.m} --effPopSize 10000 --yearsPerGen 1 --initFrequency {initFreq} --initTime -2 --selection {wildcards.s} --dominance 0.5 --precision {params.dps} --matrixCutoff {params.matrixCutoff} --maxM {params.maxM} --maxN {params.maxN} >> {output.traj_fn10}')

# Generate Plots across parameters
rule gen_plots_parameters:
    input: 
        expand('ode_output/odeOutput_dem_{demographies}_s_{{s}}_r_{{r}}_i_{{init}}_minstep_{{min_step}}_loglin_renorm_parsimonious-no.p', demographies=demographies),
        expand('data/simuPopFixed_dem_{demographies}_s_{{s}}_r_{{r}}_i_{{init}}.p', demographies=demographies),
        exact_a_traj_fn2 = exact_a_traj_fn2,
        exact_a_traj_fn10 = exact_a_traj_fn10
    output:
        a_fn_ll = a_fn_ll,
        b_fn_ll = b_fn_ll,
        ab_fn_ll = ab_fn_ll,
        ld_fn_ll = ld_fn_ll,
        a_exact_fn_ll2 = 'images/aExactTraj_constant2_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf',
        a_exact_fn_ll10 = 'images/aExactTraj_constant10_s_{s}_r_{r}_i_{init}_minstep_{min_step}_loglin_renorm_parsimonious-no.pdf'
    run:
        shell('python3 ' + prefix + '''generateFixedPlots.py -oin ode_output/odeOutput_dem_demographies_s_{wildcards.s}_r_{wildcards.r}_i_{wildcards.init}_minstep_{wildcards.min_step}_loglin_renorm_parsimonious-no.p -sin data/simuPopFixed_dem_demographies_s_{wildcards.s}_r_{wildcards.r}_i_{wildcards.init}.p -ein2 {input.exact_a_traj_fn2} -ein10 {input.exact_a_traj_fn10}  -aout {output.a_fn_ll} -bout {output.b_fn_ll} -about {output.ab_fn_ll} -ldout {output.ld_fn_ll} -eout2 {output.a_exact_fn_ll2} -eout10 {output.a_exact_fn_ll10}''')

rule aggregate_performance:
    input:
        expand(perf_fn_orders, s=s, r=r, init=init, min_step=min_step, order_test=order_test, d_size=d_size), 
        expand(perf_fn, demographies=demographies, s=s, r=r, init=init, min_step=min_step, it=it, renorm=renorm, parsimonious=parsimonious)
    output:
        out = total_perf_fn
    run:
        shell('cat data/ode* > {output.out}')


