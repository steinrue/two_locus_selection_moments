from simuOpt import setOptions
setOptions(optimized=False, alleleType='short', numThreads=1)
import simuPOP as sim
import simuPOP.demography as simdem
import numpy as np
import pickle as pkl
import json
import click
from hashlib import blake2b

@click.command()
@click.option('-n', help='Population Size of Simulation', type=int)
@click.option('-truen', help='Population Size of model', type=int)
@click.option('-reps', help='Number of repetitions', type=int)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='Initial haplotype frequencies', type=str)
@click.option('-seed', help='Starting seed', type=int)
@click.option('-demo', help='Location of demography file', type=str)
@click.option('-out_file', help='Filename to output to', type=str)
def main(n, truen, reps, s, m, r, init, seed, demo, out_file):

    seed = blake2b((out_file + str(seed)).encode(), digest_size=4)
    # Initialize parameters (making sure teo scale down modeled parameters to simulation size)
    s = s
    scale_ratio = truen/n
    init = json.loads(init)
    sim.setRNG(seed=int.from_bytes(seed.digest(), "big"))
    growth_rate = np.log(1.0025)
    s = s * scale_ratio
    m = m * scale_ratio
    r = r * scale_ratio
    bot_size_sim = int(.2*n)
    bot_gen_sim = int(.3*n)
    exp_gen_sim = int(.1*n)
    end_size = int(.2*truen) * np.exp(growth_rate * .1*truen)
    end_size_sim = int(end_size / scale_ratio)   
    growth_rate_sim = np.log(end_size_sim/bot_size_sim)/exp_gen_sim
    if demo == 'full':
        model = simdem.MultiStageModel([
        simdem.InstantChangeModel(T=int(.9*n), N0=n, G=int(.6*n), NG=bot_size_sim),
        simdem.ExponentialGrowthModel(T=exp_gen_sim, N0=bot_size_sim, NT=end_size_sim)])
        gens_sim = n
        bot_save = int(.6*n)
        exp_save = int(.9*n)
        full_save = int(n)
    elif demo == 'bottle':
        model = simdem.MultiStageModel([
        simdem.InstantChangeModel(T=bot_gen_sim, N0=bot_size_sim),
        simdem.ExponentialGrowthModel(T=exp_gen_sim, N0=bot_size_sim, NT=end_size_sim)])
        gens_sim = exp_gen_sim + bot_gen_sim
        bot_save = int(0)
        exp_save = int(.3*n)
        full_save = int(.4*n)
    elif demo == 'growth':
        model = simdem.ExponentialGrowthModel(T=exp_gen_sim, N0=bot_size_sim, NT=end_size_sim)
        gens_sim = exp_gen_sim
        bot_save = int(0)
        exp_save = int(0)
        full_save = int(.1*n)
    elif demo == 'constant2':
        model = 200
        gens_sim = 300
        bot_save = int(0)
        exp_save = int(0)
        full_save = int(gens_sim)
    elif demo == 'constant10':
        model =1000
        gens_sim = 600
        bot_save = int(0)
        exp_save = int(0)
        full_save = int(gens_sim)

    
    
    
    # initialize population
    selLocus = 0
    num_loci = 2
    loci = [i for i in range(num_loci)]
    pop = sim.Population(size=int(n), ploidy=2, loci=[num_loci],
                         infoFields='fitness')
    hap11 = [1]*num_loci
    hap10 = [0]*num_loci
    hap10[0] = 1
    hap01 = [1]*num_loci
    hap01[0] = 0
    hap00 = [0]*num_loci
    sim.initGenotype(pop, haplotypes=[hap11, hap10, hap01, hap00], prop = init)
    sim.initSex(pop, sex=[sim.MALE, sim.FEMALE], subPops=0)
    pop.dvars().freqAB = []
    pop.dvars().ld = []
    sim.stat(pop, alleleFreq=loci)
    haploFreqs_init = []

    # Run simulations
    out= []
    for k in range(reps):
        print(k)
        pop1 = pop.clone()
        pop1.dvars().freqAB = []
        pop1.dvars().ld = []
        g = pop1.evolve(initOps=[sim.Stat(alleleFreq=loci, LD=(0, 1, 1, 1), haploFreq=(0, 1)),
                                        sim.PyExec('ld.append(LD)'),
                                        sim.PyExec('freqAB.append(haploFreq[(0, 1)])')],
                            preOps=[
                            sim.Stat(alleleFreq=loci),  
                                sim.MapSelector(loci=selLocus, fitness={(0, 0): 1, (0, 1): 1+s, (1, 1): 1+2*s}),
                                        sim.SNPMutator(u=m, v=m)
                                    ],
                            matingScheme=sim.RandomMating(ops=sim.Recombinator(rates=r), subPopSize=model),
                            postOps=[sim.Stat(alleleFreq=loci, LD=(0, 1, 1, 1), haploFreq=(0, 1)),
                                        sim.PyExec('ld.append(LD)'),
                                        sim.PyExec('freqAB.append(haploFreq[(0, 1)])')],
                            gen = gens_sim)
        out.append([pop1.dvars().freqAB, pop1.dvars().ld])

    # Unpack simulated data
    a_freqs_simupop = []
    for rep in out:
        a_freqs_simupop.append([rep_step[(1, 1)] + rep_step[(1, 0)] for rep_step in rep[0]])
    a_freqs_simupop = np.array(a_freqs_simupop)

    b_freqs_simupop = []
    for rep in out:
        b_freqs_simupop.append([rep_step[(0, 1)] + rep_step[(1, 1)] for rep_step in rep[0]])
    b_freqs_simupop = np.array(b_freqs_simupop)

    ab_freqs_simupop = []
    for rep in out:
        ab_freqs_simupop.append([rep_step[(1, 1)] for rep_step in rep[0]])
    ab_freqs_simupop = np.array(ab_freqs_simupop)

    ld_freqs_simupop = []
    for rep in out:
        ld_freqs_simupop.append([rep_step[0][1] for rep_step in rep[1]])
    ld_freqs_simupop = np.array(ld_freqs_simupop)

    hap_freqs_bot = []
    for rep in out:
        hap_freqs_bot.append([rep[0][bot_save][(1,1)], rep[0][bot_save][(1,0)], rep[0][bot_save][(0,1)], rep[0][bot_save][(0,0)]])
    hap_freqs_bot = np.array(hap_freqs_bot)

    hap_freqs_exp = []
    for rep in out:
        hap_freqs_exp.append([rep[0][exp_save][(1,1)], rep[0][exp_save][(1,0)], rep[0][exp_save][(0,1)], rep[0][exp_save][(0,0)]])
    hap_freqs_exp = np.array(hap_freqs_exp)

    hap_freqs_gen = []
    for rep in out:
        hap_freqs_gen.append([rep[0][full_save][(1,1)], rep[0][full_save][(1,0)], rep[0][full_save][(0,1)], rep[0][full_save][(0,0)]])
    hap_freqs_gen = np.array(hap_freqs_gen)

    # Save simulation parameters and simulated data
    out_dict = {}
    out_dict['params'] = {'s' : s, 'init' : init, 'r' : r, 'bot' : bot_save, 'exp' : exp_save, 'gen': full_save, 'demog' : demo}
    out_dict['data'] = {'a' : a_freqs_simupop, 'b' : b_freqs_simupop, 'ab' : ab_freqs_simupop, 'ld' :  ld_freqs_simupop, 'bot' : hap_freqs_bot,
                        'exp' :  hap_freqs_exp, 'tot' : hap_freqs_gen, 'demo' : demo}
    
    pkl.dump( out_dict, open( out_file, "wb" ) )

if __name__ == '__main__':
    main()