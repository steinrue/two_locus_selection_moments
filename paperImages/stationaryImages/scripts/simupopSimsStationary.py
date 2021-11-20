from simuOpt import setOptions
setOptions(optimized=True, alleleType='short', numThreads=1)
import simuPOP as sim
import simuPOP.demography as simdem
import numpy as np
import pickle as pkl
import json
import click

@click.command()
@click.option('-n', help='Population Size', type=int)
@click.option('-truen', help='Population Size of simulation', type=int)
@click.option('-num_loci', help='Population Size of simulation', type=int)
@click.option('-reps', help='Number of repetitions', type=int)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='mutation rate', type=str)
@click.option('-window', help='size of window to model', type=int)
@click.option('-seed', help='mutation rate', type=int)
@click.option('-demo', help='demographic model', type=str)
@click.option('-out_file', help='Filename to output to', type=str)
def main(n, truen, num_loci, reps, s, m, r, init, window, seed, demo, out_file):

    # Rescale recombination and mutation rates
    m = m * window / 2 / (num_loci-1)
    r = r * window / 2 / (num_loci-1)
    
    scale_ratio = truen/n
    init = json.loads(init)
    sim.setRNG(seed=seed)
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
    elif demo == 'bottle':
        model = simdem.MultiStageModel([
                    simdem.InstantChangeModel(T=int(.3*n), N0=n, G=1, NG=bot_size_sim),
                    simdem.ExponentialGrowthModel(T=int(.1*n), N0=bot_size_sim, NT=end_size_sim)])
        gens_sim = int(.4*n)
    elif demo == 'growth':
        model = simdem.MultiStageModel([
                    simdem.InstantChangeModel(T=int(.3*n), N0=n, G=1, NG=bot_size_sim),
                    simdem.ExponentialGrowthModel(T=int(.1*n), N0=bot_size_sim, NT=end_size_sim)])
        burnin_sim = int(.3*n)
        gens_sim = int(.1*n)
    

    selLocus = 0
    loci = [i for i in range(num_loci+1)]
    pop = sim.Population(size=int(n), ploidy=2, loci=[num_loci+1],
                         infoFields='fitness')
    sim.initSex(pop, sex=[sim.MALE, sim.FEMALE], subPops=0)
    sim.stat(pop, alleleFreq=loci)
    rec_rates = [0] + [r]*(num_loci)
    haploFreqs_init = []
    ld_sets = [(selLocus, i, 1, 1) for i in loci[1:]]
    out= []
    for k in range(reps):
        pop1 = pop.clone()
        means = np.random.beta(4*n*m, 4*n*m, size=num_loci)
        genotype_focal =  [np.random.binomial(1, p) for p in means]
        for i, ind in enumerate(pop1.individuals()):
                if demo != 'growth':
                    initial_A = pop1.popSize()*init
                else:
                    initial_A = 0
                if i < initial_A:
                    genotype = tuple([1] + genotype_focal)
                    ind.setGenotype(genotype, ploidy=0)
                    genotype = tuple([1] + genotype_focal)
                    ind.setGenotype(genotype, ploidy=1)
                else:
                    genotype_non_focal = [np.random.binomial(1, p) for p in means]
                    genotype = tuple([0] + genotype_non_focal)
                    ind.setGenotype(genotype, ploidy=0)
                    genotype_non_focal = [np.random.binomial(1, p) for p in means]
                    genotype = tuple([0] + genotype_non_focal)
                    ind.setGenotype(genotype, ploidy=1)
        pop1.dvars().freqA = []
        pop1.dvars().ld = []
        if demo == 'growth':
            g = pop1.evolve(
                    preOps=[sim.SNPMutator(u=m, v=m)],
                    matingScheme=sim.RandomMating(
                    ops=sim.Recombinator(rates=rec_rates, loci=loci), subPopSize=model),
                    finalOps=[sim.Stat(alleleFreq=loci, LD=ld_sets),
                                        sim.PyExec('ld.append([LD[%d][i] for i in %s])' % (selLocus, loci[1:])),
                                        sim.PyExec('freqA.append(alleleFreq)')],
                    gen = burnin_sim)
            first_mutant = pop1.individual(0).genotype(ploidy=0)
            first_mutant[0] = 1
            initial_A = pop1.popSize()*init
            for i, ind in enumerate(pop1.individuals()):
                if i < initial_A:
                    ind.setGenotype(first_mutant, ploidy=0)
                    ind.setGenotype(first_mutant, ploidy=1)
                else:
                    genotype = ind.genotype(ploidy=0)
                    genotype[0] = 0
                    ind.setGenotype(genotype, ploidy=0)
                    genotype = ind.genotype(ploidy=1)
                    genotype[0] = 0
                    ind.setGenotype(genotype, ploidy=1)
        
        g = pop1.evolve(initOps=[sim.Stat(alleleFreq=loci, LD=ld_sets),
                                        sim.PyExec('ld.append([LD[%d][i] for i in %s])' % (selLocus, loci[1:])),
                                        sim.PyExec('freqA.append(alleleFreq)')],
                        preOps=[ sim.MapSelector(loci=selLocus, fitness={(0, 0): 1, (0, 1): 1+s, (1, 1): 1+2*s}),
                                        sim.SNPMutator(u=m, v=m)
                                    ],
                            matingScheme=sim.RandomMating(ops=sim.Recombinator(rates=rec_rates, loci=loci), subPopSize=model),
                            postOps=[sim.Stat(alleleFreq=loci, LD=ld_sets),
                                        sim.PyExec('ld.append([LD[%d][i] for i in %s])' % (selLocus, loci[1:])),
                                        sim.PyExec('freqA.append(alleleFreq)')],
                            gen = gens_sim)
        out.append([pop1.dvars().freqA, pop1.dvars().ld])
    
    a_freq_simupop = []
    for loc in loci:
        a_freq = []
        for rep in out:
            a_freq.append([rep_step[loc][1] for rep_step in rep[0]])
        a_freq_simupop.append(a_freq)

    ld_freqs_simupop = []
    for loc in loci[1:]:
        ld_freqs = []
        for rep in out:
            ld_freqs.append([rep_step[loc-1] for rep_step in rep[1]])
        ld_freqs_simupop.append(ld_freqs)

    out_dict = {}
    out_dict['params'] = {'s' : s, 'init' : init, 'r' : r, 'bot' : bot_gen_sim, 'exp' : exp_gen_sim, 'gen': gens_sim, 'demog' : demo}
    out_dict['data'] = {'a' : a_freq_simupop, 'ld' :  ld_freqs_simupop}
    
    pkl.dump( out_dict, open( out_file, "wb" ) )

if __name__ == '__main__':
    main()
