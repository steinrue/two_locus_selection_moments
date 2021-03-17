import os
import numpy as np
import pickle as pkl
import json
import tskit, pyslim
import sys
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion/paperImages/slimImages")
from scipy.stats import hypergeom
import click

@click.command()
@click.option('-n', help='Population Size of Simulation', type=int)
@click.option('-truen', help='Population Size', type=int)
@click.option('-nl', '--num_loci', help='Number of Loci to Simulate', type=int)
@click.option('-reps', help='Number of repetitions', type=int)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='Initial frequency of selected allele', type=float)
@click.option('-bot_gen', help='Generation of Bottleneck', type=int)
@click.option('-exp_gen', help='Generation of Exponential Grown', type=int)
@click.option('-bot_size', help='Size of bottleneck', type=int)
@click.option('-growth_rate', help='Population growth rate for original population size', type=float)
@click.option('-gens', help='Total number of generations', type=int)
@click.option('-seed', help='Seed', type=int)
@click.option('-saves', help='generations at which to save', type=str)
@click.option('-order', help='Sample size of interest', type=int)
@click.option('-out_file', help='Filename to output to', type=str)
def main(n, truen, num_loci, reps, s, m, r,  init, bot_gen, exp_gen, bot_size, growth_rate, gens, seed, saves, order, out_file):

    # Recale parameters for simulation
    saves = json.loads(saves)
    scale_ratio = truen/n
    s = s * scale_ratio * 2
    m = m * scale_ratio
    r = r * scale_ratio
    end_bottle_size = bot_size * np.exp(growth_rate * (gens-exp_gen))
    bot_gen_sim = int(bot_gen / scale_ratio)
    exp_gen_sim = int(exp_gen / scale_ratio)
    bot_size_sim = int(bot_size / scale_ratio)
    gens_sim = int(gens / scale_ratio)
    end_bottle_size_sim = int(end_bottle_size / scale_ratio)
    growth_rate_sim = np.log(end_bottle_size_sim/bot_size_sim)/(gens_sim-exp_gen_sim) # Must rescale growth rate
    saves = [i/scale_ratio for i in saves]
    
    # Initialize SLiM script strings
    max_gens = int(gens_sim*10)
    script_init = """
    initialize() {{
        initializeTreeSeq();
        defineConstant("N0", {});
        defineConstant("s", {});
        defineConstant("L", {});
        defineConstant("r", {});
        initializeMutationRate(0);
        initializeMutationType("m1", 0.5, "f", 0.0);
        initializeMutationType("m2", 0.5, "f", s); // introduced mutation
        initializeGenomicElementType("g1", m1, 1.0);
        initializeGenomicElement(g1, 0, L);
        initializeRecombinationRate(r);
    }}""".format(n, s, num_loci-1, r)

    script_begin1 = """
    1 late() {{ 
        defineConstant("BT", {});
        defineConstant("BS", {});
        defineConstant("ET", {});
        defineConstant("GR", {});
        defineConstant("TT", {});
        """.format(bot_gen_sim, bot_size_sim, exp_gen_sim, growth_rate_sim, gens_sim)
    
    script_setseed = """
        setSeed({seed});
        """
    
    script_begin2 = """
        defineConstant("simID", getSeed());
        sim.addSubpop("p1", N0); 

        sim.outputFull("/tmp/slim_" + simID + ".txt");

        // introduce the sweep mutation
        target = sample(p1.genomes, 1);
        target.addNewDrawnMutation(m2, asInteger(L/2)); 
    }"""

    # Have demography start after focal allele reaches initial frequency
    script_reschedule_saves = ''
    script_save_blocks = ''
    for i, time in enumerate(saves):
        if time != 0:
            block_name = 's' + str(i+3)
            reschedule_snippet = '''
            
            sim.rescheduleScriptBlock({bn}, sim.generation+{tm}, sim.generation+{tm});

            '''.format(bn=block_name, tm=int(time))
            script_reschedule_saves += reschedule_snippet

            save_block = '''
            {}  {} late() {{
                sim.treeSeqOutput("data/constBotExp_s_" + {} + "_gen_" + {} + "_init_" + {} + "_nl_" + {} + ".trees") ;
            }}
            '''.format(block_name, max_gens, s, int(time), init, num_loci)

            script_save_blocks += save_block


    script_thresh = '''
    1: late() {{
        mut = sim.mutationsOfType(m2);
        if (size(mut) == 1)
        {{
            if (sim.mutationFrequencies(NULL, mut) > {})
            {{
                cat(simID + ": ESTABLISHED");
                defineConstant("ES", sim.generation + ET);
                sim.rescheduleScriptBlock(s1, sim.generation+BT, sim.generation+BT);
                sim.rescheduleScriptBlock(s2, sim.generation+ET, sim.generation+TT);
                '''.format(init)

    script_thresh += script_reschedule_saves + '''
                sim.treeSeqOutput("data/constBotExp_s_" + {} + "_gen_0_init_" + {} + "_nl_" + {} + ".trees") ;
                sim.deregisterScriptBlock(self);
            }}

        }}     
        else
        {{
            cat(simID + ": LOST – RESTARTING");

            // go back to generation 1000
            sim.readFromPopulationFile("/tmp/slim_" + simID + ".txt");

            // start a newly seeded run
            setSeed(rdunif(1, 0, asInteger(2^62) - 1));

            // re-introduce the sweep mutation
            target = sample(p1.genomes, 1);
            target.addNewDrawnMutation(m2, asInteger(L/2));
        }}

    }}'''.format(s, init, num_loci)

    script_bottle = '''
    s1 {} {{p1.setSubpopulationSize(BS);}}
    '''.format(max_gens)

    script_growth = '''
    s2 {} {{newSize = asInteger(round(exp(GR * (sim.generation - ES)) * BS));
            p1.setSubpopulationSize(newSize);}}
            '''.format(max_gens)

    
    # Initalize output
    extra1 = order % 2
    sfs_out = {int(time) : np.zeros((reps, order//2 + extra1)) for time in saves}

    # Set seed and generate seed sequence
    sql = np.random.SeedSequence(seed)
    seed_seq = sql.generate_state(reps)

    # Run simulations
    for i in range(reps):
        print('Start SLiM Rep '+ str(i))
        script = script_init+ script_begin1 + script_setseed.format(seed=seed_seq[i]) + script_begin2 + script_thresh + script_bottle + script_growth + script_save_blocks
        print(script)
        os.system("echo '" + script + "' | slim")
        print('Start MSPrime')
        for time in saves: 
            if s == 0.:
                to_load = "data/constBotExp_s_{}_gen_{}_init_{}_nl_{}.trees".format(0, int(time), init, num_loci)
                print(to_load)
                ts = pyslim.load(to_load)
            else:
                to_load = "data/constBotExp_s_{}_gen_{}_init_{}_nl_{}.trees".format(s, int(time), init, num_loci)
                print(to_load)
                ts = pyslim.load(to_load)
            
            # Recaptitate to start from stationarity
            rts = ts.recapitate(recombination_rate = r, Ne=n) 
            start_loc = num_loci//2

            # Compute branch lengths
            sfs_full = rts.allele_frequency_spectrum(windows=None, mode='branch', span_normalise=False, polarised=True)   

            # Downsample to sample size
            sfs_out[int(time)][i] = down_samp_sfs(sfs_full, order, len(rts.samples()))

    # Save output      
    out_dict = {'sfs' :  sfs_out, 'times' :  saves}
    pkl.dump(out_dict,  open( out_file, "wb" ) )

def down_samp_sfs(old_sfs, new_order, pop_size):
    old_order = len(old_sfs)-1
    new_sfs = np.zeros(new_order+1)
    for i in range(1, old_order):
        rv = hypergeom(pop_size, i, new_order)
        new_sfs[1:-1] += rv.pmf(np.arange(1, new_order)) * old_sfs[i]
    new_sfs_len = new_order + 1
    extra1 = new_sfs_len % 2
    out = np.zeros(new_sfs_len//2 + extra1)
    for i in range(new_sfs_len//2):
        out[i] = new_sfs[i] + new_sfs[new_order - i]
    if extra1 == 1:
        out[new_sfs_len//2] = new_sfs[new_sfs_len//2]
    return out/2

if __name__ == '__main__':
    main()