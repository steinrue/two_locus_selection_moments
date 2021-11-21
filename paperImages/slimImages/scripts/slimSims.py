import os
import numpy as np
import pickle as pkl
import json
import sys
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion/paperImages/slimImages")
from scipy.stats import hypergeom
import allel
import click
from hashlib import blake2b

@click.command()
@click.option('-n', help='Population Size', type=int)
@click.option('-truen', help='Population Size of simulation', type=int)
@click.option('-nl', '--num_loci', help='Number of Loci to Simulate', type=int)
@click.option('-reps', help='Number of repetitions', type=int)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='mutation rate', type=float)
@click.option('-bot_gen', help='mutation rate', type=int)
@click.option('-exp_gen', help='mutation rate', type=int)
@click.option('-bot_size', help='mutation rate', type=int)
@click.option('-growth_rate', help='mutation rate', type=float)
@click.option('-gens', help='mutation rate', type=int)
@click.option('-seed', help='mutation rate', type=int)
@click.option('-saves', help='generations at which to save', type=str)
@click.option('-order', help='Sample size of interest', type=int)
@click.option('-out_file', help='Filename to output to', type=str)
def main(n, truen, num_loci, reps, s, m, r,  init, bot_gen, exp_gen, bot_size, growth_rate, gens, seed, saves, order, out_file):

    seed = int.from_bytes(blake2b((out_file + str(seed)).encode(), digest_size=4).digest(), "big")
    saves = json.loads(saves)
    scale_ratio = truen/n
    s = s * scale_ratio * 2
    m = m * scale_ratio * 2
    r = r * scale_ratio
    burnin = 10 * n
    end_growth_size = bot_size * np.exp(growth_rate * (gens-exp_gen))
    bot_gen_sim = int(bot_gen / scale_ratio)
    exp_gen_sim = int(exp_gen / scale_ratio)
    bot_size_sim = int(bot_size / scale_ratio)
    gens_sim = int(gens / scale_ratio)
    end_growth_size_sim = int(end_growth_size / scale_ratio)
    growth_rate_sim = np.log(end_growth_size_sim/bot_size_sim)/(gens_sim-exp_gen_sim)
    saves = [i/scale_ratio for i in saves]
    sizes = [order, 51, 71, 101]
    
    max_gens = int(gens_sim*100)
    script_init = """
    initialize() {{
        // initializeTreeSeq();
        defineConstant("N0", {});
        defineConstant("s", {});
        defineConstant("L", {});
        defineConstant("r", {});
        initializeMutationRate({});
        initializeMutationType("m1", 0.5, "f", 0.0);
        initializeMutationType("m2", 0.5, "f", s); // introduced mutation
        initializeGenomicElementType("g1", m1, 1.0);
        initializeGenomicElement(g1, 0, L);
        initializeRecombinationRate(r);
    }}""".format(n, s, num_loci-1, r, m)

    script_begin1 = """
    1 late() {{ 
        defineConstant("BT", {});
        defineConstant("BS", {});
        defineConstant("ET", {});
        defineConstant("GR", {});
        defineConstant("TT", {});
        defineConstant("Burnin", {});
        """.format(bot_gen_sim, bot_size_sim, exp_gen_sim, growth_rate_sim, gens_sim, burnin)
    
    script_setseed = """
        setSeed({seed});
        """
    
    script_begin2 = """
        defineConstant("simID", getSeed());
        sim.addSubpop("p1", N0);
        }}

        {} late() {{

            sim.addSubpopSplit("p2", 2, p1);

            // sample a bunch
            // times two because of diploids
            targets = sample (p1.genomes, asInteger(2 * {} * N0));

            // first one gets the selected mutation
            targets[0].addNewDrawnMutation (m2, asInteger(L/2));

            // and then copy the first one a couple of times
            for (i in 1:(length(targets)-1))
            {{
                // remove all mutations from target
                targets[i].removeMutations();
                // add the ones from first target
                // thus copy the background (including beneficial mutation)
                targets[i].addMutations(targets[0].mutations);
	        }}

            defineConstant("ES", Burnin + ET);
            p1.genomes.outputVCF("data/constBotExp_s_" + {} + "_gen_0_init_" + {} + "_nl_" + {} + ".txt");
            }}
            """.format(burnin, init, s, init, num_loci)

    script_save_blocks = ''
    for i, time in enumerate(saves):
        if time != 0:
            block_name = 's' + str(i+3)
            save_block = '''
            {}  {} late() {{
                p1.genomes.outputVCF("data/constBotExp_s_" + {} + "_gen_" + {} + "_init_" + {} + "_nl_" + {} + ".txt") ;
            }}
            '''.format(block_name, burnin+int(time), s, int(time), init, num_loci)

            script_save_blocks += save_block

    
    script_bottle = '''
    s1 {} {{p1.setSubpopulationSize(BS);
            print(BS);}}
    '''.format(burnin + bot_gen_sim)

    script_growth = '''
    s2 {}: {{
            newSize = asInteger(round(exp(GR * (sim.generation - ES)) * BS));
            p1.setSubpopulationSize(newSize);}}
            '''.format(burnin + exp_gen_sim)

    

    extra1 = order % 2
    sfs_out =  {sample_size : {int(time) : np.zeros((reps, sample_size+1))  for time in saves} for sample_size in sizes}
    sfs_out_folded =  {sample_size : {int(time) : np.zeros((reps, sample_size//2 + extra1))  for time in saves} for sample_size in sizes}

    # Set seed and generate seed sequence
    sql = np.random.SeedSequence(seed)
    seed_seq = sql.generate_state(reps)
    for i in range(reps):
        print('Start SLiM Rep '+ str(i))
        script = script_init+ script_begin1 + script_setseed.format(seed=seed_seq[i]) + script_begin2  + script_bottle + script_growth + script_save_blocks
        print(script)
        os.system("echo '" + script + "' | slim")
        
        for time in saves: 
            if s == 0.:
                to_load = "data/constBotExp_s_{}_gen_{}_init_{}_nl_{}.txt".format(0, int(time), init, num_loci)
                pop = allel.read_vcf(to_load, fields=['calldata/GT'])
            else:
                to_load = "data/constBotExp_s_{}_gen_{}_init_{}_nl_{}.txt".format(s, int(time), init, num_loci)
                pop = allel.read_vcf(to_load, fields=['calldata/GT'])

            start_loc = num_loci//2
            
            for samp_size in sizes:
                if time < 600:
                    current_size = n
                elif time <= 900:
                    current_size = bot_size_sim
                else:
                    current_size = end_growth_size_sim
                sfs_full = allel.sfs(allel.GenotypeArray(pop['calldata/GT']).count_alleles()[:,1], n=2*current_size) 
                sfs_full[0] = num_loci - sum(sfs_full[1:])
                sfs_out_folded[samp_size][int(time)][i], sfs_out[samp_size][int(time)][i] = down_samp_sfs(sfs_full, samp_size, 2*current_size)
            
    out_dict = {'sfs' :  sfs_out, 'sfs_folded' : sfs_out_folded, 'times' :  saves}
    pkl.dump(out_dict,  open( out_file, "wb" ) )

def down_samp_sfs(old_sfs, new_order, pop_size):
    old_order = pop_size
    new_sfs = np.zeros(new_order+1)
    for i in range(0, len(old_sfs)):
        rv = hypergeom(old_order, i, new_order)
        new_sfs += rv.pmf(np.arange(0, new_order+1)) * old_sfs[i]
    new_sfs_len = new_order + 1
    extra1 = new_sfs_len % 2
    out = np.zeros(new_sfs_len//2 + extra1)
    for i in range(new_sfs_len//2):
        out[i] = new_sfs[i] + new_sfs[new_order - i]
    if extra1 == 1:
        out[new_sfs_len//2] = new_sfs[new_sfs_len//2]
    return out/2, new_sfs/2

if __name__ == '__main__':
    main()