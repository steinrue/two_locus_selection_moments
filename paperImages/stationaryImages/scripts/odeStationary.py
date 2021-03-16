import numpy as np
import sys
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion")
import momlink.ode as ode
import momlink.demography as dem
from momlink.helper_funcs import Moments, MomentReducer, MomentMarginalizer
import pickle as pkl
import json
import click


@click.command()
@click.option('-ord', '--order', help='Moment Order', type=int)
@click.option('-it', '--interp_type', help='What type of interpolation to use', type=str)
@click.option('-demog', help='Demography')
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-out_file', help='Filename to output LD to', type=str)
@click.option('-min_step', help='Minimum step size for ode', type=float)
@click.option('-num_loci', help='How many loci to include', type=int)
@click.option('-window', help='Window size being modeled', type=int)
def main(order, demog, m, s, r, init, out_file, min_step, num_loci, window):

    # Rescale recombination and mutation rates
    m = m * window / 2 / (num_loci-1)
    r = r * window / 2 / (num_loci-1)
    
    # Use correct demography file
    if demog == 'full':
        demog_fn = 'demographies/constBotExpDem.txt'
        save_points = 11
        gens = 10000
        g0 = 0
    elif demog == 'bottle':
        demog_fn = 'demographies/botExpDem.txt'
        save_points = 5
        gens = 4000
        g0 = 0
    elif demog == 'growth':
        demog_fn = 'demographies/expDem.txt'
        save_points = 3
        gens = 1000
        g0 = 0

    # Define demography
    demo_obj = dem.Demography(fn=demog_fn)

    # Initialize moment ODE instance, moment reducers, and empty lists
    ode_inst = ode.TwoLocMomOde(order, interp_type='loglin', renorm=True, clip_ind=True, parsimonious=False)
    recs = [r*i for i in range(num_loci)]
    heterozygosity = []
    d2 = []
    focal_het = [] 
    d2_moms = [(0, 2, 2, 0), (1, 1, 1, 1), (2, 0, 0, 2)]
    d2_coef = [1/6,-1/12, 1/6]
    moms = ode_inst.moms
    moms_one = Moments(1)
    moms_two = Moments(2)
    moms_four = Moments(4)
    one_reducer = MomentReducer(moms, moms_one)
    two_reducer = MomentReducer(moms, moms_two)
    four_reducer = MomentReducer(moms, moms_four)

    # Integrate ode for every recombination rate in list
    for rec_rate in recs:   
        params = {
            'demog' : demo_obj,
            'lam' : 1, 
            'mut' : [m, m, m, m],
            's' : s,
            'r' : rec_rate,
            'init' : 'Stationary',
            'initial_freq' : init
        }
        ode_inst.set_parameters(**params)
        [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens, first_gen=g0, num_points=save_points, keep_traj=True, min_step_size=min_step)

        # Compute lower order moments
        hap_freqs = [np.array(one_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]
        ord2_freqs = [np.array(two_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]
        ord4_freqs = [np.array(four_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]

        # Compute heterozygosity and D2
        heterozygosity.append([ord2_freq[8]+ord2_freq[6]+ord2_freq[4]+ord2_freq[1] for ord2_freq in ord2_freqs])
        d2.append([sum([coef * ord4_freq[moms_four.lookup(mom_idx)] for coef, mom_idx in zip(d2_coef, d2_moms)]) 
                    for ord4_freq in ord4_freqs])
        if rec_rate == 0:
            focal_het.append([ord2_freq[7] + ord2_freq[6] + ord2_freq[4] + ord2_freq[3] for ord2_freq in ord2_freqs])

    # Save data
    out_dict = {}
    out_dict['data'] = {'het': heterozygosity, 'd2' : d2, 'focal_het' : focal_het, 'times' : times}
    recs_scaled = [i*4*demo_obj.n0 for i in recs]
    out_dict['params'] = {'s' : s, 'r' :r , 'recs' : recs_scaled, 'init' : init, 'min_step' : min_step, 'order' : order, 'demog' : demog}
    pkl.dump(out_dict,  open( out_file, "wb" ) )
    
if __name__ == '__main__':
    main()
