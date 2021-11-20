import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
# sys.path.append ("/scratch/efriedlander/paperImages/stationaryImages")
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
@click.option('-renorm/-renorm-no', help='Whether or not to clip', default=False)
@click.option('-clip/-clip-no', help='Whether or not to clip', default=False)
@click.option('-parsimonious/-parsimonious-no', help='Whether to use the parsimonious projection', default=False)
@click.option('-demog', help='Demography')
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-out_file', help='Filename to output LD to', type=str)
@click.option('-min_step', help='Minimum step size for ode', type=float)
@click.option('-num_loci', help='How many loci to include', type=int)
@click.option('-window', help='Window size being modeled', type=int)
def main(order, interp_type, renorm, clip, parsimonious, demog, m, s, r, init, out_file, min_step, num_loci, window):

    # Rescale recombination and mutation rates
    m = m * window / 2 / (num_loci-1)
    r = r * window / 2 / (num_loci-1)
    recs = [r*i for i in range(num_loci)]
    
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
        # print('Check -1')
        marginalizer = MomentMarginalizer(Moments(order+1))
        demo_obj_init = dem.Demography(fn='demographies/botExpDem.txt')
        ode_inst_init = ode.TwoLocMomOde(order+1, interp_type=interp_type, renorm=renorm, clip_ind=clip, parsimonious=parsimonious)
        order_reducer = MomentReducer(Moments(order+1), Moments(order))
        # print('Check 0')

        params_burnin = {
            'demog' : demo_obj_init,
            'lam' : 1, 
            'mut' : [m, m, m, m],
            's' : 0,
            'init' : 'Stationary',
            'initial_freq' : 0
        }
        init_mom_dict = {}
            

    demo_obj = dem.Demography(fn=demog_fn)

    ode_inst = ode.TwoLocMomOde(order, interp_type=interp_type, renorm=renorm, clip_ind=clip, parsimonious=parsimonious)
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
    for rec_rate in recs: 
        if demog == 'growth':  
            params_burnin['r'] = rec_rate
            ode_inst_init.set_parameters(**params_burnin)
            init_mom = ode_inst_init.integrate_forward_RK45(3000, first_gen=0, keep_traj=False, min_step_size=min_step)
        params = {
            'demog' : demo_obj,
            'lam' : 1, 
            'mut' : [m, m, m, m],
            's' : s,
            'r' : rec_rate,
            'init' : 'Stationary',
            'initial_freq' : init
        }

        if demog == 'growth':
            params['init'] = 'MarginalMoms'
            params['marg_mom'] = marginalizer.marginalizeB(init_mom)
        ode_inst.set_parameters(**params)
        [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens, first_gen=g0, num_points=save_points, keep_traj=True, min_step_size=min_step)
        if demog == 'growth':
            moment_trajectory[0] = order_reducer.computeLower(init_mom)
        hap_freqs = [np.array(one_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]
        ord2_freqs = [np.array(two_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]
        ord4_freqs = [np.array(four_reducer.computeLower(final_mom)) for final_mom in moment_trajectory]


        heterozygosity.append([ord2_freq[8]+ord2_freq[6]+ord2_freq[4]+ord2_freq[1] for ord2_freq in ord2_freqs])
        d2.append([sum([coef * ord4_freq[moms_four.lookup(mom_idx)] for coef, mom_idx in zip(d2_coef, d2_moms)]) 
                    for ord4_freq in ord4_freqs])
        if rec_rate == 0:
            focal_het.append([ord2_freq[7] + ord2_freq[6] + ord2_freq[4] + ord2_freq[3] for ord2_freq in ord2_freqs])

    out_dict = {}
    out_dict['data'] = {'het': heterozygosity, 'd2' : d2, 'focal_het' : focal_het, 'times' : times}
    recs_scaled = [i*4*demo_obj.n0 for i in recs]
    out_dict['params'] = {'s' : s, 'r' :r , 'recs' : recs_scaled, 'init' : init, 'min_step' : min_step, 'order' : order,
                            'it': interp_type, 'rn' : renorm, 'clip' : clip, 'par' : parsimonious, 'demog' : demog}
    pkl.dump(out_dict,  open( out_file, "wb" ) )
    
if __name__ == '__main__':
    main()
