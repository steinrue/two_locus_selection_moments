import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
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
@click.option('-renorm/-renorm-no', help='Whether or not to renormalize', default=False)
@click.option('-parsimonious/-parsimonious-no', help='Whether to use the parsimonious projection', default=False)
@click.option('-demog', help='Type demography')
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='mutation rate', type=str)
@click.option('-out_file', help='Filename to output LD to', type=str)
@click.option('-min_step', help='Minimum step size of integrator', type=float)
def main(order, interp_type, renorm, parsimonious, demog, m, s, r, init, out_file, min_step):

    # Convert initial haplotype frequencies to list
    init = json.loads(init)

    # Use correct demography file
    if demog == 'full':
        demog_fn = 'demographies/constBotExpDem.txt'
        save_points = 101
        gens = 10000
    elif demog == 'bottle':
        demog_fn = 'demographies/botExpDem.txt'
        save_points = 41
        gens = 4000
    elif demog == 'growth':
        demog_fn = 'demographies/expDem.txt'
        save_points = 11
        gens = 1000
    demo = dem.Demography(fn=demog_fn)

    # Initialize ODE
    ode_inst = ode.TwoLocMomOde(order, interp_type=interp_type, renorm=renorm, clip_ind=True, parsimonious=parsimonious)
    params = {
        'demog' : demo,
        'lam' : 1, 
        'mut' : [m, m, m, m],
        's' : s,
        'r' : r,
        'init' : init
    }
    
    # Sometimes the ODE will fail to integrate, which case just save 'error'
    try:

        # Set parameters and integrate ODE
        ode_inst.set_parameters(**params)
        [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens, num_points=save_points, keep_traj=True, min_step_size=min_step)
    except:
        out_dict = {}
        out_dict['data'] = 'Error'
        out_dict['params'] = {'s' : s, 'r' :r , 'init' : init, 'min_step' : min_step, 'order' : order,
                            'it': interp_type, 'rn' : renorm, 'par' : parsimonious, 'demog' :  demog}
        pkl.dump(out_dict, open(out_file,'wb'))
        return None

    # Marginalize/downsample moments
    marginalizer = MomentMarginalizer(ode_inst.moms) 
    moms = ode_inst.moms
    moms_one = Moments(1)
    moms_two = Moments(2)
    one_reducer = MomentReducer(moms, moms_one)
    two_reducer = MomentReducer(moms, moms_two)
    hap_freqs = np.zeros((moms_one.nmom, len(times)))
    d2_freqs = np.zeros(len(times))
    ord2_freqs = np.zeros((moms_two.nmom, len(times)))

    # Compute expected haplotype frequencies and LD 
    for i in range(len(times)):
        hap_freqs[:, i] = np.array(one_reducer.computeLower(moment_trajectory[i, :]))
        ord2_freqs[:, i] = np.array(two_reducer.computeLower(moment_trajectory[i, :]))
    ld_moms = [[1, 0, 0, 1], [0, 1, 1, 0]]
    lds = ord2_freqs[moms_two.lookup(tuple(ld_moms[0])), :]/2 - ord2_freqs[moms_two.lookup(tuple(ld_moms[1])), :]/2 
    out_dict = {}
    out_dict['data'] = {'hap': hap_freqs, 'ld' : lds, 'liks' : moment_trajectory, 'times' : times, 'demog' :  demog}
    out_dict['params'] = {'s' : s, 'r' :r , 'init' : init, 'min_step' : min_step, 'order' : order,
                            'it': interp_type, 'rn' : renorm, 'par' : parsimonious, 'demog' :  demog}
    pkl.dump(out_dict,  open( out_file, "wb" ) )
    
if __name__ == '__main__':
    main()
