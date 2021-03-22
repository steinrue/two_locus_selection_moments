import numpy as np
import sys
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion")
import momlink.ode as ode
import momlink.demography as dem
from momlink.helper_funcs import Moments, MomentReducer, MomentMarginalizer
import pickle as pkl
import json
import scipy
import click


@click.command()
@click.option('-ord', '--order', help='Moment Order', type=int)
@click.option('-demog_fn', help='Filename for demography')
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-gens', help='list of save points', type=str   )
@click.option('-out_file', help='Filename to output to', type=str)
@click.option('-min_step', help='Minimum step size for ode', type=float)
@click.option('-num_loci', help='How many loci to include', type=int)
def main(order, demog_fn, m, s, r, init, gens, out_file, min_step, num_loci):

    # Initialize parameters and recombination grid
    gens = json.loads(gens)
    max_rec = (num_loci // 2+1 )* r
    rec_grid = list(np.geomspace(r, max_rec, 20))

    # Generate ode instance
    ode_inst = ode.TwoLocMomOde(order)

    # Simulate across recombination rate grid
    folded_sfs = {time : [] for time in gens}
    folded_sfs_cond = {time : [] for time in gens}
    demog = dem.Demography(fn=demog_fn)
    ode_inst = ode.TwoLocMomOde(order, interp_type='loglin', renorm=True, clip_ind=True, parsimonious=False)
    for i, r_point in enumerate(rec_grid):
        params = {
            'demog' : demog,
            'lam' : 1,
            'mut' : [0, 0, m, m],
            's' : s,
            'r' : r_point,
            'init' : 'Stationary',
            'initial_freq' : init
        }
    
        ode_inst.set_parameters(**params)
        [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens[-1], time_steps=gens, keep_traj=True, min_step_size=min_step)

        # Marginalize out selected site
        marginalizer = MomentMarginalizer(ode_inst.moms) 
        margs = [marginalizer.marginalizeB(moment_freq) for moment_freq in moment_trajectory]
        margs_cond = [marginalizer.marginalizeB(marginalizer.condition_on_focal_present(moment_freq)) for moment_freq in moment_trajectory]

        # Compute SFS' for different times
        for j, time in enumerate(gens):
            sfs_len = order+1
            extra1 = sfs_len % 2
            foldedmarg = np.zeros(sfs_len//2+extra1)
            foldedmarg[:sfs_len//2+extra1] = margs[j][:sfs_len//2+extra1]+np.flip(margs[j][sfs_len//2+extra1:])

            foldedmarg_cond = np.zeros(sfs_len//2+extra1)
            foldedmarg_cond[:sfs_len//2+extra1] = margs_cond[j][:sfs_len//2+extra1]+np.flip(margs_cond[j][sfs_len//2+extra1:])
            if extra1 == 1:
                foldedmarg_cond[sfs_len//2] = margs_cond[j][sfs_len//2]
                foldedmarg[sfs_len//2] = margs[j][sfs_len//2]
            folded_sfs_cond[time].append(foldedmarg_cond)
            folded_sfs[time].append(foldedmarg)

    # Interpolate across sfs grid to create expected SFS'
    exp_sfs_out = {}
    exp_sfs_cond_out = {}
    exp_sfs_fixed_dist_out = {}
    exp_sfs_fixed_dist_cond_out = {}
    for time in gens:
        folded_sfs_time = np.array(folded_sfs[time])
        interp_sfs = scipy.interpolate.interp1d(rec_grid, folded_sfs_time.transpose())

        folded_sfs_cond_time = np.array(folded_sfs_cond[time])
        interp_sfs_cond = scipy.interpolate.interp1d(rec_grid, folded_sfs_cond_time.transpose())
        
        exp_sfs = np.zeros(sfs_len//2 + extra1)
        exp_sfs_cond = np.zeros(sfs_len//2 + extra1)
        last_cl = 0
        for j in range(1, num_loci//2+1):
            exp_sfs += interp_sfs(j*r)*2
            exp_sfs_cond += interp_sfs_cond(j*r)*2
        exp_sfs_out[time] = exp_sfs
        exp_sfs_cond_out[time] = exp_sfs_cond

    # Account for focal site (only gets added in once)
    params = {
                'demog' : demog,
                'lam' : 1,
                'mut' : [0, 0, m, m],
                's' : s,
                'r' : 0,
                'init' : 'Stationary',
                'initial_freq' : init
            }
    ode_inst.set_parameters(**params)
    [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens[-1], time_steps=gens, keep_traj=True, min_step_size=min_step)
    marginalizer = MomentMarginalizer(ode_inst.moms) 
    margs_focal = [marginalizer.marginalizeA(moment_freq) for moment_freq in moment_trajectory]
    margs_cond_focal = [marginalizer.marginalizeA(marginalizer.condition_on_focal_present(moment_freq)) for moment_freq in moment_trajectory]

    for j, time in enumerate(gens):
            sfs_len = order+1
            extra1 = sfs_len % 2
            foldedmarg = np.zeros(sfs_len//2+extra1)
            foldedmarg[:sfs_len//2+extra1] = margs_focal[j][:sfs_len//2+extra1]+np.flip(margs_focal[j][sfs_len//2+extra1:])

            foldedmarg_cond = np.zeros(sfs_len//2+extra1)
            foldedmarg_cond[:sfs_len//2+extra1] = margs_cond_focal[j][:sfs_len//2+extra1]+np.flip(margs_cond_focal[j][sfs_len//2+extra1:])
            if extra1 == 1:
                foldedmarg_cond[sfs_len//2] = margs_cond_focal[j][sfs_len//2]
                foldedmarg[sfs_len//2] = margs_focal[j][sfs_len//2]
            exp_sfs_out[time] += foldedmarg
            exp_sfs_cond_out[time] += foldedmarg_cond

    # Output results
    out_dict = {'sfs' : exp_sfs_out, 'conditioned_sfs' : exp_sfs_cond_out, 'times' : gens, 'm' : m}
    pkl.dump(out_dict,  open( out_file, "wb" ) )
    
if __name__ == '__main__':
    main()
   
