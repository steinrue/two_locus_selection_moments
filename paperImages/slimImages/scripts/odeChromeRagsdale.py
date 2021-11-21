import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion")
import momlink.mutationHackedOde as ode
import momlink.demography as dem
from momlink.helper_funcs import Moments, MomentReducer, MomentMarginalizer, MomentInterpolator
import pickle as pkl
import json
import scipy
import click


@click.command()
@click.option('-ord', '--order', help='Moment Order', type=int)
@click.option('-it', '--interp_type', help='What type of interpolation to use', type=str)
@click.option('-renorm/-renorm-no', help='Whether or not to clip', default=False)
@click.option('-clip/-clip-no', help='Whether or not to clip', default=False)
@click.option('-parsimonious/-parsimonious-no', help='Whether to use the parsimonious projection', default=False)
@click.option('-demog_fn', help='Filename for demography')
@click.option('-m', help='mutation rate', type=float)
@click.option('-s', help='selection coefficient', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-gens', help='list of save points', type=str   )
@click.option('-out_file', help='Filename to output LD to', type=str)
@click.option('-min_step', help='Minimum step size for ode', type=float)
@click.option('-num_loci', help='How many loci to include', type=int)
@click.option('-folded/-unfolded', help='Whether SFS should be folded', default=True)
def main(order, interp_type, renorm, clip, parsimonious, demog_fn, m, s, r, init, gens, out_file, min_step, num_loci, folded):
    gens = json.loads(gens)
    higher_orders = [51, 71, 101]
    all_orders = [order] + higher_orders
    max_rec = (num_loci // 2+1 )* r
    rec_grid = list(np.geomspace(r, max_rec, 20))
    ode_inst = ode.TwoLocMomOde(order)
    folded_sfs = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    folded_sfs_cond = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    sfs = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    sfs_cond = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    demog = dem.Demography(fn=demog_fn)
    ode_inst = ode.TwoLocMomOde(order, interp_type=interp_type, renorm=renorm, clip_ind=clip, parsimonious=parsimonious)
    moment_dict = {samp_size : Moments(samp_size) for samp_size in all_orders}
    interpolators = {samp_size :  MomentInterpolator(moment_dict[order], moment_dict[samp_size]) for samp_size in higher_orders}
    marginalizers = {samp_size : MomentMarginalizer(moment_dict[samp_size]) for samp_size in all_orders}
    for i, r_point in enumerate(rec_grid):
        params = {
            'demog' : demog,
            'lam' : 1,
            'mutModel' : 'prf',
            'mut' : [0, 0, m, m],
            's' : s,
            'r' : r_point,
            'init' : 'StationaryPRF',
            'initial_freq' : init
        }
        if folded:
            params['mutModel'] = 'rec'
            params['init'] = 'Stationary'
        ode_inst.set_parameters(**params)
        [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens[-1], time_steps=gens, keep_traj=True, min_step_size=min_step)

        margs = {order : [marginalizers[order].marginalizeB(moment_freq) for moment_freq in moment_trajectory]}
        margs.update({samp_size :  [marginalizers[samp_size].marginalizeB(interpolators[samp_size].interp(moment_freq)) for moment_freq in moment_trajectory] for samp_size in higher_orders})

        margs_cond = {order : [marginalizers[order].marginalizeB(marginalizers[order].condition_on_focal_present(moment_freq)) for moment_freq in moment_trajectory]}
        margs_cond.update({samp_size : [marginalizers[samp_size].marginalizeB(marginalizers[samp_size].condition_on_focal_present(interpolators[samp_size].interp(moment_freq))) for moment_freq in moment_trajectory] for samp_size in higher_orders})


        for j, time in enumerate(gens):
            for samp_size in all_orders:
                sfs_len = samp_size+1
                extra1 = sfs_len%2
                folded_sfs_len = sfs_len//2 + extra1
                
                sfs[samp_size][time].append(margs[samp_size][j])
                sfs_cond[samp_size][time].append(margs_cond[samp_size][j])
                foldedmarg = np.zeros(folded_sfs_len)
                foldedmarg[:folded_sfs_len] = margs[samp_size][j][:folded_sfs_len]+np.flip(margs[samp_size][j][folded_sfs_len:])

                foldedmarg_cond = np.zeros(folded_sfs_len)
                foldedmarg_cond[:folded_sfs_len] = margs_cond[samp_size][j][:folded_sfs_len]+np.flip(margs_cond[samp_size][j][folded_sfs_len:])
                if extra1 == 1:
                    foldedmarg_cond[sfs_len//2] = margs_cond[j][sfs_len//2]
                    foldedmarg[sfs_len//2] = margs[j][sfs_len//2]
                folded_sfs_cond[samp_size][time].append(foldedmarg_cond)
                folded_sfs[samp_size][time].append(foldedmarg)


    params = {
            'demog' : demog,
            'lam' : 1,
            'mutModel' : 'prf',
            'mut' : [0, 0, m, m],
            's' : s,
            'r' : 0,
            'init' : 'StationaryPRF',
            'initial_freq' : init
        }
    if folded:
            params['mutModel'] = 'rec'
            params['init'] = 'Stationary'
    exp_sfs_out = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    exp_sfs_cond_out = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    exp_folded_sfs_out = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    exp_folded_sfs_cond_out = {samp_size : {time : [] for time in gens} for samp_size in all_orders}
    for i, time in enumerate(gens):
        for samp_size in all_orders:
            sfs_len = samp_size+1
            extra1 = sfs_len%2
            folded_sfs_len = sfs_len//2 + extra1
            folded_sfs_time = np.array(folded_sfs[samp_size][time])
            interp_folded_sfs = scipy.interpolate.interp1d(rec_grid, folded_sfs_time.transpose())

            sfs_time = np.array(sfs[samp_size][time])
            interp_sfs = scipy.interpolate.interp1d(rec_grid, sfs_time.transpose())

            folded_sfs_cond_time = np.array(folded_sfs_cond[samp_size][time])
            interp_folded_sfs_cond = scipy.interpolate.interp1d(rec_grid, folded_sfs_cond_time.transpose())
            sfs_cond_time = np.array(sfs_cond[samp_size][time])
            interp_sfs_cond = scipy.interpolate.interp1d(rec_grid, sfs_cond_time.transpose())
            
            exp_sfs = np.zeros(sfs_len)
            exp_sfs_cond = np.zeros(sfs_len)
            exp_folded_sfs = np.zeros(folded_sfs_len)
            exp_folded_sfs_cond = np.zeros(folded_sfs_len)
            last_cl = 0
            for j in range(1, num_loci//2+1):
                exp_sfs += interp_sfs(j*r)*2
                exp_sfs_cond += interp_sfs_cond(j*r)*2
                exp_folded_sfs += interp_folded_sfs(j*r)*2
                exp_folded_sfs_cond += interp_folded_sfs_cond(j*r)*2
            exp_sfs_out[samp_size][time] = exp_sfs
            exp_sfs_cond_out[samp_size][time] = exp_sfs_cond
            exp_folded_sfs_out[samp_size][time] = exp_folded_sfs
            exp_folded_sfs_cond_out[samp_size][time] = exp_folded_sfs_cond


    ode_inst.set_parameters(**params)
    [times, moment_trajectory] = ode_inst.integrate_forward_RK45(gens[-1], time_steps=gens, keep_traj=True, min_step_size=min_step)

    margs_focal = {order : [marginalizers[order].marginalizeA(moment_freq) for moment_freq in moment_trajectory]}
    margs_focal.update({samp_size :  [marginalizers[samp_size].marginalizeA(interpolators[samp_size].interp(moment_freq)) for moment_freq in moment_trajectory] for samp_size in higher_orders})

    margs_cond_focal = {order : [marginalizers[order].marginalizeA(marginalizers[order].condition_on_focal_present(moment_freq)) for moment_freq in moment_trajectory]}
    margs_cond_focal.update({samp_size : [marginalizers[samp_size].marginalizeA(marginalizers[samp_size].condition_on_focal_present(interpolators[samp_size].interp(moment_freq))) for moment_freq in moment_trajectory] for samp_size in higher_orders})

    for j, time in enumerate(gens):
        for samp_size in all_orders:
            sfs_len = samp_size+1
            extra1 = sfs_len%2
            folded_sfs_len = sfs_len//2 + extra1

            sfs[samp_size][time].append(margs_focal[samp_size][j])
            sfs_cond[samp_size][time].append(margs_cond_focal[samp_size][j])
            
            foldedmarg = np.zeros( folded_sfs_len)
            foldedmarg[: folded_sfs_len] = margs_focal[samp_size][j][: folded_sfs_len]+np.flip(margs_focal[samp_size][j][ folded_sfs_len:])

            foldedmarg_cond = np.zeros( folded_sfs_len)
            foldedmarg_cond[: folded_sfs_len] = margs_cond_focal[samp_size][j][: folded_sfs_len]+np.flip(margs_cond_focal[samp_size][j][ folded_sfs_len:])
            if extra1 == 1:
                foldedmarg_cond[sfs_len//2] = margs_cond_focal[samp_size][j][sfs_len//2]
                foldedmarg[sfs_len//2] = margs_focal[samp_size][j][sfs_len//2]
            exp_folded_sfs_out[samp_size][time] += foldedmarg
            exp_folded_sfs_cond_out[samp_size][time] += foldedmarg_cond


    out_dict = {'sfs' : exp_sfs_out, 'conditioned_sfs' : exp_sfs_cond_out, 'folded_sfs' : exp_folded_sfs_out, 'conditioned_folded_sfs' : exp_folded_sfs_cond_out, 'times' : gens, 'm' : m}
    pkl.dump(out_dict,  open( out_file, "wb" ) )
    
if __name__ == '__main__':
    main()
   
