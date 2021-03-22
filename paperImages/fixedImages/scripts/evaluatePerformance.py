import pandas as pd
import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
sys.path.append ("/gpfs/data/steinruecken-lab/efriedlander-folder/momentsProject/twoLocusDiffusion")
from momlink.helper_funcs import Moments
import click
import pickle as pkl
import sympy

@click.command()
@click.option('-oin', '--ode_in', help='Ode Filename', type=str)
@click.option('-sin', '--simupop_in', help='Simupop simulation filename', type=str)
@click.option('-out', '--out_fn', help='Output Filename', type=str)
def main(ode_in, simupop_in, out_fn):
    
    # Load ODE output
    ode_all = pkl.load( open(ode_in, 'rb'))
    ode_params = ode_all['params']

    # Output NANs if ODE failed to integrate
    if ode_all['data'] == 'Error':
        out = [ode_params['it'], ode_params['interp_type'], ode_params['rn'], ode_params['par'], 
        ode_params['min_step'], ode_params['s'], ode_params['r'],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]
        out = [[x] for x in out]
        out = pd.DataFrame(out).T
        out.to_csv(out_fn, index=False, header=False, sep=',')
        return None

    # Extract moments from ODE file
    ode = ode_all['data']
    
    # Load simupop simulation outoput and extract trajectories and parameters
    simupop_all = pkl.load( open(simupop_in, 'rb'))
    simupop_data = simupop_all['data']
    simupop_params = simupop_all['params']

    # Extract ode parameters and moments
    ode_times = ode['times']
    ode_liks = ode['liks']
    order = ode_params['order']
    demog = ode_params['demog']

    # Precompute multinomial coefficients
    mc = sympy.ntheory.multinomial.multinomial_coefficients(4, int(order))

    # initialize array of moments
    moments = Moments(order)

    # set position in momdent ode where the different demographic events fall
    if demog == 'full':
        bot_idx = 59
        exp_idx = 89
        full_idx = 99
    elif demog == 'bottle':
        bot_idx = 0
        exp_idx = 29
        full_idx = 39
    elif demog == 'growth':
        bot_idx = 0
        exp_idx = 0
        full_idx = 9

    # Compute differences between sampling probabilities
    hap_freqs_bot = simupop_data['bot']
    lik_dif_bot, sim_lik_bot = get_likelihood_diffs(ode['liks'][bot_idx, :], hap_freqs_bot, moments, mc)

    hap_freqs_exp = simupop_data['exp']
    lik_dif_exp, sim_lik_exp = get_likelihood_diffs(ode['liks'][exp_idx, :], hap_freqs_exp, moments, mc)

    hap_freqs_gen = simupop_data['tot']
    lik_dif_gen, sim_lik_tot = get_likelihood_diffs(ode['liks'][full_idx, :], hap_freqs_gen, moments, mc)

    # generate output
    out = [ode_params['it'], ode_params['init'], demog, ode_params['rn'], ode_params['par'], ode_params['min_step'],
            ode_params['s'], ode_params['r'],
            np.abs(lik_dif_bot).mean(),
            np.abs(lik_dif_exp).mean(),
            np.abs(lik_dif_gen).mean(),
            ((np.abs(lik_dif_bot))/sim_lik_bot).mean(),
            (np.abs(lik_dif_exp)/sim_lik_exp).mean(),
            (np.abs(lik_dif_gen)/sim_lik_tot).mean(),
            np.abs(lik_dif_bot**2).mean(),
            np.abs(lik_dif_exp**2).mean(),
            np.abs(lik_dif_gen**2).mean()
    ]

    out = [[x] for x in out]
	
    out = pd.DataFrame(out).T
    out.to_csv(out_fn, index=False, header=False, sep=',')

    
    
def get_likelihood_diffs(ode_lik, sims, moms, mc):
    out = []
    sim_liks = []
    for lik, mom in zip(ode_lik, moms.moms):
        sim_lik = np.mean(mc[mom] * sims[:, 0]**mom[0]* sims[:, 1]**mom[1]* sims[:, 2]**mom[2]* sims[:, 3]**mom[3])
        out.append(lik - sim_lik)
        sim_liks.append(sim_lik)
    return np.array(out), np.array(sim_liks)

if __name__ == '__main__':
    main()

