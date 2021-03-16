import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
import click
import json


@click.command()
@click.option('-hzgridout', '--hzgrid_fn', help='Filename to hz plot', type=str)
@click.option('-d2gridout', '--d2grid_fn', help='Filename to hz plot', type=str)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-window', help='window size', type=int)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-s_vals', help='list of save points', type=str   )
def main(hzgrid_fn, d2grid_fn, m, r, window, init, s_vals):

    # parse selection coefficients
    s_vals = json.loads(s_vals)

    # Set plotting attributes
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    lw = 3
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE, linewidth=lw)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('xtick.major', width=lw)
    plt.rc('xtick.minor', width=0.5*lw)
    plt.rc('ytick.major', width=lw)
    plt.rc('ytick.minor', width=0.5*lw)
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('lines', linewidth=lw*1.25)
    plt.rc('figure', autolayout=True)
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    # Define important file names
    sim_temp = 'data/simuPopStationary_dem_{{demographies}}_m_{m}_s_{{s}}_r_{r}_window_{window}_i_{init}.p'.format(m=m, r=r, window=window, init=init)
    ode_temp = 'ode_output/odeOutput_dem_{{demographies}}_m_{m}_s_{{s}}_r_{r}_window_{window}_i_{init}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.p'.format(m=m, r=r, window=window, init=init)
    hz_temp = 'images/hzPlot_dem_{{demographies}}_m_{m}_s_{{s}}_r_{r}_window_{window}_i_{init}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.pdf'.format(m=m, r=r, window=window, init=init)
    d2_temp = 'images/d2Plot_dem_{{demographies}}_m_{m}_s_{{s}}_r_{r}_window_{window}_i_{init}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.pdf'.format(m=m, r=r, window=window, init=init)

    ndems = 3
    ns = len(s_vals)
    demographies = ['full', 'bottle', 'growth']
    demo_symbol = ['$\eta_1$', '$\eta_2$', '$\eta_3$']

    fig2, axes2 = plt.subplots(ns, ndems, figsize=(6.4*3, 4.8*3), sharex='col', sharey='col')
    fig3, axes3 = plt.subplots(ns, ndems, figsize=(6.4*3, 4.8*3), sharex='col', sharey='col')
    lgds2 = []
    lgds3 = []

    for i, s in enumerate(s_vals):
        for j, demo in enumerate(demographies):
            ode_in = ode_temp.format(demographies=demo, s=s)

            ode = pkl.load( open(ode_in, 'rb'))
            ode_data = ode['data']
            ode_params = ode['params']

            simupop_in = sim_temp.format(demographies=demo, s=s)
            simupop = pkl.load( open(simupop_in, 'rb'))
            simupop_data = simupop['data']
            simupop_param = simupop['params']
            demog = ode_params['demog']

            if demo == 'full':
                recs = ode_params['recs']
                recs_list = [-i for i in recs[-1:0:-1]] + recs

            a_freq_sim = simupop_data['a']
            c1=np.array(mpl.colors.to_rgb('red'))
            c2=np.array(mpl.colors.to_rgb('blue'))
            colors = [mpl.colors.to_hex((1- j/5)*c1 + j/5*c2) for j in range(6)]
            colors_5 = [mpl.colors.to_hex((1- j/4)*c1 + j/4*c2) for j in range(5)]
            colors_3 = [mpl.colors.to_hex((1- j/2)*c1 + j/2*c2) for j in range(3)]

            def add_hz_to_plot(ax, plot_color, sim_gen, ode_gen, true_gen):
                hz_slice = [time_slice[ode_gen] for time_slice in ode_data['het']]
                hz_ode = hz_slice[-1:0:-1] + hz_slice

                
                a_freq= np.array([[loc_gen[sim_gen] for loc_gen in loc] for loc in a_freq_sim])
                hz_freqs = 2*a_freq*(1-a_freq)
                
                hz_means = np.mean(hz_freqs, axis=1)
                hz_means = np.array(list(hz_means)[-1:1:-1] + list(hz_means[1:]))
                hz_sd = np.std(hz_freqs, axis=1)/np.sqrt(1000)
                hz_sd = np.array(list(hz_sd)[-1:1:-1] + list(hz_sd[1:]))
                ax.plot(recs_list, hz_ode, color=plot_color, label=str(10000-true_gen) + ' generations')
                ax.plot(recs_list, hz_means, color=plot_color, linestyle='dotted')
                ax.fill_between(recs_list, hz_means-1.96*hz_sd, hz_means+1.96*hz_sd, color=plot_color, alpha=.2)

            ax1 = axes2[i, j]
            ax1.ticklabel_format(axis='y', style='sci')
            ax1.yaxis.major.formatter._useMathText = True
            if demog == 'full':
                for l, k in enumerate([0, 3, 6, 9, 10]):
                    add_hz_to_plot(ax1, colors[l], k*100, k, k*1000)
            elif demog == 'bottle':
                for l, k in enumerate([0, 2, 3, 4]):
                    add_hz_to_plot(ax1, colors_5[l], k*100, k, k*1000+6000)
            elif demog == 'growth':
                for l, k in enumerate([0, 1, 2]):
                    add_hz_to_plot(ax1, colors_3[l], k*50, k, k*500+9000)
            if j == 0:
                ax1.set_ylabel('Exp. Heterozygosity')
            ax1.set_title('$\sigma$ = '+ str(s*4*10000) + ' and time of introduction ' + demo_symbol[j])
            

            ld_sim = simupop_data['ld']
            def add_d2_to_plot(ax, plot_color, sim_gen, ode_gen, true_gen):
                d2_slice = [time_slice[ode_gen] for time_slice in ode_data['d2']]
                d2_ode = d2_slice[-1:0:-1] + d2_slice
                # import pdb; pdb.set_trace()
                ld_freqs = np.array([[loc_gen[sim_gen] for loc_gen in loc] for loc in ld_sim])
                d2_freqs = ld_freqs**2
                d2_means = np.mean(d2_freqs, axis=1)
                d2_means = np.array(list(d2_means)[-1:0:-1] + list(d2_means))
                d2_sd = np.std(d2_freqs, axis=1)/np.sqrt(1000)
                d2_sd = np.array(list(d2_sd)[-1:0:-1] + list(d2_sd))
                ax.plot(recs_list, d2_ode, color=plot_color, label=str(10000-true_gen) + ' generations')
                ax.plot(recs_list, d2_means, color=plot_color, linestyle='dotted')
                ax.fill_between(recs_list, d2_means-1.96*d2_sd, d2_means+1.96*d2_sd, color=plot_color, alpha=.2)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            ax1 = axes3[i, j]
            if demog == 'full':
                for l, k in enumerate([0, 3, 6, 9, 10]):
                    add_d2_to_plot(ax1, colors[l], k*100, k, k*1000)
            elif demog == 'bottle':
                for l, k in enumerate([0, 2, 3, 4]):
                    add_d2_to_plot(ax1, colors_5[l], k*100, k, k*1000+6000)
            elif demog == 'growth':
                for l, k in enumerate([0, 1, 2]):
                    add_d2_to_plot(ax1, colors_3[l], k*50, k, k*500+9000)

            # ax1.set_xlabel('Distance from Focal Local')
            if j == 0:
                ax1.set_ylabel('Expected $D_2$')
            ax1.set_title('$\sigma$ = '+ str(s*4*10000) + ' and time of introduction ' + demo_symbol[j])
            
            if i == 2:
                axes2[i, j].set_xlabel('Distance from Selected Locus ($\\rho$)')
                axes3[i, j].set_xlabel('Distance from Selected Locus ($\\rho$)')
                lgds2.append(axes2[i, j].legend(title='Generations before present', loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=False, ncol=1))
                lgds3.append(axes3[i, j].legend(title='Generations before present', loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=False, ncol=1))
    
    
    fig2.savefig(hzgrid_fn, format='pdf', bbox_extra_artists=tuple(lgds2), bbox_inches='tight')
    fig3.savefig(d2grid_fn, format='pdf', bbox_extra_artists=tuple(lgds3), bbox_inches='tight')

if __name__ == '__main__':
    main()

