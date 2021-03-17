import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import click
import json
import matplotlib as mpl

@click.command()
@click.option('-gridout', '--grid_fn', help='Filename to hz plot', type=str)
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-s_vals', help='list of save points', type=str   )
@click.option('-window_vals', help='list of save points', type=str   )
def main(grid_fn, m, r, init, s_vals, window_vals):
    s_vals = json.loads(s_vals)
    window_vals = json.loads(window_vals)


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

    slim_temp = 'data/slim_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}.p'.format(m=m, r=r, init=init)
    ode_temp = 'ode_output/odeOutput_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.p'.format(m=m, r=r, init=init)
    sfs_temp = 'images/sfs_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.pdf'.format(m=m, r=r, init=init)

    c1=np.array(mpl.colors.to_rgb('red'))
    c2=np.array(mpl.colors.to_rgb('blue'))
    colors_5 = [mpl.colors.to_hex((1- j/4)*c1 + j/4*c2) for j in range(5)]

    # for s in s_vals:
    #     for num_loci in window_vals:
    #         ode_in = ode_temp.format(s=s, num_loci=num_loci)
    #         ode = pkl.load( open(ode_in, 'rb'))
    #         ode_data = ode['sfs']
    #         ode_params = ode['times']
    #         # m = ode_params['m']
    #         m = 1.25 * 10**-8

    #         slim_in = slim_temp.format(s=s, num_loci=num_loci)
    #         slim = pkl.load( open(slim_in, 'rb'))
    #         slim_sfs = slim['sfs']
    #         slim_sfs_means = {time : np.mean(slim_sfs[time], axis=0)*2*m*10 for time in slim_sfs.keys()}
    #         fig1 = plt.figure()
    #         ax1 = fig1.add_subplot(111)
    #         x_vals = [i+1 for i in range(len(ode_data[0])-1)]
    #         ax1.plot(x_vals, ode_data[0][1:], color=colors_5[0], label='Gen. 0')
    #         ax1.plot(x_vals, ode_data[3000][1:], color=colors_5[1], label='Gen. 3k')
    #         ax1.plot(x_vals, ode_data[6000][1:], color=colors_5[2], label='Gen. 6k')
    #         ax1.plot(x_vals, ode_data[9000][1:], color=colors_5[3], label='Gen. 9k')
    #         ax1.plot(x_vals, ode_data[10000][1:], color=colors_5[4], label='Gen. 10k')
    #         ax1.plot(x_vals, slim_sfs_means[0][1:], color=colors_5[0], linestyle='dotted')
    #         ax1.plot(x_vals, slim_sfs_means[300][1:], color=colors_5[1], linestyle='dotted')
    #         ax1.plot(x_vals, slim_sfs_means[600][1:], color=colors_5[2], linestyle='dotted')
    #         ax1.plot(x_vals, slim_sfs_means[900][1:], color=colors_5[3], linestyle='dotted')
    #         ax1.plot(x_vals, slim_sfs_means[1000][1:], color=colors_5[4], linestyle='dotted')
    #         ax1.set_xlabel('Minor Allele Frequency')
    #         ax1.set_ylabel('Frequency')
    #         ax1.legend(loc='upper right')

    #         sfs_fn = sfs_temp.format(s=s, num_loci=num_loci)
    #         fig1.savefig(sfs_fn, format='pdf')

    fig2, axes = plt.subplots(len(s_vals),len(window_vals), figsize=(6.4*3, 4.8*3), sharex='col', sharey='col')
    lgds = []
    
    for i, s in enumerate(s_vals):
        for j, num_loci in enumerate(window_vals):
            ode_in = ode_temp.format(s=s, num_loci=num_loci)
            ode = pkl.load( open(ode_in, 'rb'))
            ode_data = ode['sfs']
            ode_params = ode['times']
            # m = ode_params['m']
            m = 1.25 * 10**-8

            slim_in = slim_temp.format(s=s, num_loci=num_loci)
            slim = pkl.load( open(slim_in, 'rb'))
            slim_sfs = slim['sfs']
            slim_sfs_means = {time : np.mean(slim_sfs[time], axis=0)*2*m*10 for time in slim_sfs.keys()}
            slim_sfs_error = {time : np.std(slim_sfs[time]*2*m*10, axis=0)*1.96/np.sqrt(len(slim_sfs[time][:, 0])) for time in slim_sfs.keys()}
            x_vals = [i+1 for i in range(len(ode_data[0])-1)]
            axes[i, j].plot(x_vals, ode_data[0][1:], color=colors_5[0], label='10000 generations')
            axes[i, j].plot(x_vals, ode_data[3000][1:], color=colors_5[1], label='7000 generations')
            axes[i, j].plot(x_vals, ode_data[6000][1:], color=colors_5[2], label='4000 generations')
            axes[i, j].plot(x_vals, ode_data[9000][1:], color=colors_5[3], label='1000 generations')
            axes[i, j].plot(x_vals, ode_data[10000][1:], color=colors_5[4], label='0 generations')
            # axes[i, j].plot(x_vals, slim_sfs_means[0][1:], color=colors_5[0], linestyle='dotted')
            # axes[i, j].plot(x_vals, slim_sfs_means[300][1:], color=colors_5[1], linestyle='dotted')
            # axes[i, j].plot(x_vals, slim_sfs_means[600][1:], color=colors_5[2], linestyle='dotted')
            # axes[i, j].plot(x_vals, slim_sfs_means[900][1:], color=colors_5[3], linestyle='dotted')
            # axes[i, j].plot(x_vals, slim_sfs_means[1000][1:], color=colors_5[4], linestyle='dotted')
            axes[i, j].errorbar(x_vals, slim_sfs_means[0][1:], yerr=slim_sfs_error[0][1:], color=colors_5[0], linestyle='dotted')
            axes[i, j].errorbar(x_vals, slim_sfs_means[300][1:], yerr=slim_sfs_error[300][1:], color=colors_5[1], linestyle='dotted')
            axes[i, j].errorbar(x_vals, slim_sfs_means[600][1:], yerr=slim_sfs_error[600][1:], color=colors_5[2], linestyle='dotted')
            axes[i, j].errorbar(x_vals, slim_sfs_means[900][1:], yerr=slim_sfs_error[900][1:], color=colors_5[3], linestyle='dotted')
            axes[i, j].errorbar(x_vals, slim_sfs_means[1000][1:], yerr=slim_sfs_error[1000][1:], color=colors_5[4], linestyle='dotted')
            

            max_value_ode = np.log10(max(i for v in ode_data.values() for i in v[1:]))
            max_value_slim = np.log10(max(i for v in slim_sfs_means.values() for i in v[1:]))
            max_value = np.ceil(max(max_value_ode, max_value_slim))

            min_value_ode = np.log10(min(i for v in ode_data.values() for i in v[1:]))
            min_value_slim = np.log10(min(i for v in slim_sfs_means.values() for i in v[1:]))
            min_value = np.floor(min(min_value_ode, min_value_slim))

            for oom in range(int(min_value), int(max_value)+1):
                axes[i, j].plot ([-1,31], [np.power(10, oom),np.power(10, oom)], "--", color="black", lw=0.7)
                axes[i, j].plot ([-1,31], [np.power(10, oom+.5), np.power(10, oom+.5)], "--", color="black", lw=0.7)


            
            
            if j == 0:
                axes[i, j].set_ylabel('Expected Frequency')
            axes[i, j].set_yscale('log')
            axes[i, j]. set_ylim([np.power(10, min_value), np.power(10, max_value)])
            axes[i, j]. set_xlim([0, 16])
            # ymin, ymax = axes[i, j].get_ylim()
            # axes[i, j].set_yticks(np.geomspace(ymin, ymax, 5))
            axes[i, j].yaxis.get_major_locator().numticks = 4
            # axes[i, j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{:.2e}"))
            axes[i, j].set_title('$\sigma$ = '+ str(s*4*10000) + ' and Window Size = ' + str(num_loci//1000)+'kBP')
            if i == 2:
                lgds.append(axes[i, j].legend(title='Generations before present', loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=False, ncol=1))
                axes[i, j].set_xlabel('Minor Allele Frequency')
                axes[i, j].xaxis.set_ticks(np.arange(1, 16, 2))

    fig2.savefig(grid_fn, format='pdf', bbox_extra_artists=tuple(lgds), bbox_inches='tight')


if __name__ == '__main__':
    main()

