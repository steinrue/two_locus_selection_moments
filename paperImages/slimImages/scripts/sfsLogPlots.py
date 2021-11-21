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
from matplotlib.lines import Line2D

@click.command()
@click.option('-m', help='mutation rate', type=float)
@click.option('-r', help='recombination rate', type=float)
@click.option('-init', help='initial frequency of focal allele', type=float)
@click.option('-s_vals', help='list of save points', type=str   )
@click.option('-window_vals', help='list of save points', type=str   )
@click.option('-folded/-unfolded', help='Whether SFS should be folded', default=True)
@click.option('-rags/-reg', help='Which mutation method', default=True)
def main(m, r, init, s_vals, window_vals, folded, rags):
    includefixed = False
    s_vals = json.loads(s_vals)
    window_vals = json.loads(window_vals)
    orders = [31, 51, 71, 101]


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
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('lines', linewidth=lw*1.25)
    plt.rc('figure', autolayout=True)
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    if folded:
        folded_name = 'folded'
    else:
        folded_name = 'unfolded'

    if includefixed:
        includefixed_name = 'includefixed'
    else:
        includefixed_name = 'excludefixed'

    if rags:
        slim_temp = 'data/slim_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}.p'.format(m=m, r=r, init=init)
        ode_temp = 'ode_output/odeOutputRags_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{folded}.p'.format(m=m, r=r, init=init, folded=folded_name)
        sfs_temp = 'images/sfslogrags_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_order_{{order}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.pdf'.format(m=m, r=r, init=init)
        sfs_grid_temp = 'images/sfsloggridrags_m_{m}_r_{r}_i_{init}_order_{{order}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{folded}_{includeFixed}.pdf'.format(m=m, r=r, init=init, folded=folded_name, includeFixed=includefixed_name)
    else:
        slim_temp = 'data/slim_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}.p'.format(m=m, r=r, init=init)
        ode_temp = 'ode_output/odeOutput_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{folded}.p'.format(m=m, r=r, init=init, folded=folded_name)
        sfs_temp = 'images/sfslog_m_{m}_s_{{s}}_r_{r}_i_{init}_win_{{num_loci}}_order_{{order}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no.pdf'.format(m=m, r=r, init=init)
        sfs_grid_temp = 'images/sfsloggrid_m_{m}_r_{r}_i_{init}_order_{{order}}_minstep_0.0001_loglin_renorm_clip_parsimonious-no_{folded}_{includeFixed}.pdf'.format(m=m, r=r, init=init, folded=folded_name, includeFixed=includefixed_name)
    

    c1=np.array(mpl.colors.to_rgb('red'))
    c2=np.array(mpl.colors.to_rgb('blue'))
    colors_5 = [mpl.colors.to_hex((1- j/4)*c1 + j/4*c2) for j in range(5)]

    if includefixed:
        plus1 = 0
    else:
        plus1 = 1

    for s in s_vals:
        for order in orders:
            for num_loci in window_vals:
                ode_in = ode_temp.format(s=s, num_loci=num_loci)
                ode = pkl.load( open(ode_in, 'rb'))
                slim_in = slim_temp.format(s=s, num_loci=num_loci)
                slim = pkl.load( open(slim_in, 'rb'))
                
                if folded:
                    ode_data = ode['folded_sfs'][order]
                    slim_sfs = slim['sfs_folded'][order]
                else:
                    ode_data = ode['sfs'][order]
                    slim_sfs = slim['sfs'][order]
                ode_params = ode['times']

                
                # import pdb; pdb.set_trace()
                
                slim_sfs_means = {time : np.mean(slim_sfs[time], axis=0) for time in slim_sfs.keys()}
                slim_sfs_error = {time : np.std(slim_sfs[time], axis=0)*1.96/np.sqrt(len(slim_sfs[time][:, 0])) for time in slim_sfs.keys()}
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                if folded:
                    end = len(ode_data[0]) 
                else:
                    end = len(ode_data[0]) - plus1
                x_vals = [i for i in range(plus1, end)]
                ax1.plot(x_vals, ode_data[0][plus1:end], color=colors_5[0], label='Gen. 0')
                ax1.plot(x_vals, ode_data[3000][plus1:end], color=colors_5[1], label='Gen. 3k')
                ax1.plot(x_vals, ode_data[6000][plus1:end], color=colors_5[2], label='Gen. 6k')
                ax1.plot(x_vals, ode_data[9000][plus1:end], color=colors_5[3], label='Gen. 9k')
                ax1.plot(x_vals, ode_data[10000][plus1:end], color=colors_5[4], label='Gen. 10k')
                ax1.plot(x_vals, slim_sfs_means[0][plus1:end], color=colors_5[0], linestyle='dotted')
                ax1.plot(x_vals, slim_sfs_means[300][plus1:end], color=colors_5[1], linestyle='dotted')
                ax1.plot(x_vals, slim_sfs_means[600][plus1:end], color=colors_5[2], linestyle='dotted')
                ax1.plot(x_vals, slim_sfs_means[900][plus1:end], color=colors_5[3], linestyle='dotted')
                ax1.plot(x_vals, slim_sfs_means[1000][plus1:end], color=colors_5[4], linestyle='dotted')
                ax1.errorbar(x_vals, slim_sfs_means[0][plus1:end], yerr=slim_sfs_error[0][plus1:end], color=colors_5[0], linestyle='dotted')
                ax1.errorbar(x_vals, slim_sfs_means[300][plus1:end], yerr=slim_sfs_error[300][plus1:end], color=colors_5[1], linestyle='dotted')
                ax1.errorbar(x_vals, slim_sfs_means[600][plus1:end], yerr=slim_sfs_error[600][plus1:end], color=colors_5[2], linestyle='dotted')
                ax1.errorbar(x_vals, slim_sfs_means[900][plus1:end], yerr=slim_sfs_error[900][plus1:end], color=colors_5[3], linestyle='dotted')
                ax1.errorbar(x_vals, slim_sfs_means[1000][plus1:end], yerr=slim_sfs_error[1000][plus1:end], color=colors_5[4], linestyle='dotted')
                # import pdb; pdb.set_trace()
                max_value_ode = np.log10(max(i for v in ode_data.values() for i in v[plus1:end])+1)
                max_value_slim = np.log10(max(i for v in slim_sfs_means.values() for i in v[plus1:end])+1)
                max_value = np.ceil(max(max_value_ode, max_value_slim), dtype=np.float)

                min_value_ode = np.log10(min(i for v in ode_data.values() for i in v[plus1:end])+1)
                min_value_slim = np.log10(min(i for v in slim_sfs_means.values() for i in v[plus1:end])+1)
                min_value = np.floor(min(min_value_ode, min_value_slim), dtype=np.float)

                for oom in range(int(min_value), int(max_value)+1):
                    ax1.plot ([-1,31], [np.power(10., oom),np.power(10., oom)], "--", color="black", lw=0.7)
                    ax1.plot ([-1,31], [np.power(10., oom+.5), np.power(10., oom+.5)], "--", color="black", lw=0.7)
                ax1.set_xlabel('Minor Allele Frequency')
                ax1.set_ylabel('Expected Frequency')
                ax1.legend(loc='upper right', prop={"size":12})
                ax1.set_title('$\sigma$ = '+ str(s*4*10000) + ' and Window Size = ' + str(num_loci//1000)+'kBP')
                ax1.set_yscale('log')
                ax1.set_xscale('log')
                ax1. set_ylim([np.power(10, min_value), np.power(10, max_value)])
                ax1. set_xlim([x_vals[0], x_vals[-1]])
                ax1.yaxis.get_major_locator().numticks = 4
                sfs_fn = sfs_temp.format(s=s, num_loci=num_loci, order=order)
                fig1.savefig(sfs_fn, format='pdf')
    plt.close('all')
    
    
    for order in orders:
        if order > 31:
            last_s = -1
        else:
            last_s = None
        fig2, axes = plt.subplots(len(s_vals[:last_s]),len(window_vals), figsize=(6.4*3, 4.8*3), sharex='col', sharey='col')
        lgds = []
        for i, s in enumerate(s_vals[:last_s]):
            for j, num_loci in enumerate(window_vals):
                ode_in = ode_temp.format(s=s, num_loci=num_loci)
                ode = pkl.load( open(ode_in, 'rb'))
                slim_in = slim_temp.format(s=s, num_loci=num_loci)
                slim = pkl.load( open(slim_in, 'rb'))
                
                if folded:
                    ode_data = ode['folded_sfs'][order]
                    slim_sfs = slim['sfs_folded'][order]
                else:
                    ode_data = ode['sfs'][order]
                    slim_sfs = slim['sfs'][order]
                ode_params = ode['times']
                # m = ode_params['m']
                m = 1.25 * 10**-8

                slim_sfs_means = {time : np.mean(slim_sfs[time], axis=0) for time in slim_sfs.keys()}
                slim_sfs_error = {time : np.std(slim_sfs[time], axis=0)*1.96/np.sqrt(len(slim_sfs[time][:, 0])) for time in slim_sfs.keys()}
                if folded:
                    end = len(ode_data[0]) 
                else:
                    end = len(ode_data[0]) - plus1
                x_vals = [i for i in range(plus1, end)]
                axes[i, j].plot(x_vals, ode_data[0][plus1:end], color=colors_5[0], label='10000 generations')
                axes[i, j].plot(x_vals, ode_data[3000][plus1:end], color=colors_5[1], label='7000 generations')
                axes[i, j].plot(x_vals, ode_data[6000][plus1:end], color=colors_5[2], label='4000 generations')
                axes[i, j].plot(x_vals, ode_data[9000][plus1:end], color=colors_5[3], label='1000 generations')
                axes[i, j].plot(x_vals, ode_data[10000][plus1:end], color=colors_5[4], label='0 generations')
                # axes[i, j].plot(x_vals, slim_sfs_means[0][plus1:end], color=colors_5[0], linestyle='dotted')
                # axes[i, j].plot(x_vals, slim_sfs_means[300][plus1:end], color=colors_5[1], linestyle='dotted')
                # axes[i, j].plot(x_vals, slim_sfs_means[600][plus1:end], color=colors_5[2], linestyle='dotted')
                # axes[i, j].plot(x_vals, slim_sfs_means[900][plus1:end], color=colors_5[3], linestyle='dotted')
                # axes[i, j].plot(x_vals, slim_sfs_means[1000][plus1:end], color=colors_5[4], linestyle='dotted')
                axes[i, j].errorbar(x_vals, slim_sfs_means[0][plus1:end], yerr=slim_sfs_error[0][plus1:end], color=colors_5[0], linestyle='dotted')
                axes[i, j].errorbar(x_vals, slim_sfs_means[300][plus1:end], yerr=slim_sfs_error[300][plus1:end], color=colors_5[1], linestyle='dotted')
                axes[i, j].errorbar(x_vals, slim_sfs_means[600][plus1:end], yerr=slim_sfs_error[600][plus1:end], color=colors_5[2], linestyle='dotted')
                axes[i, j].errorbar(x_vals, slim_sfs_means[900][plus1:end], yerr=slim_sfs_error[900][plus1:end], color=colors_5[3], linestyle='dotted')
                axes[i, j].errorbar(x_vals, slim_sfs_means[1000][plus1:end], yerr=slim_sfs_error[1000][plus1:end], color=colors_5[4], linestyle='dotted')

                # axes[i, j].plot(x_vals, [4*10000*m*num_loci*(1/num_der+1/(30-num_der)) for num_der in x_vals], color='black')

                max_value_ode = np.log10(max(i for v in ode_data.values() for i in v[plus1:end]))
                max_value_slim = np.log10(max(i for v in slim_sfs_means.values() for i in v[plus1:end]))
                max_value = np.ceil(max(max_value_ode, max_value_slim)+.1)

                min_value_ode = np.log10(min(i for v in ode_data.values() for i in v[plus1:end]))
                min_value_slim = np.log10(min(i for v in slim_sfs_means.values() for i in v[plus1:end]))
                min_value = np.floor(min(min_value_ode, min_value_slim)-.1)

                for oom in range(int(min_value)-1, int(max_value)+1):
                    axes[i, j].plot ([-1,103], [np.power(10., oom),np.power(10., oom)], "--", color="black", lw=0.7)
                    axes[i, j].plot ([-1,103], [np.power(10., oom+.5), np.power(10., oom+.5)], "--", color="black", lw=0.7)


                
                
                if j == 0:
                    axes[i, j].set_ylabel('Expected Frequency')
                axes[i, j].set_yscale('log')
                axes[i, j].set_xscale('log')
                axes[i, j]. set_ylim([np.power(10, min_value), np.power(10, max_value)])
                axes[i, j]. set_xlim([x_vals[0], x_vals[-1]])
                # ymin, ymax = axes[i, j].get_ylim()
                # axes[i, j].set_yticks(np.geomspace(ymin, ymax, 5))
                axes[i, j].yaxis.get_major_locator().numticks = 4
                # axes[i, j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{:.2e}"))
                axes[i, j].set_title('$\sigma$ = '+ str(s*4*10000) + ' and Window Size = ' + str(num_loci//1000)+'kBP')
                if i == len(s_vals[:last_s])-1:
                    lgds.append(axes[i, j].legend(title='Generations before present', loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=False, ncol=1))
                    axes[i, j].set_xlabel('Minor Allele Frequency')
                    axes[i, j].xaxis.set_ticks(np.round(np.geomspace(1, max(x_vals), 4)))
                    axes[i, j].xaxis.set_ticklabels(np.round(np.geomspace(1, max(x_vals), 4)).astype(int))
                if i == 0:
                    custom_lines = [Line2D([0], [0], color='black'),
                    Line2D([0], [0], color='black', linestyle='dashed')]
                    axes[i, j].legend(custom_lines, ['ODE', 'SLiM'], loc='upper right')

        fig2.savefig(sfs_grid_temp.format(order=order), format='pdf', bbox_extra_artists=tuple(lgds), bbox_inches='tight')

    plt.close('all')

if __name__ == '__main__':
    main()

