import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
# sys.path.append ("/Users/efriedlander/Dropbox/Research/twoLocusDiffusion")
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import click
import pandas as pd


@click.command()
@click.option('-oin', '--ode_fn', help='Ode Filename', type=str)
@click.option('-sin', '--simupop_fn', help='Simupop simulation filename', type=str)
@click.option('-ein2', '--exact_traj_fn2', help='Exact trajectory filename', type=str)
@click.option('-ein10', '--exact_traj_fn10', help='Exact trajectory filename', type=str)
@click.option('-aout', '--a_freq_fn', help='Filename to output A Frequencies to', type=str)
@click.option('-bout', '--b_freq_fn', help='Filename to output B Frequencies to', type=str)
@click.option('-about', '--ab_freq_fn', help='Filename to output AB Frequencies to', type=str)
@click.option('-ldout', '--ld_freq_fn', help='Filename to output LD to', type=str)
@click.option('-eout2', '--a_exact_fn2', help='Filename to output Exact Trajectorys to (2000 Ne)', type=str)
@click.option('-eout10', '--a_exact_fn10', help='Filename to output Exact Trajectorys to (10000 Ne)', type=str)
def main(ode_fn, simupop_fn, exact_traj_fn2, exact_traj_fn10, a_freq_fn, b_freq_fn, ab_freq_fn, ld_freq_fn, a_exact_fn2, a_exact_fn10):

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
    
    # Create empty figures
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)

    # Set colors
    colors = ['blue', 'red', 'green']

    # Iterate through demographic scenarios
    ode_fn  = ode_fn.replace('demographies', '{demographies}')
    simupop_fn  = simupop_fn.replace('demographies', '{demographies}')
    demogs = ['full', 'bottle', 'growth']
    for col, demog in zip(colors, demogs):

        # Load data
        ode_in = ode_fn.format(demographies=demog)
        simupop_in = simupop_fn.format(demographies=demog)
        ode = pkl.load( open(ode_in, 'rb'))['data']
        simupop = pkl.load( open(simupop_in, 'rb'))
        simupop_data = simupop['data']
        simupop_param = simupop['params']
        # If ode didn't work then just save an empty image
        if ode == 'Error':
            fig1 = plt.figure()
            fig1.savefig(a_freq_fn, format='pdf')
            fig1.savefig(b_freq_fn, format='pdf')
            fig1.savefig(ab_freq_fn, format='pdf')
            fig1.savefig(ld_freq_fn, format='pdf')
            fig1.savefig(a_exact_fn2, format='pdf')
            fig1.savefig(a_exact_fn10, format='pdf')
            return None
    
        # Extract moment haplotype frequencies and times
        hap_freqs = ode['hap']
        times = ode['times']

        # Adjust starting points
        if demog == 'full':
            start_gen = 0
            n0 = 10000
        elif demog == 'bottle':
            start_gen = 6000
            n0 = 2000
        elif demog == 'growth':
            start_gen = 9000
            n0 = 2000
        

        # Convert simulation generations to full population size generations
        tot_gens = 10000
        sim_gens = simupop_param['gen']
        x_axis = [tot_gens-(i*10 + start_gen) for i in range(sim_gens+1)]
        
        # Compute trajectories and convert moment times to generations
        ode_x_axis = tot_gens - (times * 2 * n0 + start_gen)
        a_freqs_ode = hap_freqs[2, :] + hap_freqs[3, :]
        b_freqs_ode = hap_freqs[3, :] + hap_freqs[1, :]
        ab_freqs_ode = hap_freqs[3, :]
        ld_freqs_ode = ode['ld']

        # Compute mean and se for simulated trajectories
        a_freqs_simupop = simupop_data['a']
        reps = a_freqs_simupop.shape[0]
        a_freqs_mean = np.mean(a_freqs_simupop, axis=0)
        a_freqs_se = np.std(a_freqs_simupop, axis=0)/np.sqrt(reps)

        b_freqs_simupop = simupop_data['b']
        b_freqs_mean = np.mean(b_freqs_simupop, axis=0)
        b_freqs_se = np.std(b_freqs_simupop, axis=0)/np.sqrt(reps)

        ab_freqs_simupop = simupop_data['ab']
        ab_freqs_mean = np.mean(ab_freqs_simupop, axis=0)
        ab_freqs_se = np.std(ab_freqs_simupop, axis=0)/np.sqrt(reps)
        
        ld_freqs_simupop = simupop_data['ld']
        ld_freqs_mean = np.mean(ld_freqs_simupop, axis=0)
        ld_freqs_se = np.std(ld_freqs_simupop, axis=0)/np.sqrt(reps)

        # Add a-trajectories to plot
        ax1.plot(x_axis, a_freqs_mean, label='Simulations',  color=col, linestyle='dashed')
        ax1.fill_between(x_axis, a_freqs_mean - 1.96*a_freqs_se, a_freqs_mean + 1.96*a_freqs_se, color=col, alpha=.2)
        ax1.plot(ode_x_axis, a_freqs_ode, label='ODE', color=col)
        ax1.set_ylabel('Exp. Selected Allele Frequency')
        ax1.set_xlabel('Generations Before Present')
        ax1.set_xlim(10000, 0)
        
        
        # Add b-trajectories to plot      
        ax2.plot(x_axis, b_freqs_mean, label='Simulations',  color=col, linestyle='dashed')
        ax2.fill_between(x_axis, b_freqs_mean - 1.96*b_freqs_se, b_freqs_mean + 1.96*b_freqs_se, alpha=.2, color=col)
        ax2.plot(ode_x_axis, b_freqs_ode, label='ODE', color=col)
        ax2.set_ylabel('Exp. Neutral Allele Frequency')
        ax2.set_xlabel('Generations Before Present')
        ax2.set_xlim(10000, 0)

        # Add AB-trajectories to plot
        ax3.plot(x_axis, ab_freqs_mean, label='Simulations',  color=col, linestyle='dashed')
        ax3.fill_between(x_axis, ab_freqs_mean - 1.96*ab_freqs_se, ab_freqs_mean + 1.96*ab_freqs_se, alpha=.2, color=col)
        ax3.plot(ode_x_axis, ab_freqs_ode, label='ODE', color=col)
        ax3.set_ylabel('Exp. Haplotype Frequency')
        ax3.set_xlabel('Generations Before Present')
        ax3.set_xlim(10000, 0)

        # Add LD-trajectories to plot
        ax4.plot(x_axis, ld_freqs_mean, label='Simulations',  color=col, linestyle='dashed')
        ax4.fill_between(x_axis, ld_freqs_mean - 1.96*ld_freqs_se, ld_freqs_mean + 1.96*ld_freqs_se, alpha=.2, color=col)
        ax4.plot(ode_x_axis, ld_freqs_ode, label='ODE', color=col)
        ax4.set_ylabel('Exp. Linkage Disequilibrium')
        ax4.set_xlabel('Generations Before Present')
        ax4.set_xlim(10000, 0)

    # Load data
    demog = 'constant2'
    ode_in = ode_fn.format(demographies=demog)
    simupop_in = simupop_fn.format(demographies=demog)
    ode = pkl.load( open(ode_in, 'rb'))['data']
    simupop = pkl.load( open(simupop_in, 'rb'))
    simupop_data = simupop['data']
    simupop_param = simupop['params']
    exact_data = pd.read_csv (exact_traj_fn2, delimiter="\t", header=None, comment="#")
    tot_gens = 3000
    exact_traj_times = np.arange (0, tot_gens, 10)
    exact_traj = exact_data[3]

    # Extract moment haplotype frequencies and times
    hap_freqs = ode['hap']
    times = ode['times']

    # Adjust starting points
    start_gen = 0
    n0=2000
    
    # Convert simulation generations to full population size generations
    
    sim_gens = simupop_param['gen']
    x_axis = [tot_gens-(i*10 + start_gen) for i in range(sim_gens+1)]
    
    # Compute trajectories and convert moment times to generations
    ode_x_axis = tot_gens - (times * 2 * n0 + start_gen)
    a_freqs_ode = hap_freqs[2, :] + hap_freqs[3, :]

    # Compute mean and se for simulated trajectories
    a_freqs_simupop = simupop_data['a']
    reps = a_freqs_simupop.shape[0]
    a_freqs_mean = np.mean(a_freqs_simupop, axis=0)
    a_freqs_se = np.std(a_freqs_simupop, axis=0)/np.sqrt(reps)

    # Add a-trajectories to plot
    ax5.plot(x_axis, a_freqs_mean, label='SimuPOP',  color='blue', linestyle='dashed')
    ax5.fill_between(x_axis, a_freqs_mean - 1.96*a_freqs_se, a_freqs_mean + 1.96*a_freqs_se, color='blue', alpha=.2)
    ax5.plot(ode_x_axis, a_freqs_ode, label='ODE', color='red')
    ax5.plot(np.flip(exact_traj_times), exact_traj, label='Exact WF', color='green', linestyle='dashdot')
    ax5.set_ylabel('Exp. Selected Allele Frequency')
    ax5.set_xlabel('Generations Before Present')
    ax5.set_xlim(tot_gens, 0)

    # Load data
    demog = 'constant10'
    ode_in = ode_fn.format(demographies=demog)
    simupop_in = simupop_fn.format(demographies=demog)
    ode = pkl.load( open(ode_in, 'rb'))['data']
    simupop = pkl.load( open(simupop_in, 'rb'))
    simupop_data = simupop['data']
    simupop_param = simupop['params']
    exact_data = pd.read_csv (exact_traj_fn10, delimiter="\t", header=None, comment="#")
    tot_gens = 6000
    exact_traj_times = np.arange (0, tot_gens, 10)
    exact_traj = exact_data[3]

    # Extract moment haplotype frequencies and times
    hap_freqs = ode['hap']
    times = ode['times']

    # Adjust starting points
    start_gen = 0
    n0 = 10000
    
    # Convert simulation generations to full population size generations
    
    sim_gens = simupop_param['gen']
    x_axis = [tot_gens-(i*10 + start_gen) for i in range(sim_gens+1)]
    
    # Compute trajectories and convert moment times to generations
    ode_x_axis = tot_gens - (times * 2 * n0 + start_gen)
    a_freqs_ode = hap_freqs[2, :] + hap_freqs[3, :]

    # Compute mean and se for simulated trajectories
    a_freqs_simupop = simupop_data['a']
    reps = a_freqs_simupop.shape[0]
    a_freqs_mean = np.mean(a_freqs_simupop, axis=0)
    a_freqs_se = np.std(a_freqs_simupop, axis=0)/np.sqrt(reps)

    # Add a-trajectories to plot
    ax6.plot(x_axis, a_freqs_mean, label='SimuPOP',  color='blue', linestyle='dashed')
    ax6.fill_between(x_axis, a_freqs_mean - 1.96*a_freqs_se, a_freqs_mean + 1.96*a_freqs_se, color='blue', alpha=.2)
    ax6.plot(ode_x_axis, a_freqs_ode, label='ODE', color='red')
    ax6.plot(np.flip(exact_traj_times), exact_traj, label='Exact WF', color='green', linestyle='dashdot')
    ax6.set_ylabel('Exp. Selected Allele Frequency')
    ax6.set_xlabel('Generations Before Present')
    ax6.set_xlim(tot_gens, 0)
        
    # Set Legend
    custom_lines = [Line2D([0], [0], color='black'),
                    Line2D([0], [0], color='black', linestyle='dashed')]
    ax1.legend(custom_lines, ['ODE', 'SimuPOP'])
    ax2.legend(custom_lines, ['ODE', 'SimuPOP'])
    ax3.legend(custom_lines, ['ODE', 'SimuPOP'])
    ax4.legend(custom_lines, ['ODE', 'SimuPOP'])
    # custom_lines.append(Line2D([0], [0], color='black', linestyle='dashdot', marker='x'))
    ax5.legend()
    ax6.legend()
    
    # Save figures
    fig1.savefig(a_freq_fn, format='pdf')
    fig2.savefig(b_freq_fn, format='pdf')
    fig3.savefig(ab_freq_fn, format='pdf')
    fig4.savefig(ld_freq_fn, format='pdf')
    fig5.savefig(a_exact_fn2, format='pdf')
    fig6.savefig(a_exact_fn10, format='pdf')
    
    

    
if __name__ == '__main__':
    main()

