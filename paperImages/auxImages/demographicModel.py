import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt 

import numpy as np 

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




# General Figure 2
x = np.linspace(0, 10000, 10001)
y = np.array([10000 for i in range(6000)] + [2000 for i in range(3000)] + [2000*(1.0025)**i for i in range(1001)])
plt.xlim(10000,0)
plt.plot(10000-x, y, linewidth=3)
plt.annotate('$\eta_1$', xy=(10000, 10000), xytext=(9000, 5000), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('$\eta_2$', xy=(4000, 10000), xytext=(5000, 14000), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('$\eta_3$', xy=(1000, 2000), xytext=(2500, 5000), arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel('Generations Before Present')
plt.ylim(0, 25000)
plt.ylabel('Population Size')
plt.savefig('demographicModel.pdf')