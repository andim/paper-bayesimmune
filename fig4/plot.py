import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['../optdynim.mplstyle'])

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
black = matplotlib.rcParams['text.color']
import pandas as pd
import scipy.special
import scipy.stats

import sys
sys.path.append('../lib')
import optdynlib
import plotting
import misc


def bayesian(p, kappa, alpha=1.0):
    return (p**(1.0+alpha)+kappa)**(1.0/(1.0+alpha))
    #return p+kappa

def proportional(p, factor):
    return p * factor


def plot_fits(ax, x, y, preinf, foldexpansion=False):
    div = preinf if foldexpansion else 1.0
    ls = []
    for alpha in [0, 0.5, 1.0]:
        popt, pcov = scipy.optimize.curve_fit(lambda p, kappa: bayesian(p, kappa, alpha), x, y, method='lm')
        postinf = bayesian(preinf, popt, alpha)
        l, = ax.plot(preinf, postinf/div, lw=2, label=r'$\alpha=%g$'%alpha)
        ls.append(l)
    #popt, pcov = scipy.optimize.curve_fit(proportional, x, y, method='lm')
    #postinf = proportional(preinf, popt)
    #ax.plot(preinf, postinf/div, '-', label='proportional')
    #ax.plot(preinf, preinf/div, '--', c='.6', label='no boosting')
    return ls

fig = plt.figure(figsize=(3, 3.75))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, width_ratios=[3.5,1], height_ratios=[1.0, 0.65])
ax3 = plt.subplot(gs[0, :])
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])


theta0 = np.logspace(-5.5, 1.5)

ax = ax1
#ax.plot(CV**2, CV**2, label='log')
ax.plot(theta0, (1.0+1.0/theta0), label='log')
ax.plot(theta0, (1.0+1.0/theta0)**(1/1.5), label='log')
ax.plot(theta0, (1.0+1.0/theta0)**.5, label='lin')
#ax.fill_between(CV**2, 30, 1000, facecolor=colors[4])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_yticks(10.0**np.arange(0, 7, 2))
ax.set_xticks(10.0**np.arange(-5, 2, 2))
ax.set_xlim(min(theta0), max(theta0))
ax.set_ylim(0.5, 1/min(theta0))
#ax.legend()
#ax.grid()
ax.set_ylabel('Fold change')
ax.set_xlabel(r'Sparsity $\theta$')

theta0 = 0.01
ax = ax2
ninfections = np.arange(0, 5)
thetas = ninfections/theta0+1.0
dplin = (thetas[1:]**.5-thetas[:-1]**.5)
ax.bar(ninfections[1:], dplin/dplin[0], color=colors[2])
ax.set_ylabel('Abs. change')
ax.set_xlabel('Infection')
ax.set_xticks(ninfections[1:])
ax.set_ylim(0.0, 1.05)


ax = ax3

df = pd.read_csv('data/ellebedy2014titersH5N1.csv')
#ax.plot(df['Head (Pre-)'], df['Head (Post-)'], 'o', label='Head')
#ax.plot(df['Stem (Pre-)'], df['Stem (Post-)'], 'o', label='Stem')

x = np.concatenate([df['Head (Pre-)'], df['Stem (Pre-)']])
y = np.concatenate([df['Head (Post-)'], df['Stem (Post-)']])
#preinf = np.logspace(5, 14, base=2)
#plot_fits(ax, x, y, preinf)

l, = ax.plot(df['Head (Pre-)'], df['Head (Post-)']/df['Head (Pre-)'], 'o', color=colors[4], label='Experiment')#, label='Head epitope')
ax.plot(df['Stem (Pre-)'], df['Stem (Post-)']/df['Stem (Pre-)'], 'o', color=l.get_color())#, label='Stem epitope')

preinf = np.logspace(5, 14, base=2)
ls = plot_fits(ax, x, y, preinf, foldexpansion=True)


ax.set_xlabel('Prevaccination titer')
ax.set_ylabel('Fold change')
#ax.set_xscale('log', basex=2)
#ax.set_yscale('log', basey=2)
ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)
ax.legend(loc='upper right')#, handles=[ls[0], l], labels=['Model', 'Experiment'])
#ax.set_xlim(2**5.1, 2**14.5)
#ax.set_ylim(2**5.1, 2**14.5)
ax.autoscale(tight=True)
ax.set_ylim(0.5, 2e2)

fig.tight_layout(h_pad=0.1)
plotting.label_axes(fig)
fig.savefig('fig4.svg')
fig.savefig('fig4.png', dpi=300)
