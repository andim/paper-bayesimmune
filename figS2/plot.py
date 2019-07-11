import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use(['../optdynim.mplstyle'])

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
import pandas as pd
import scipy.special

import palettable

import sys
sys.path.append('../lib')
import optdynlib
import plotting
import misc


df = misc.loadnpz('data/data.npz')
N = float(np.unique(df['N']))
theta0 = float(np.unique(df['theta0']))
theta0priors = list(np.unique(df['theta0prior']))
Ts = list(np.unique(df['T']))
df = df[df['T']<1000000]
df['relcost'] = df.apply(lambda row: row['cost']/float(df[(np.abs(df['theta0prior'] - theta0)<1e-3) & (df['T'] == row['T'])]['cost']), axis=1)

fig, axes = plt.subplots(figsize=(5.5, 2.3), ncols=2)

#colors_theta0prior = np.asarray(eval('palettable.colorbrewer.diverging.BrBG_%i'%len(theta0priors)).mpl_colors)
#colors_theta0prior[len(colors_theta0prior)//2] = .5
#colors_T = np.asarray(eval('palettable.colorbrewer.sequential.OrRd_%i'%len(theta0priors)).mpl_colors)
#colors_T = colors
colors_theta0prior = colors
colors_T = colors[3:]

ax = axes[0]
thetaplot = [0.001, 0.002, 0.01, 0.1]
dfgprior = df[df['theta0prior'].isin(thetaplot)].groupby(['theta0prior'])
for theta0prior, dfgg in dfgprior:
    ax.plot(dfgg['T']/N, dfgg['cost']/(dfgg['N']), 'o-', label=theta0prior,
            c=colors_theta0prior[theta0priors.index(theta0prior)])
ax.set_xscale('log')
ax.legend(title=r'$\tilde \theta$', ncol=1)
ax.set_xlabel(r'$\lambda t/K$')
ax.set_ylabel(r'$c_{\tilde \theta}(t)/c_0$')

ax = axes[1]
dfgprior = df[df['theta0prior'].isin(thetaplot)].groupby(['theta0prior'])
for theta0prior, dfgg in dfgprior:
    ax.plot(dfgg['T']/N, dfgg['relcost'], 'o-', label=theta0prior,
           c=colors_theta0prior[theta0priors.index(theta0prior)])
ax.set_xscale('log')
ax.set_ylabel(r'$c_{\tilde \theta}(t)/c_{\theta}(t)$')
ax.set_xlabel(r'$\lambda t/K$')
#ax.legend(title=r'$\tilde \theta$', ncol=2, loc='lower center')
ax.set_ylim(0.95, 1.35)

if False:
    ax = axes[2]
    dfg = df[df['T']==1*N].groupby(['T'])
    for T, dfgg in dfg:
        refcost = float(dfgg[np.abs(dfgg['theta0prior'] - theta0)<1e-3]['cost'])
        ax.errorbar(dfgg['theta0prior'], dfgg['cost']/refcost, dfgg['costse']/refcost, label='%g'%(T/N),
                   c=colors_T[Ts.index(T)])
    ax.set_xscale('log')
    ax.set_xlabel(r'prior, $\tilde \theta$')
    ax.set_ylabel(r'$c_{\tilde \theta}(K/\lambda)/c_{\theta}(K/\lambda)$')
    #ax.legend(title=r'$\lambda t/K$', loc='upper center', ncol=2)
    ax.set_ylim(0.95, 1.35)


for ax in axes:
    plotting.despine(ax)
plotting.label_axes(axes, xy=(-0.22, 0.95))

fig.tight_layout()
fig.savefig('figS2.svg')
fig.savefig('figS2.png', dpi=300)
