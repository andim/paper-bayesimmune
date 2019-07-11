import numpy as np
import pandas as pd
import scipy.special

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use(['../optdynim.mplstyle'])

import palettable

import sys
sys.path.append('../lib')
import optdynlib
import plotting
import misc

df = misc.loadnpz('data/data.npz')
df['tauc'] = 2*df['tauQ']/(df['K']*df['theta'])
lambda_ = 40.0

fig, axes = plt.subplots(figsize=(5, 2.25), nrows=1, ncols=2, sharex=True)
for i, (tauc, dfg) in enumerate(df.groupby('tauc')):
    c0 = dfg['K']**.5
    axes[i].errorbar(dfg['taurel'], dfg['cost']/c0, 2*dfg['costse']/c0)
    axes[i].text(0.5, 1.0, r'$\tau_c = %g$ years'%(tauc/lambda_), ha='center', va='top', transform=axes[i].transAxes)
axes[0].set_xscale('log')
for ax in axes:
    ax.set_ylabel('$c(t)/c_0$')
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(2e-2, 5e1)
    plotting.despine(ax)
for ax in axes:
    ax.set_xlabel(r'relative time scale $\tilde \tau/\tau$')
plotting.label_axes(fig)
fig.tight_layout()
fig.savefig('figS3.svg')
fig.savefig('figS3.png', dpi=300)
