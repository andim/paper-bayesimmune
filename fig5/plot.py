import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use(['../optdynim.mplstyle'])

import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
import pandas as pd
import scipy.special
import scipy.stats

import sys
sys.path.append('../lib')
import optdynlib
import plotting
import misc

df = pd.read_csv('data/infectionhospitalization.csv', skiprows=1)
dfmem = pd.read_csv('data/memory.csv', skiprows=1)
dfmemearly = pd.read_csv('data/memoryearly.csv', skiprows=1)
dfmemearly['Age in years'] = dfmemearly['Age group']/12.0
dfmemearly = dfmemearly[dfmemearly['Age in years'] < 10.0]

lambda_ = 40.

npz = np.load('data/results.npz')
bins = npz['bins']
alphas = npz['alphas']
thetavec = npz['thetavec']
K = len(thetavec)
mecosts = npz['ecosts'][:, :, 0]
mmemfracs = npz['memfracs'][:, :, 0]
mmementropies = npz['mementropies'][:, :, 0]
mmemunumbers = npz['memrichness'][:, :, 0]

fig, axess = plt.subplots(figsize=(6, 4.0), ncols=3, nrows=2, sharex=False)

axes = axess[0, :]

labels = [r'$\alpha=0$', r'$\alpha=0.5$', r'$\alpha=1.0$']
c0s = [np.log(K), K**0.5, K]
for i, alpha in enumerate(alphas):
    costs = [1.0]
    costs.extend(mecosts[:, i]/c0s[i])
    axes[0].plot(bins/lambda_, costs, label=labels[i])
axes[0].legend()
axes[0].set_ylabel('Relative cost $c/c_0$')

for i, alpha in enumerate(alphas):
    memfrac = [0.0]
    memfrac.extend(mmemfracs[:, i])
    axes[1].plot(bins/lambda_, memfrac)
    l, = axes[2].plot(bins[:-1]/lambda_, 2**mmementropies[:, i], label='Entropy' if i == 0 else None)
    axes[2].plot(bins[:-1]/lambda_, mmemunumbers[:, i], '--', color=l.get_color(), label='Richness' if i == 0 else None)
axes[1].set_ylabel('Memory fraction')
axes[2].set_ylabel('Memory diversity')
leg = axes[2].legend(loc='lower right')
for handle in leg.legendHandles:
    handle.set_color('k')
axes[2].set_ylim(0.0)

ax = axes[1]
l, = ax.plot(dfmem['Age group'], (100.0-dfmem['CD4 naive'])/100.0, 'o',
             label='Data', c=colors[4])
ax.plot(dfmemearly['Age in years'], dfmemearly['median']/100.0, 'o',
        c=colors[4], label='')
ax.legend([l], ['Data'])


for ax in axes:
    ax.set_xlim(0.0, 60.0)
    ax.set_xlabel('Age in years')
for ax in axes[:-1]:
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

filenames = ['results-early.npz', 'results-proportional.npz', "results-proportional-memmax.npz"]
npzs = [np.load('data/%s'%filename) for filename in filenames]
bins = [npz['bins'] for npz in npzs]
alphas = [npz['alphas'] for npz in npzs]
thetavec = [npz['thetavec'] for npz in npzs]
K = len(thetavec[0])
mecosts = [npz['ecosts'][:, :, 0] for npz in npzs]
mmemfracs = [npz['memfracs'][:, :, 0] for npz in npzs]
mmementropies = [npz['mementropies'][:, :, 0] for npz in npzs]
mmemunumbers = [npz['memrichness'][:, :, 0] for npz in npzs]

axes = axess[1, :]

labels = ['optimal', 'const. fold\nchange', 'const. fold\nchange + cap']
c0s = [np.log(K), K**0.5, K]
for i in range(len(npzs)):
    costs = [1.0]
    # plot for alpha=.5
    costs.extend(mecosts[i][:, 1]/c0s[1])
    axes[0].plot(bins[i]/lambda_, costs, label=labels[i],
                 color=colors[i+5])
axes[0].legend(ncol=1)
axes[0].set_ylabel('Relative cost $c/c_0$')

for i in range(len(npzs)):
    memfrac = [0.0]
    memfrac.extend(mmemfracs[i][:, 1])
    axes[1].plot(bins[i]/lambda_, memfrac, color=colors[i+5])
#    l, = axes[2].plot(bins[i][:-1]/lambda_, 2**mmementropies[i][:, 1], label='Shannon' if i == 0 else None,
#                     color=colors[i+5])
    #axes[2].plot(bins[i][:-1]/lambda_, mmemunumbers[i][:, 1], '--', color=l.get_color(), label='richness' if i == 0 else None)
axes[1].set_ylabel('Memory fraction')

axes[0].set_ylim(0, 1.2)
axes[0].set_yticks(np.arange(0, 1.21, 0.2))
axes[1].set_ylim(0, 1.01)
axes[1].set_yticks(np.arange(0, 1.01, 0.2))

for ax in axes[:2]:
    ax.set_xlim(0.0, 5.0)
    ax.set_xlabel('Age in years')

ax = axes[2]

filenames = ['results-tauQ10000.npz', 'results-tauQ1000.npz']
npzs = [np.load('data/%s'%filename) for filename in filenames]
bins = [npz['bins'] for npz in npzs]
thetavec = [npz['thetavec'] for npz in npzs]
K = len(thetavec[0])
mecosts = [npz['ecosts'] for npz in npzs]

c0 = K**0.5
tauc = 2*np.array([10000, 1000])/(lambda_*np.sum(thetavec[0]))
lss = ['-', ':']
ls = []
for i in range(len(npzs)):
    for j in range(2):
        costs = [1.0]
        costs.extend(mecosts[i][:, j]/c0)
        l, = ax.plot(bins[i]/lambda_, costs, label='%g'%tauc[i] if j==0 else None,
                ls=lss[j], color=colors[i+8])
        if i == 0:
            ls.append(l)
leg1 = ax.legend(title=r'$\tau_c$', ncol=1, loc='upper left')
leg = ax.legend(title='attrition', handles=ls, labels=['yes', 'no'], loc='lower center', ncol=2)
for handle in leg.legendHandles:
    handle.set_color('k')
ax.add_artist(leg1)
ax.set_ylabel('Relative cost $c/c_0$')

ax.set_xlim(0.0, 60)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Age in years')


plotting.label_axes(fig)
fig.tight_layout()
fig.savefig('fig5.svg')
fig.savefig('fig5.png', dpi=300)
