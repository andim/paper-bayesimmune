import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('../lib/')
import misc
import plotting
plt.style.use(['../optdynim.mplstyle'])
import matplotlib
colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

df = misc.loadnpz('data/counter.npz')
df.columns = df.columns.astype(str)
print(df)

def se(data):
    return np.std(data, ddof=1)/len(data)**.5
dfmean = df.groupby(list(df.columns[:-2])).agg([np.mean, se]).reset_index()

dfmean['tauc'] = dfmean['tauQ']/(0.5*(dfmean['theta0']*dfmean['K']))
fig, axes = plt.subplots(figsize=(7.2, 2.3), ncols=3)
eps = 1e-8

coloriter = itertools.cycle(colors)

def counts_theta0(t, tau, theta0):
    """theta0 is total"""
    epsilon = ((1.0+theta0)**2+8.0*tau)**.5
    return 0.5*(1.0+theta0+epsilon*np.tanh((t*epsilon)/(4.0*tau)-np.arctanh((1.0-theta0)/epsilon)))-theta0

@misc.memoized
def mincostlin(K, theta0, nrepeat=10000):
    return np.mean([np.sum(np.random.dirichlet(theta0*np.ones(K))**.5)**2 for i in range(nrepeat)])

import scipy.special
def mean_entropy_dirichlet(N, theta0):
    "return the mean entropy of a distribution drawn from a uniform Dirichlet distribution"
    return scipy.special.polygamma(0, N*theta0+1)-scipy.special.polygamma(0, theta0+1)

dfmean = dfmean[(dfmean['tauQ']>100.0) & (dfmean['theta0']>0.015)]
dfmean['countstheta0'] = counts_theta0(dfmean['t'], dfmean['tauQ'], dfmean['theta0']*dfmean['K'])
dfmean['theta0tot'] = dfmean['theta0']*dfmean['K']
dfmean['relcounts'] = dfmean['countstheta0']/(dfmean['theta0']*dfmean['K'])
dfmean['rcg'] = dfmean.apply(lambda x: (x['lincost']['mean']-mincostlin(int(x['K']),float(x['theta0'])))\
                                        /(x['K']-mincostlin(int(x['K']), float(x['theta0']))), axis=1)
dfmean['rcgse'] = dfmean.apply(lambda x: x['lincost']['se']/(x['K']-mincostlin(int(x['K']), float(x['theta0']))), axis=1)
dfmean['rcglog'] = dfmean.apply(lambda x: (x['logcost']['mean']-mean_entropy_dirichlet(int(x['K']),float(x['theta0'])))\
                                        /(np.log(x['K'])-mean_entropy_dirichlet(int(x['K']), float(x['theta0']))), axis=1)


ax = axes[0]
for theta0, dfg in dfmean[(np.abs(dfmean['tauQ']-1e5)<eps)&(dfmean['K']==2000)].groupby('theta0'):
    ax.plot(dfg['t'], dfg['lincost']['mean']/dfg['K'], label=theta0, color=next(coloriter))
ax.legend(title=r'$\theta$')
ax.set_xscale('log')
ax.set_ylabel('relative cost\n$c/c_0$')
ax.set_xlabel('$\lambda t$\n pathogen encounters')

ax = axes[1]
theta0 = 0.02
K = 2000
for tauQ, dfg in dfmean[(np.abs(dfmean['theta0']-theta0)<eps)&(dfmean['K']==K)].groupby('tauQ'):
    tauc = 2*tauQ / (theta0*K)
    ax.plot(dfg['t'], dfg['lincost']['mean']/dfg['K'], label=int(tauQ), color=next(coloriter))
ax.legend(title=r'$\lambda \tau$', loc='lower left')
ax.set_xscale('log')
ax.set_ylabel('relative cost\n$c/c_0$')
ax.set_xlabel('$\lambda t$\n pathogen encounters')

ax = axes[2]
theta0s = sorted(dfmean.theta0.unique())
theta0_to_color = dict(zip(theta0s, colors[:len(theta0s)]))
Ks = np.asarray(dfmean.K.unique(), dtype=int)
lss = ['o', 's', 'x', '+', '>']
K_to_ls = dict(zip(Ks, lss[:len(Ks)]))
handles = []
for (K, theta0) , dfg in dfmean.groupby(['K', 'theta0']):
    l, = ax.plot(dfg['relcounts'], dfg['rcg'], K_to_ls[K], ms=2, c=theta0_to_color[theta0])
    if theta0 == theta0s[0]:
        handles.append(l)
ax.set_xscale('log')
ax.set_xlabel(r'$\lambda t_e/K \theta$'+'\n remembered encounters per\n eff. number of pathogens')
ax.set_ylabel('relative cost gap\n'+'$(c - c_\infty)/ (c_0 - c_\infty)$')
ax.legend(handles, Ks, title=r'$K$', ncol=1, loc='lower left')
ax.set_xticks(np.logspace(-2, 2, 5, base=10))
ax.set_xlim(5e-3, 2e2)

for ax in axes[:2]:
    ax.set_xticks(np.logspace(0, 4, 5, base=10))
    ax.set_xlim(2.0, 2e4)

for ax in axes:
    ax.set_ylim(0, 1.05)
    plotting.despine(ax)
plotting.label_axes(fig, xy=(-0.25, 1.0))

fig.tight_layout()
fig.savefig('fig3.svg')
fig.savefig('fig3.png', dpi=300)


fig, ax = plt.subplots(figsize=(3.5, 2.75))
theta0s = sorted(dfmean.theta0.unique())
theta0_to_color = dict(zip(theta0s, colors[:len(theta0s)]))
Ks = np.asarray(dfmean.K.unique(), dtype=int)
lss = ['o', 's', 'x', '+', '>']
K_to_ls = dict(zip(Ks, lss[:len(Ks)]))
handles = []

ax.plot(dfmean['relcounts'], dfmean['rcg'], marker='o', c='.5', ms=2, ls='None')
for (K, theta0) , dfg in dfmean.groupby(['K', 'theta0']):
    l, = ax.plot(dfg['relcounts'], dfg['rcglog'], K_to_ls[K], ms=2, c=theta0_to_color[theta0])
    if theta0 == theta0s[0]:
        handles.append(l)

ax.set_xscale('log')
ax.set_xlabel(r'$\lambda t_e/K \theta$'+'\nremembered encounters per\n eff. number of pathogens')
ax.set_xticks(np.logspace(-2, 2, 5, base=10))
ax.set_xlim(5e-3, 2e2)

ax.set_ylabel('relative cost gap\n'+'$(c - c_\infty)/(c_0 - c_\infty)$')
fig.tight_layout()
fig.savefig('figS1.svg')
fig.savefig('figS1.png', dpi=300)

