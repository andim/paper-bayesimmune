import numpy as np
import pandas as pd

import scipy.special
import scipy.stats

import sys
sys.path.append('../lib')
from optdynlib import *
import misc


rate = 1.0
tauQ = 1e4
tauP = tauQ
theta = 0.00025
tend = 4000
K = int(1e5)

alpha = 0.5


nbins = 125
bins = np.linspace(0.0, tend, num=nbins+1, endpoint=True)
nrepeat = 10
epsilon = 1e-15 # machine epsilon for float comparisons

ncounters = 2
mecosts = np.zeros((nbins, ncounters))
mmemfracs = np.zeros((nbins, ncounters))
mmementropies = np.zeros((nbins, ncounters))
mmemunumbers = np.zeros((nbins, ncounters))

seed = 12345
prng = np.random.RandomState(seed)
for repeat in range(nrepeat):

    ## sim ##
    thetavec = theta*np.ones(K)
    counter = Counter(thetavec)
    counter_nodiscount = Counter(thetavec)
    counters = [counter, counter_nodiscount]
    counters_desc = ['optimal', 'nodiscount']
    
    ts = [0.0]
    ts.extend(poisson_times(rate=rate, tmax=tend, seed=prng.randint(0, 10000)))
    dts = np.diff(ts)
    nsteps = len(dts)
    
    Q = prng.dirichlet(thetavec)
    stepQ = lambda Q, dt, prng: stepdiffusionanalytical(Q, dt, theta, prng=prng)
        
    ecosts = np.zeros((nsteps, ncounters))
    memfracs = np.zeros((nsteps, ncounters))
    mementropies = np.zeros((nsteps, ncounters))
    memunumbers = np.zeros((nsteps, ncounters))
    for i in range(nsteps):
        for k, c in enumerate(counters):
            Qest = c.mean()
            if alpha != 0.0:
                p = n_to_p(Qest, alpha=alpha)
                ecosts[i, k] = powercost(Q, P=p, alpha=alpha)
            else:
                p = Qest
                ecosts[i, k] = logcost(Q, p)
            pmem = p[c.n > c.theta]
            memfracs[i, k] = np.sum(pmem)
            pmemnorm = pmem / np.sum(pmem)
            mementropies[i, k] = -np.sum(pmemnorm*np.log2(pmemnorm))
            memunumbers[i, k] = len(pmem)
        
        dt = dts[i]
        counter.predict(dt/tauP)
        Q = stepQ(Q, dt/tauQ, prng)
        ind = prng.choice(K, p=Q)
        for c in counters:
            c.update(ind)
        
    ## sim ##
    for k in range(ncounters):
        add_binned_stat(mecosts[:, k], ts[1:], ecosts[:, k], bins)
        add_binned_stat(mmemfracs[:, k], ts[1:], memfracs[:, k], bins)
        add_binned_stat(mmementropies[:, k], ts[1:], mementropies[:, k], bins)
        add_binned_stat(mmemunumbers[:, k], ts[1:], memunumbers[:, k], bins)
    
mecosts /= nrepeat
mmemfracs /= nrepeat
mmementropies /= nrepeat
mmemunumbers /= nrepeat

np.savez_compressed('data/results-tauQ%g.npz'% tauQ, ecosts=mecosts,
        bins=bins, thetavec=thetavec,
        counters_desc=counters_desc,
        memfracs=mmemfracs, mementropies=mmementropies,
        memrichness=mmemunumbers)
