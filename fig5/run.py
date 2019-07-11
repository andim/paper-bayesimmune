import numpy as np
import pandas as pd

import scipy.special
import scipy.stats

import sys
sys.path.append('../lib')
from optdynlib import *
import misc


rate = 1.0
#tauQ = 1e4
#tauP = 1e4
theta = 0.00025
tend = 4000
K = int(1e5)

alphas = [0.0, 0.5, 1.0]


nbins = 125
bins = np.linspace(0.0, tend, num=nbins+1, endpoint=True)
nrepeat = 3
epsilon = 1e-15 # machine epsilon for float comparisons

ncounters = 1
mecosts = np.zeros((nbins, len(alphas), ncounters))
mmemfracs = np.zeros((nbins, len(alphas), ncounters))
mmementropies = np.zeros((nbins, len(alphas), ncounters))
mmemunumbers = np.zeros((nbins, len(alphas), ncounters))

class ProportionalCounter(Counter):
    def mean(self):
        m = self.n - self.theta
        p = np.ones(len(m))
        mask = m > 0.0
        p[mask] = self.theta[mask]**(-m[mask])
        return p/np.sum(p)

seed = 12345
prng = np.random.RandomState(seed)
for repeat in range(nrepeat):

    ## sim ##
    thetavec = theta*np.ones(K)
    counter = Counter(thetavec)
    counters = [counter]
    counters_desc = ['optimal']
    
    ts = [0.0]
    ts.extend(poisson_times(rate=rate, tmax=tend, seed=prng.randint(0, 10000)))
    dts = np.diff(ts)
    nsteps = len(dts)
    
    Q = prng.dirichlet(thetavec)
    stepQ = lambda Q, dt, prng: stepdiffusionanalytical(Q, dt, theta, prng=prng)
        
    ecosts = np.zeros((nsteps, len(alphas), ncounters))
    memfracs = np.zeros((nsteps, len(alphas), ncounters))
    mementropies = np.zeros((nsteps, len(alphas), ncounters))
    memunumbers = np.zeros((nsteps, len(alphas), ncounters))
    for i in range(nsteps):
        for k, c in enumerate(counters):
            Qest = c.mean()
            for j, alpha in enumerate(alphas):
                if alpha != 0.0:
                    p = n_to_p(Qest, alpha=alpha)
                    ecosts[i, j, k] = powercost(Q, P=p, alpha=alpha)
                else:
                    p = Qest
                    ecosts[i, j, k] = logcost(Q, p)
                pmem = p[c.n > c.theta]
                memfracs[i, j, k] = np.sum(pmem)
                pmemnorm = pmem / np.sum(pmem)
                mementropies[i, j, k] = -np.sum(pmemnorm*np.log2(pmemnorm))
                memunumbers[i, j, k] = len(pmem)
        
        dt = dts[i]
#        for c in counters:
#            c.predict(dt/tauP)
#        #counter.predict(dt/tauQ)
#        Q = stepQ(Q, dt/tauQ, prng)
        ind = prng.choice(K, p=Q)
        for c in counters:
            c.update(ind)
        
    ## sim ##
    for j in range(len(alphas)):
        for k in range(ncounters):
            add_binned_stat(mecosts[:, j, k], ts[1:], ecosts[:, j, k], bins)
            add_binned_stat(mmemfracs[:, j, k], ts[1:], memfracs[:, j, k], bins)
            add_binned_stat(mmementropies[:, j, k], ts[1:], mementropies[:, j, k], bins)
            add_binned_stat(mmemunumbers[:, j, k], ts[1:], memunumbers[:, j, k], bins)
    
mecosts /= nrepeat
mmemfracs /= nrepeat
mmementropies /= nrepeat
mmemunumbers /= nrepeat

np.savez_compressed('data/results.npz', ecosts=mecosts,
        bins=bins, thetavec=thetavec,
        alphas=alphas, counters_desc=counters_desc,
        memfracs=mmemfracs, mementropies=mmementropies,
        memrichness=mmemunumbers)
