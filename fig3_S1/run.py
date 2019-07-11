# general simulation run script

import sys
sys.path.append('../lib/')
import numpy as np
from misc import *
import optdynlib

## parameter definitions
theta0s = [0.2, 0.1, 0.05, 0.02, 0.01]
Ks = [1e2, 2e2, 5e2, 1e3, 2e3]
tauQs = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
reltimes = 1.0
nrepeats = 300

tsaves = [1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4]


def logcost(Q, Qest):
    return -np.sum(Q*np.log(Qest))

def lincost(Q, Qest):
    P = Qest**.5
    P /= np.sum(P)
    return np.sum(Q/P)

def run(theta0=1.0, K=2, tauQ=1.0, reltime=1.0, rate=1.0, seed=None):
    prng = np.random.RandomState(seed) if seed else np.random

    K = int(K)
    theta0vec = theta0*np.ones(K)
    counter = optdynlib.Counter(theta0vec)
    Q = prng.dirichlet(theta0vec)

    ts = [0.0]
    ts.extend(optdynlib.poisson_times(rate=rate, tmax=max(tsaves), seed=prng.randint(0, 10000)))
    ts.extend(tsaves)
    ts = np.asarray(sorted(ts))
    dts = np.diff(ts)

    stepQ = lambda Q, dt, prng: optdynlib.stepdiffusionanalytical(Q, dt, theta0, prng=prng)
    nsteps = len(dts)

    logcosts = []
    lincosts = []
    for i in range(nsteps):
        dt = dts[i]
        counter.predict(dt/(reltime*tauQ))
        Q = stepQ(Q, dt/tauQ, prng)
        if ts[i+1] in tsaves:
            Qest = counter.mean()
            logcosts.append(logcost(Q, Qest))
            lincosts.append(lincost(Q, Qest))
        else:
            counter.update(prng.choice(K, p=Q))
    return logcosts, lincosts

## batch run parameters
nbatch = 1
disp = True
datadir = 'data/'
outname = 'scan'

## batch run logic
paramscomb = params_combination((theta0s, Ks, tauQs, reltimes))
columns = ['theta0', 'K', 'tauQ', 'reltime', 't', 'logcost', 'lincost']
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in progressbar(range(nbatch)):
        n = (njob-1) * nbatch + i
        paramdict = dict(zip(columns[:len(paramscomb[n])], paramscomb[n]))
        if disp:
            print(paramdict) 
        for j in range(nrepeats):
            logcosts, lincosts = run(**paramdict)
            for k, tsave in enumerate(tsaves):
                row = list(paramscomb[n])
                row.append(tsave)
                row.append(logcosts[k])
                row.append(lincosts[k])
                data.append(row)
    np.savez_compressed(datadir + '%s%g' % (outname, njob), data=data, columns=columns)
