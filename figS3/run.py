import sys
sys.path.append('../lib/')
import numpy as np
from misc import *
from optdynlib import *

## parameter definitions
thetas = 2.5e-4#0.1
Ks = int(1e5)#2e2
tauQs = [1000.0, 10000.0]
taurels = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
tends = 40.0
nrepeats = 100

alpha = 0.5
lambda_ = 40.0


## simulation logic
def run(theta, K, tauQ, taurel, tend, nrepeat):
    K = int(K)
    nrepeat = int(nrepeat)
    thetavec = theta*np.ones(K)
    tauP = tauQ * taurel
    prng = np.random
    stepQ = lambda Q, dt, prng: stepdiffusionanalytical(Q, dt, theta, prng=prng)
    costss = []
    for i in range(nrepeat):
        Q = prng.dirichlet(thetavec)
        ts = [0.0]
        rate = 1.0
        ts.extend(poisson_times(rate=rate, tmax=tend*lambda_, seed=prng.randint(0, 10000)))
        dts = np.diff(ts)
        nsteps = len(dts)
        counter = Counter(thetavec)
        costs = np.array(nsteps)
        t = 0.0
        for i in range(nsteps):
            dt = dts[i]
            t += dt
            counter.predict(dt/tauP, euler=True)
            Q = stepQ(Q, dt/tauQ, prng)
            ind = prng.choice(K, p=Q)
            counter.update(ind)
        dt = tend*lambda_ - t
        counter.predict(dt/tauP)
        Q = stepQ(Q, dt/tauQ, prng)
        p = n_to_p(counter.mean(), alpha=alpha)
        costss.append(powercost(Q, P=p, alpha=alpha))
    cost, costse = np.mean(costss), np.std(costss, ddof=1)/nrepeat**.5
    return cost, costse

## batch run parameters
nbatch = 1
disp = True
datadir = 'data/'
outname = 'scanlinwrong'

## batch run logic
paramscomb = params_combination((thetas, Ks, tauQs, taurels, tends, nrepeats))
columns = ['theta', 'K', 'tauQ', 'taurel',  'tend', 'nrepeat', 'cost', 'costse']
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in progressbar(range(nbatch)):
        n = (njob-1) * nbatch + i
        if disp:
            print zip(columns[:len(paramscomb[n])], paramscomb[n])
        res = run(*paramscomb[n])
        row = list(paramscomb[n])
        row.extend(res)
        data.append(row)
    np.savez_compressed(datadir + '%s%g' % (outname, njob), data=data, columns=columns)
