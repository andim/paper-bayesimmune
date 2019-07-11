import sys
sys.path.append('../lib/')
import numpy as np
from misc import *
import optdynlib

## parameter definitions
theta0s = 0.01
theta0priors = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])*theta0s
Ns = 1e3
Ts = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
nrepeats = 1000

## simulation logic
def run(theta0, theta0prior, N, T, nrepeat):
    N = int(N)
    nrepeat = int(nrepeat)
    g = lambda x: 1.0/x
    A = lambda x: 1.0/(x * theta0prior)
    n_to_p = lambda x: x**.5
    costs = [np.sum(optdynlib.integrate_popdyn_stoch(np.random.dirichlet(theta0*np.ones(N)),
                                                     A, g, T, n_to_p=n_to_p, full_output=False)[1])/T
                                                     for i in range(nrepeat)]
    cost, costse = np.mean(costs), np.std(costs, ddof=1)/nrepeat**.5
    return cost, costse

## batch run parameters
nbatch = 1
disp = True
datadir = 'data/'
outname = 'linwrongtheta'

## batch run logic
paramscomb = params_combination((theta0s, theta0priors, Ns, Ts, nrepeats))
columns = ['theta0', 'theta0prior', 'N', 'T', 'nrepeat', 'cost', 'costse']
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in range(nbatch):
        n = (njob-1) * nbatch + i
        if disp:
            print zip(columns[:len(paramscomb[n])], paramscomb[n])
        res = run(*paramscomb[n])
        row = list(paramscomb[n])
        row.extend(res)
        data.append(row)
    np.savez_compressed(datadir + '%s%g' % (outname, njob), data=data, columns=columns)
