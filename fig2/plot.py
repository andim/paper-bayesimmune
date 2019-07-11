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

def runcounts(theta0=1.0, K=2, tauQ=1.0, tend=1.0,
              dtmax=1.0, rate=1.0, seed=None, stepcounts=None):
    
    if stepcounts is None:
        def stepcounts(c, co, lambda1, dt):
            #return c*np.exp(-lambda1*dt)
            denomin = np.exp(dt*(2*lambda1-1)/2)*(2*lambda1-1+c+co) - (c+co)
            return c*(2*lambda1-1)/denomin, co*(2*lambda1-1)/denomin

    K = int(K)
    prng = np.random.RandomState(seed) if seed else np.random
    alpha = theta0
    beta = (K-1)*theta0
    alpha = theta0
    lambda1 = optdynlib.lambdan(alpha, beta, 1)[1]
    Q = prng.beta(alpha, beta)
    counts = 0 # number of counts
    countsother = 0 # number of counts
    ts = [0.0]
    ts.extend(optdynlib.poisson_times(rate=rate, tmax=tend, seed=prng.randint(0, 10000)))
    dts = np.diff(ts)
    stepQ = lambda Q, dt, prng: optdynlib.step1ddiffusionanalytical(Q,
                                                                    dt/(tauQ*0.5*(alpha+beta)),
                                                                    alpha, beta,
                                                                    dtmax=dtmax, prng=prng)
    nsteps = len(dts)
    countss = [counts]
    countssother = [countsother]
    Qs = [Q]
    for i in range(nsteps):
        dt = dts[i]
        Q = stepQ(Q, dt, prng)
        counts, countsother = stepcounts(counts, countsother, lambda1, dt/tauQ)
#        counts *= np.exp(-lambda1*dt/tauQ)
#        countsother *= np.exp(-lambda1*dt/tauQ)
        if prng.rand() < Q:
            counts += 1
        else:
            countsother += 1
        countss.append(counts)
        countssother.append(countsother)
        Qs.append(Q)
        ts = np.asarray(ts)
    Qs = np.asarray(Qs)
    qest = (theta0+np.array(countss))/(K*theta0 + np.array(countss) + np.array(countssother))
    return ts, Qs, qest

def run(theta0=1.0, K=2, tauQ=1.0, tend=1.0, nmax=100, dtmax=1.0, rate=1.0, seed=None):
    K = int(K)
    nmax = int(nmax)
    prng = np.random.RandomState(seed) if seed else np.random
    alpha = theta0
    beta = (K-1)*theta0
    Q = prng.beta(alpha, beta)
    d = np.zeros(nmax+1)
    d[0] = 1
    c = optdynlib.recursionmatrix(alpha, beta, nmax)
    lambdan = optdynlib.lambdan(alpha, beta, nmax)
    ts = [0.0]
    ts.extend(optdynlib.poisson_times(rate=rate, tmax=tend, seed=prng.randint(0, 10000)))
    dts = np.diff(ts)
    stepQ = lambda Q, dt, prng: optdynlib.step1ddiffusionanalytical(Q,
                                                                    dt/(tauQ*0.5*(alpha+beta)),
                                                                    alpha, beta,
                                                                    dtmax=dtmax, prng=prng)
    nsteps = len(dts)
    ds = [d]
    Qs = [Q]
    for i in range(nsteps):
        dt = dts[i]
        Q = stepQ(Q, dt, prng)
        d = optdynlib.dpredict(d, dt/tauQ, lambdan)
        if prng.rand() < Q:
            d = optdynlib.dstep(d, c)
        else:
            d = optdynlib.dstep_opp(d, c)
        ds.append(d)
        Qs.append(Q)
    ts = np.asarray(ts)
    ds = np.asarray(ds)
    Qs = np.asarray(Qs)
    qest = c[0, 0]*ds[:, 0] + c[0, 1]*ds[:, 1]
    return ts, Qs, qest

theta0 = 0.02
K = 500
print(K * theta0)
tauQ = 200.0
tend = 1000
seed = 23173
t, q, qh = run(theta0=theta0, K=K, tauQ=tauQ, tend=tend, nmax=100, dtmax=1e-2, seed=seed)
tc, qc, qhc = runcounts(theta0=theta0, K=K, tauQ=tauQ, tend=tend, dtmax=1e-2, seed=seed)

lambda_ = 10.0
fig, ax = plt.subplots(figsize=(2.7, 2.0))
ax.plot(t/lambda_, q, label='$Q_a$')
ax.plot(t/lambda_, qh, label=r'$P^\star_a$', lw=2, alpha=.7)
ax.plot(t/lambda_, qhc, label=r'$P_a$', alpha=.7)
ax.set_yscale('log')
ax.set_ylim(0.0002, 0.2)
ax.set_xlim(0, 100)
ax.set_xlabel('time in years')
ax.set_ylabel('frequency')
plotting.despine(ax)
ax.legend(ncol=1, loc='upper left')
fig.tight_layout(pad=0.3)
plt.show()
fig.savefig('fig2.svg')
fig.savefig('fig2.png', dpi=300)
