import math
import numpy as np
import matplotlib.pyplot as plt

import projgrad

import sys
sys.path.append('../lib')
import plotting
from optdynlib import *

plt.style.use(['../optdynim.mplstyle'])


def logcost(p, Q, F):
    Ptilde = np.dot(p, F)
    f = -np.sum(Q * np.log(Ptilde))
    return f

def optp_numerical(Q, F, p0factor=1.0, **kwargs):
    defaultkwargs = dict(reltol=1e-6, nboundupdate=1, algo='slow')
    defaultkwargs.update(kwargs)
    N = Q.shape
    p0 = p0factor*np.ones(N)/N
    def objective(p):
        Ptilde = np.dot(p, F)
        f = -np.sum(Q * np.log(Ptilde))
        grad = -np.dot(Q / Ptilde, F)
        return f, grad
    return projgrad.minimize(objective, p0, **defaultkwargs)

def ei(N, i):
    ei = np.zeros(N)
    ei[i] = 1.0
    return ei

sigma = 0.01
Delta = 0.5 * sigma
K = int(1.0/Delta)
# ensure K is even
K = K if K % 2 == 0 else K+1
x = np.linspace(0.0, 1.0, K, endpoint=False)
def func(x, sigma):
    return np.exp(- x**2 /  (2.0 * sigma**2))
F = build_1d_frp_matrix(func, x, sigma*np.ones(len(x)))

thetatot = 10.0
n = thetatot * np.ones(K)/K

q = n/np.sum(n)
Fnorm = F.copy()
Fnorm /= Fnorm[:, 0].sum()
pnew = q + (Fnorm[K//2] - q)/(np.sum(n)+1.0)
oF = projgrad.project_simplex(pnew)

nplus = n + ei(K, K//2)
qplus = nplus / np.sum(nplus)
opt = optp_numerical(qplus, F, reltol=1e-9, algo='fast')
on = opt.x

optlocal = optp_numerical(qplus, F, p0factor=np.sum(n)/(np.sum(n)+1.0),
                          mask=np.abs(x-0.5)>2.0*sigma+1e-10, maxiters=1e4)
olocal = optlocal.x

optcost = opt.fun
optlocalcost = optlocal.fun
Fcost = logcost(oF, qplus, F)
uniformcost = logcost(np.ones(K)/K, qplus, F)

costgaps = []
for cost in [optlocalcost, Fcost]:
    costgaps.append(np.abs((cost-optcost)/(uniformcost-optcost)))
costgaps

fig, ax = plt.subplots(figsize=(6, 3.5))
lss = ['-', '-', '--', ':', '--']
axinset = plt.axes([.2, .65, .15, .3])
for i, (label, p) in enumerate([('optimal, 0.0', on),
                 #(r'$F^{-1}$, %g'%round(costgaps[3], 2), oFinv),
                 ('only local changes, %g'%round(costgaps[0], 2), olocal),
#                 ('delta, %g'%round(costgaps[2], 2), odelta),
#                 ('proportional to affinity, %g'%round(costgaps[1], 2), oF),
                                ]):
    l, = ax.plot((x-0.5)/sigma, p, lss[i], label=label)
    axinset.plot((x-0.5)/sigma, p, lss[i], label=label, c=l.get_color())
axinset.set_xlim(-5, 5)
axinset.set_yscale('log')
axinset.set_ylim(2e-3, 2e-1)

ax.set_ylim(0.0, 0.03)
ax.set_xlim(-20.0, 20.0)
ax.legend(loc='upper right', title='Update rule, relative cost increase')
ax.set_ylabel('$P^\star$')
ax.set_xlabel('Position / $\sigma$')


fig.tight_layout()
fig.savefig('figS5.svg')
fig.savefig('figS5.png', dpi=300)
