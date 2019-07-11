import math
import numpy as np
import matplotlib.pyplot as plt

import projgrad

import sys
sys.path.append('../lib')
import plotting
from optdynlib import *

plt.style.use(['../optdynim.mplstyle'])

sigma = 0.025
#Delta = 0.1
Delta = 2.0 * sigma
N = int(1.0/Delta)
# ensure N is always even
N = N if N % 2 == 0 else N+1
x = np.linspace(0.0, 1.0, N, endpoint=False)
def func(x, sigma):
    return np.exp(- x**2 /  (2.0 * sigma**2))
F = build_1d_frp_matrix(func, x, sigma*np.ones(len(x)))

fig, ax = plt.subplots()

i = N//2
dp = np.linalg.inv(F).dot(ei(N, i))
dp /= np.sum(dp)

ax.plot((x-0.5)/sigma, dp, 'o-', label='$F^{-1} e_a$')

sigmacorr = 2*sigma
dpcorr = np.linalg.inv(F).dot(np.exp(-(Delta*(np.arange(N)-i))**2/(2*sigmacorr**2)))
dpcorr /= np.sum(dpcorr)
ax.plot((x-0.5)/sigma, dpcorr, 'o-', label='$F^{-1} f_a$')
ax.set_xlabel('$x/\sigma$')
ax.set_ylabel('relative update')
ax.set_xlim(-10, 10)
ax.legend()
plotting.despine(ax)
fig.tight_layout()
fig.savefig('figS4.svg')
fig.savefig('figS4.png', dpi=300)
