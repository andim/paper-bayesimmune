import cython
import numpy as np
cimport numpy as np

@cython.wraparound(False)
def popdyn(int L, np.ndarray[np.double_t] dts,
           np.ndarray[np.int_t] observations,
           double alpha, double beta, double tau):
    """Population dynamics solver.

    Parameter
    ---------

    L: int
        dimension of receptor/pathogen space

    dts: float array
        time intervals between observations

    observations: int array
        index of observed pathogen

    alpha: double
        constant update of population size upon encounter

    beta: double
        fractional update of population size upon encounter

    tau: double
        relaxation time scale
    """

    cdef np.ndarray[np.double_t, ndim=2] p = np.empty((len(observations), L),
                                                      dtype=np.float)
    cdef np.ndarray[np.double_t] n = np.ones(L, dtype=np.float)
    cdef np.ndarray[np.double_t] edts = np.exp(-dts/tau)
    cdef Py_ssize_t i, j
    cdef int observation
    cdef double nsum
    for i in range(dts.shape[0]):
        nsum = 0.0
        for j in range(n.shape[0]):
            n[j] = 1.0 + (n[j]-1.0)*edts[i]
            nsum += n[j]
        for j in range(n.shape[0]):
            p[i, j] = n[j]/nsum
        observation = observations[i]
        n[observation] = alpha + (1.0+beta)*n[observation]
    return p

cdef double _clip(double x, double xmin=0.0, double xmax=1.0):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x

@cython.boundscheck(False)
def step1ddiffusion(double q, double dt, double alpha, double beta, double dtmax, prng=np.random):
    cdef double dW
    if dt < dtmax:
        dW = prng.normal()
        q += dt*0.5*(alpha-(alpha+beta)*q)+(q*(1.0-q)*dt)**.5 * dW
        return _clip(q, 0, 1)
    cdef int nsteps = int(dt/dtmax)+1
    dt /= nsteps
    cdef np.ndarray[np.double_t] rand = prng.normal(size=nsteps)
    for i in range(nsteps):
        q += dt*0.5*(alpha-(alpha+beta)*q)+(q*(1.0-q)*dt)**.5 * rand[i]
        q = _clip(q, 0, 1) 
    return q


