import numpy as np
from scipy.integrate import odeint
import scipy.stats

def poisson_times(rate=1.0, tmax=1.0, seed=None):
    t = 0.0
    ts = []
    prng = np.random.RandomState(seed)
    while True:
        t += prng.exponential(1.0/rate)
        if t < tmax:
            ts.append(t)
        else:
            break
    return np.asarray(ts)

class TelegraphProcess(object):
    def __init__(self, mu, lambda_, t0=0.0, seed=None):
        """
        mu: switching rate towards state 0
        lambda: switching rate towards state 1
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.prng = np.random.RandomState(seed)
        self.state = 0 if self.prng.rand() < self.steadystatepdf()[0] else 1
        self.told = 0.0
    def get_state(self, t):
        p0 = self.propagator(0, t, self.state, self.told)
        self.state = 0 if self.prng.rand() < p0 else 1
        self.told = t
        return self.state
    def steadystatepdf(self):
        return np.array([self.mu, self.lambda_])/(self.mu + self.lambda_)
    def propagator(self, x, t, x0, t0):
        mu = self.mu
        lambda_ = self.lambda_
        return mu/(lambda_+mu) + (np.where(x==x0, lambda_/(lambda_+mu), -mu/(lambda_+mu))
                                  * np.exp(-(lambda_+mu)*np.abs(t-t0)))
    def loglikelihood(self, t, a):
        t = np.asarray(t)
        a = np.asarray(a)
        propagators = self.propagator(a[1:], t[1:], a[:-1], t[:-1])
        return np.sum(np.log(propagators))

#def logcost(pathogen, x, nburnin=None, costfunc=lambda p: -np.log(p)):
#    x = np.asarray(x)
#    costs = costfunc(x[range(len(pathogen)), pathogen])
#    if nburnin:
#        costs = costs[nburnin:]
#    return np.mean(costs), np.std(costs, ddof=1)/len(costs)**.5

def powercost(Q, P=None, Qest=None, alpha=1.0):
    if Qest:
        P = np.asarray(Qest)**(1.0/(1.0+alpha))
        P /= np.sum(P)
    return np.sum(Q/P**alpha)

def n_to_p(n, alpha=1.0):
    # q propto n so we do not first need to calculate q
    p = np.asarray(n)**(1.0/(1.0+alpha))
    p /= np.sum(p)
    return p

def logcost(Q, Qest):
    return -np.sum(Q*np.log(Qest))

def add_binned_stat(array, x, y, bins):
    "bin y according to x in bins and add to array"
    statistic, bin_edges, binnumbers = scipy.stats.binned_statistic(x, y, bins=bins)
    mask = ~np.isnan(statistic)
    array[mask] += statistic[mask]

def lognormdist(kappa, N, prng=np.random):
    """Lognormally distributed abundances normalized"""
    Q = prng.lognormal(mean=0.0,
                        sigma=sigma_lognormal_from_cv(kappa),
                        size=N)
    Q /= np.sum(Q)
    return Q

def recursionmatrix(alpha, beta, nmax):
    c = np.zeros((nmax+1, nmax+1))
    c[0, 0] = alpha/(alpha+beta)
    c[1, 0] = 1.0/(alpha+beta)
    for n in range(1, nmax+1):
        c[n-1, n] = (n+alpha-1.0)*(n+beta-1.0)/((2.0*n+alpha+beta-1.0)*(2.0*n+alpha+beta-2.0))
        c[n, n] = 0.5 - (beta**2-alpha**2-2.0*(beta-alpha))/(2*(2*n+alpha+beta)*(2*n+alpha+beta-2))
        if n < nmax:
            c[n+1, n] = (n+1)*(n+alpha+beta-1)/((2*n+alpha+beta)*(2*n+alpha+beta-1))
    return c

def meancost(P, F, Q, g, gdiff=None, grad=True):
    """Calculates mean cost of infection.

    P: distribution of receptors
    F: cross-reactivity matrix
    g: mapping from Ptilde to cost of infection
    gdiff: derivative of g [needed if grad=True]
    """
    Ptilde = np.dot(P, F)
    f = np.sum(Q * g(Ptilde))
    if not grad:
        return f
    grad = np.dot(Q * gdiff(Ptilde), F)
    return f, grad

def lambdan(alpha, beta, nmax):
    n = np.arange(0, nmax+1)
    return 0.5*n*(n+alpha+beta-1)

def dstep(d, c):
    dp = c.dot(d)
    dp /= dp[0]
    return dp

def dstep_opp(d, c):
    dp = d - c.dot(d)
    dp /= dp[0]
    return dp

def dpredict(d, dt, lambdan):
    return np.exp(-lambdan*dt)*d

def sigma_lognormal_from_cv(cv):
    """ Lognormal parameter sigma from coefficient of variation. """
    return (np.log(cv**2 + 1.0))**.5

def ei(N, i):
    "return ith unit vector"
    vec = np.zeros(N)
    vec[i] = 1.0
    return vec

def integrate_popdyn_stoch_fixedsampling(Q, A, g, d, tend, dt=1e0, frp=None, n0=None, nsave=1,
                           callback=None, prng=np.random):
    """
        Simulate stochastic population dynamics.

        Q : pathogen distribution,
        A : availability function
        g : \overline{F}
        d : free parameter of population dynamics (death rate)
        dt : time interval that each pathogen is present
        frp : cross-reactivity matrix, None for one-to-one mapping
        n0 : initial population distribution
        nsave : how often to save
        callback : function to be called at each time point
    """

    def cost(n):
        p = n / np.sum(n)
        if frp is None:
            return np.sum(Q * g(p))
        return np.sum(Q * g(np.dot(p, frp)))

    if n0 is None:
        if frp is None:
            n = np.ones(Q.shape[0]) 
        else:
            n = np.ones(frp.shape[1]) 
    else:
        n = np.copy(n0)

    nsteps = int(np.ceil(tend / dt))
    nsavetot = nsteps // nsave
    ns = np.empty((nsavetot, len(n)))
    costs = np.empty((nsavetot,))
    ts = np.empty((nsavetot,))

    inds = prng.choice(len(Q), size=nsteps, p=Q)

    def f(i, n):
        ind = inds[i]
        if frp is None:
            f = -n*d
            f[ind] += n[ind]*A(n[ind])
            return f
        return n * (A(np.sum(frp[ind] * n)) * frp[ind] - d)

    for i in range(nsteps):
        n += dt * f(i, n)
        if callback:
            callback(i*dt, n, cost(n))
        if (i+1) % nsave == 0:
            ns[i / nsave] = n
            costs[i / nsave] = cost(n)
            ts[i / nsave] = dt * (i+1)
    return ts, ns, costs

def integrate_popdyn_stoch(Q, A, g, tend, rate=1.0, frp=None, n0=None,
                           stepn = None,
                           stepQ = None,
                           n_to_p = None,
                           nsave=1, full_output=True,
                           callback=None, prng=np.random):
    """
        Simulate stochastic population dynamics.

        Q : pathogen distribution (initial distribution if stepQ != None)
        A : availability function
        g : \overline{F}
        rate : rate with which pathogens are encountered
        frp : cross-reactivity matrix, None for one-to-one mapping
        n0 : initial population distribution
        stepn: prediction phase time stepper for repertoire distribution
        stepQ: time stepper for pathogen distribution
        n_to_p : mapping from counts to probability (not necessarily normalized), use to simuylate precise optimal dynamics
        nsave : how often to save
        full_output : if True return ts, ns, costs, ecosts, else return ts, costs, ecosts
        callback : function to be called at each time point
    """

    def cost(n, Q, ind=None):
        """return cost for infection with pathogen ind
        if ind=None return expected cost of infection"""
        if n_to_p is not None:
            p = n_to_p(n)
        else:
            p = n
        p = p / np.sum(p)
        if ind is not None:
            if frp is None:
                return g(p[ind])
            return g(np.dot(p, frp)[ind])
        if frp is None:
            return np.sum(Q * g(p))
        return np.sum(Q * g(np.dot(p, frp)))

    if n0 is None:
        if frp is None:
            n = np.ones(Q.shape[0]) 
        else:
            n = np.ones(frp.shape[1]) 
    else:
        n = np.copy(n0)

    ts = [0.0]
    ts.extend(poisson_times(rate=rate, tmax=tend, seed=prng.randint(0, 10000)))
    dts = np.diff(ts)

    nsteps = len(dts)
    nsavetot = nsteps // nsave
    if full_output:
        ns = np.empty((nsavetot, len(n)))
    # cost of infections
    costs = np.empty((nsavetot,))
    # expected cost of next infections
    ecosts = np.empty((nsavetot,))
    tsave = np.empty((nsavetot,))
    # precompute which pathogens are encountered in static environment (faster to do in one batch)
    if not stepQ:
        inds = prng.choice(len(Q), size=nsteps, p=Q)

    def f(ind, n):
        if frp is None:
            f = np.zeros(n.shape)
            f[ind] = n[ind]*A(n[ind])
            return f
        return n * A(np.sum(frp[ind] * n)) * frp[ind]

    for i in range(nsteps):
        save = i % nsave == 0
        dt = dts[i]
        if stepn:
            n = stepn(n, dt)
        if stepQ:
            Q = stepQ(Q, dt)
            ind = prng.choice(len(Q), p=Q)
        else:
            ind = inds[i]
        if save:
            isave = i / nsave
            costs[isave] = cost(n, Q, ind)
        n += f(ind, n)
        if callback:
            callback(ts[i+1], n, cost(n, Q))
        if save:
            if full_output:
                ns[isave] = n
            ecosts[isave] = cost(n, Q)
            tsave[isave] = ts[i+1]
    if full_output:
        return tsave, ns, costs, ecosts
    return tsave, costs, ecosts

def WFdiffusion_ev(n, alpha, beta):
    "eigenvalues of Wright-Fisher diffusion operator"
    return 0.5*n*(n+alpha+beta-1.0)

def project(x, mask=None):
    """ Take a vector x (with possible nonnegative entries and non-normalized)
        and project it onto the unit simplex.

        mask:   do not project these entries
                project remaining entries onto lower dimensional simplex
    """
    if mask is not None:
        mask = np.asarray(mask)
        xsorted = np.sort(x[~mask])[::-1]
        # remaining entries need to sum up to 1 - sum x[mask]
        sum_ = 1.0 - np.sum(x[mask])
    else:
        xsorted = np.sort(x)[::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
    lambda_a = (np.cumsum(xsorted) - sum_) / np.arange(1.0, len(xsorted)+1.0)
    for i in xrange(len(lambda_a)-1):
        if lambda_a[i] >= xsorted[i+1]:
            astar = i
            break
    else:
        astar = -1
    p = np.maximum(x-lambda_a[astar],  0)
    if mask is not None:
        p[mask] = x[mask]
    return p

def step1ddiffusion(q, dt, alpha, beta, dtfactor=None, dtmax=0.001, prng=np.random):
    def clip(x, xmin=0.0, xmax=1.0):
        if x < xmin:
            return xmin
        if x > xmax:
            return xmax
        return x
    dtmax = min(dtfactor/(alpha+beta), dtmax) if dtfactor else dtmax
    if dt < dtmax:
        q += dt*0.5*(alpha-(alpha+beta)*q)+(q*(1.0-q)*dt)**.5 * prng.normal()
        return clip(q, 0, 1)
    nsteps = int(dt/dtmax)+1
    dt /= nsteps
    rand = prng.normal(size=nsteps)
    for i in range(nsteps):
        q += dt*0.5*(alpha-(alpha+beta)*q)+(q*(1.0-q)*dt)**.5 * rand[i]
        q = clip(q, 0, 1) 
    return q

def step1ddiffusionanalytical(q, dt, alpha, beta, prng=np.random, **kwargs):
    """Analytical time stepping as proposed in Jenkins, Spano arXiv:1506.06998
    
       Uses the asymptotic normality of the death process for small times
       (see Griffiths, J. Math. Bio, 1984)
    """
    theta = alpha+beta
    beta_ = 0.5*(theta-1.0)*dt
    if beta_ == 0.0:
        eta = 1.0
        sigma = (2.0/(3.0*dt))**.5
    else:
        eta =  beta_/np.expm1(beta_)
        # calculation can sometimes give negative numbers due to numerical precision
        factor = max(0, 2.0*eta/dt *(1.0 + eta/(eta+beta_)-2.0*eta))
        sigma = max((eta+beta_) * factor**.5 / beta_, 1e-16)
    mu = 2.0*eta/dt
    m = max(int(round(prng.normal(mu, sigma))), 0)
    l = prng.binomial(m, q)
    qnew = prng.beta(alpha+l, beta+m-l)
    return qnew


def stepdiffusionanalytical(q, dt, theta0, prng=np.random, **kwargs):
    """Analytical time stepping in multiple dimensions
    
       Uses technique proposed in Jenkins, Spano arXiv:1506.06998 
    
       Uses the asymptotic normality of the death process for small times
       (see Griffiths, J. Math. Bio, 1984)
    """
    theta = len(q)*theta0
    beta_ = 0.5*(theta-1.0)*dt
    if beta_ == 0.0:
        eta = 1.0
        sigma = (2.0/(3.0*dt))**.5
    else:
        eta =  beta_/np.expm1(beta_)
        # calculation can sometimes give negative numbers due to numerical precision
        factor = max(0, 2.0*eta/dt *(1.0 + eta/(eta+beta_)-2.0*eta))
        sigma = max((eta+beta_) * factor**.5 / beta_, 1e-16)
    mu = 2.0*eta/dt
    m = max(int(round(prng.normal(mu, sigma))), 0)
    l = prng.multinomial(m, q)
    qnew = prng.dirichlet(theta0*np.ones(len(q))+l)
    return qnew


def stepPIMWFdiffusion(Q, dt, theta=1.0, prng=np.random):
    """time stepping routine for a symmetric Wright-Fisher diffusion
    
    symmetric = Parent-independent mutation
    
    dq_i = 0.5 (theta  - N theta q_i) dt + sum_j sigma_ij d B_j
    with independent Wiener processes B_j and sigma_ij = sqrt(q_i) (delta_ij - sqrt(q_i q_j))

    Q : initial frequencies
    dt : time step
    theta : mutation rate

    returns final frequencies
    """
    sqQ = Q**.5
    M = - sqQ[:, np.newaxis] * sqQ[np.newaxis, :]
    diag = np.diag_indices_from(M)
    M[diag] = 1.0 + M[diag]
    M *= sqQ[:, np.newaxis]
    B = prng.normal(size=len(Q))
    Q = project(Q + dt*0.5*(theta-len(Q)*theta*Q)
                          + dt**.5 * M.dot(B))
    return Q


class Counter:
    def __init__(self, theta, n0=None):
        """If theta is None it will be set equal to theta"""
        self.theta = theta
        self.N = len(theta)
        self.n = n0 if n0 else theta.copy()
        self.dn = lambda n, t: - 0.5 * (np.sum(n) - 1.0) * (n-self.theta)
    
    def predict(self, dt, euler=True):
        """dt: time step"""
        # euler algorithm
        if euler:
            self.n += dt * self.dn(self.n, 0.0)
            self.n = np.maximum(self.n, self.theta)
        else:
            # odeint algorithm 
            ys = odeint(self.dn, self.n, [0, dt])
            self.n = ys[-1, :]
    
    def update(self, a):
        """a: index of pathogen"""
        self.n += ei(self.N, a)
    
    def mean(self):
        return self.n/np.sum(self.n)


class CounterTwoCompartments:
    def __init__(self, theta):
        self.theta = theta
        self.nnaive = theta
        self.N = len(theta)
        self.nmemory = np.zeros(self.N)
   
    def ntot(self):
        return np.sum(self.nmemory + self.nnaive)

    def predict(self, dt):
        """dt: time step"""
        # euler algorithm
        self.nnaive += dt * (- 0.5 * (self.ntot() - 1.0) * (self.nnaive-self.theta))
        self.nmemory += dt * (- 0.5 * (self.ntot() - 1.0) * self.nmemory)
    
    def update(self, a):
        """a: index of pathogen"""
        self.nmemory += ei(self.N, a) * (1.0 + self.nnaive[a])
        self.nnaive[a] = 0.0
    
    def mean(self):
        return (self.nmemory+self.nnaive)/self.ntot()

def build_1d_frp_matrix(func, x, sigma, B=1):
    """ Builds quadratic frp matrix respecting pbc.

    func: Kernel function
    x: position of points
    sigma: width of Kernel
    """
    N = len(x)
    A = np.zeros((N, N))
    shifts = np.arange(-5, 6) * B
    for r in range(N):
        for p in range(N):
            value = 0
            for shift in shifts:
                value += func(x[r] - x[p] + shift, sigma[r])
            A[r, p] = value
    return A
