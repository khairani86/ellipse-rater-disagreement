"""
mfrm_fast.py - adjacent-category many-facet Rasch model with random essay ability,
MMLE by Gauss-Hermite quadrature, ANALYTIC gradient (same model/parameterisation as
rq1/severity.py MFRM; used for RQ2 re-estimation loops).

  log P(X=k)/P(X=k-1) = theta - s_r - tau_k ,  theta_i ~ N(mu_g(i), sigma^2)

parameters p = [t(3) | sfree(R-1) | mu(G) | log sigma]; tau = [t, -sum t]; s = [sfree, -sum sfree].
"""
import numpy as np
from scipy import optimize
from scipy.special import logsumexp

K = 5
KK = np.arange(K)


class FastMFRM:
    def __init__(self, ia, ib, xa, xb, g, R, G, Q=21, cnt=None):
        """ia, ib: rater index (0..R-1) of slot a/b; xa, xb: scores 0..4; g: segment index;
        cnt: optional integer weights (identical rows merged; likelihood is unchanged)."""
        self.ia, self.ib = np.asarray(ia), np.asarray(ib)
        self.xa, self.xb = np.asarray(xa), np.asarray(xb)
        self.g = np.asarray(g)
        self.cnt = np.ones(len(self.g)) if cnt is None else np.asarray(cnt, float)
        self.R, self.G, self.n = R, G, len(self.g)
        nodes, w = np.polynomial.hermite_e.hermegauss(Q)
        self.nodes = nodes
        self.logw = np.log(w / np.sqrt(2 * np.pi))
        self.Q = Q

    def unpack(self, p):
        t = p[:3]
        tau = np.concatenate([t, [-t.sum()]])
        sfree = p[3:3 + self.R - 1]
        s = np.concatenate([sfree, [-sfree.sum()]])
        mu = p[3 + self.R - 1:3 + self.R - 1 + self.G]
        sig = np.exp(p[-1])
        return tau, s, mu, sig

    def init(self):
        marg = np.bincount(np.concatenate([self.xa, self.xb]), weights=np.concatenate([self.cnt, self.cnt]), minlength=K) / (2 * self.cnt.sum())
        lg = np.log(np.maximum(marg, 1e-6))
        tau = -np.diff(lg)
        tau = tau - tau.mean()
        return np.concatenate([tau[:3], np.zeros(self.R - 1), np.zeros(self.G), [0.0]])

    def nll_grad(self, p, want_grad=True):
        tau, s, mu, sig = self.unpack(p)
        Q, n = self.Q, self.n
        cum = np.concatenate([[0.0], np.cumsum(tau)])
        theta = mu[self.g][:, None] + sig * self.nodes[None, :]
        lp = np.zeros((n, Q))
        slots = []
        for ir, xr in ((self.ia, self.xa), (self.ib, self.xb)):
            eta = theta - s[ir][:, None]
            U = eta[:, :, None] * KK - cum
            m = U.max(2, keepdims=True)
            E = np.exp(U - m)
            Z = E.sum(2)
            lse = m[:, :, 0] + np.log(Z)
            lp += eta * xr[:, None] - cum[xr][:, None] - lse
            if want_grad:
                slots.append((ir, xr, E / Z[:, :, None]))
        a = lp + self.logw[None, :]
        ll = logsumexp(a, axis=1)
        nll = -(self.cnt * ll).sum()
        if not want_grad:
            return nll
        post = np.exp(a - ll[:, None]) * self.cnt[:, None]
        g_cum = np.zeros(K)
        g_s = np.zeros(self.R)
        g_mu = np.zeros(self.G)
        g_ls = 0.0
        sn = sig * self.nodes[None, :]
        for ir, xr, P in slots:
            Ek = P @ KK
            wq = post * (xr[:, None] - Ek)
            sw = wq.sum(1)
            g_s -= np.bincount(ir, weights=sw, minlength=self.R)
            g_mu += np.bincount(self.g, weights=sw, minlength=self.G)
            g_ls += (wq * sn).sum()
            g_cum += np.einsum("nq,nqk->k", post, P) - np.bincount(xr, weights=self.cnt, minlength=K)
        g_tau = np.cumsum(g_cum[::-1])[::-1][1:]            # d/d tau_m = sum_{k>=m} g_cum[k], m=1..4
        g_t = g_tau[:3] - g_tau[3]
        g_sf = g_s[:self.R - 1] - g_s[self.R - 1]
        grad_ll = np.concatenate([g_t, g_sf, g_mu, [g_ls]])
        return nll, -grad_ll

    def fit(self, p0=None, maxiter=500, gtol=1e-5):
        if p0 is None:
            p0 = self.init()
        res = optimize.minimize(self.nll_grad, p0, jac=True, method="L-BFGS-B",
                                options=dict(maxiter=maxiter, maxfun=100000, gtol=gtol, ftol=1e-12))
        self.res = res
        return res


def expected_shift(tau, s_c, mu_bar):
    """delta_r = E[X | theta=mu_bar, s_r] - E[X | theta=mu_bar, s=0] in score points (1..5)."""
    cum = np.concatenate([[0.0], np.cumsum(tau)])

    def ex(s):
        u = KK * (mu_bar - s) - cum
        p = np.exp(u - logsumexp(u))
        return (p * (KK + 1)).sum()
    e0 = ex(0.0)
    return np.array([ex(s) - e0 for s in s_c])


def severity_from_fit(m, p, n_ratings):
    """Centre severity to ratings-weighted mean zero (shifting mu equally), return
    (tau, s_c, mu_c, sig, mu_bar, delta)."""
    tau, s, mu, sig = m.unpack(p)
    w = n_ratings / n_ratings.sum()
    c = (w * s).sum()
    s_c, mu_c = s - c, mu - c
    mu_bar = np.average(mu_c[m.g], weights=m.cnt)
    return tau, s_c, mu_c, sig, mu_bar, expected_shift(tau, s_c, mu_bar)


def compress(ia, ib, xa, xb, g):
    """Canonicalise slot order (ia<=ib) and merge identical (ia,ib,xa,xb,g) rows."""
    ia, ib, xa, xb, g = map(np.asarray, (ia, ib, xa, xb, g))
    sw = ia > ib
    ia2, ib2 = np.where(sw, ib, ia), np.where(sw, ia, ib)
    xa2, xb2 = np.where(sw, xb, xa), np.where(sw, xa, xb)
    key = (((ia2 * 64 + ib2) * 5 + xa2) * 5 + xb2) * 16 + g
    u, first, cnt = np.unique(key, return_index=True, return_counts=True)
    return ia2[first], ib2[first], xa2[first], xb2[first], g[first], cnt
