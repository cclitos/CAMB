"""Cosmic string (unconnected segment model) UETC eigenmode sources for CAMB.

Analytic USM UETC (Avgoustidis et al., 'Fast analytic computation of cosmic string power spectra'),
evaluated by quadrature over u = cos(theta).
<Th_a(k,t1) Th_b(k,t2)>/mu^2 = k/(xi^3 (k tmax)^3 (1-v^2)) * sum_i A_i [I_i(x_-,rho) - I_i(x_+,rho)]
x1,2 = k xi t1,2 ; x_pm=(x1 pm x2)/2 ; rho = k|t1-t2| v ; L_f -> 1.
"""
import numpy as np
from scipy.special import j0, j1, sici

NGL = 1500
_u, _w = np.polynomial.legendre.leggauss(NGL)
U = 0.5 * (_u + 1.0)      # nodes on [0,1]
W = 0.5 * _w
S2 = 1.0 - U ** 2


def dI(xm, xp, rho):
    """Delta I_i = I_i(x_-,rho) - I_i(x_+,rho), i=1..6; arrays of equal shape (N,). Returns (6,N)."""
    N = xm.shape[0]
    out = np.zeros((6, N))
    for a in range(0, N, 200):
        sl = slice(a, min(a + 200, N))
        xm_, xp_, r_ = xm[sl, None], xp[sl, None], rho[sl, None]
        s = np.sqrt(S2)[None, :]
        z = r_ * s
        J0 = j0(z)
        J1o = np.where(z > 1e-8, j1(z) / np.maximum(z, 1e-300), 0.5)
        dc = np.cos(xm_ * U[None, :]) - np.cos(xp_ * U[None, :])
        cm, cp = np.cos(xm_ * U[None, :]), np.cos(xp_ * U[None, :])
        w = W[None, :]
        # (1/2) int_{-1}^{1} du = int_0^1 du  (even integrands)
        out[0, sl] = (w * dc / U[None, :] ** 2 * J0).sum(1)
        out[1, sl] = (w * dc * J0).sum(1)
        out[2, sl] = (w * dc * S2[None, :] * J0).sum(1)
        out[3, sl] = (w * dc / U[None, :] ** 2 * J1o).sum(1)
        out[4, sl] = (w * dc * J1o).sum(1)
        out[5, sl] = (w * dc * S2[None, :] * J1o).sum(1)
    return out


def amplitudes(kind, v, al, rho):
    v2, v4, a2, a4 = v * v, v ** 4, al * al, al ** 4
    r2 = rho ** 2
    z = np.zeros_like(rho)
    if kind == '00':
        return [2 * a2 + z, z, z, z, z, z]
    if kind == '0S':
        return [1 + v2 * (2 * a2 - 1) + z, -3 * (1 + v2 * (a2 - 1)) + z, z, -3 * v2 * a2 + z, 3 * v2 * a2 + z, z]
    if kind == 'SS':
        c = (1 + v2 * (2 * a2 - 1))
        return [(-27 * v4 * a4 + r2 * c ** 2) / (2 * a2 * r2),
                3 * (9 * v4 * a4 + r2 * (1 - 2 * v2 - v4 * (a4 - 1))) / (2 * a2 * r2),
                -9 * (1 + v2 * (a2 - 1)) ** 2 / (2 * a2) + z,
                3 * v2 * (v2 * (1 + a2 * (9 / r2 - 2)) - 1),
                -3 * v2 * (v2 * (1 + a2 * (9 / r2 - 2)) - 1),
                9 * v2 * (1 + v2 * (a2 - 1)) + z]
    if kind == 'TT':
        return [(r2 * (v2 - 1) ** 2 - 3 * a4 * v4) / (4 * a2 * r2),
                (r2 * (-1 + 2 * v2 + v4 * (a4 - 1)) + 3 * a4 * v4) / (4 * a2 * r2),
                -(1 + v2 * (a2 - 1)) ** 2 / (4 * a2) + z,
                v2 / 2 * (1 + v2 * (3 * a2 / r2 - 1)),
                -v2 / 2 * (1 + v2 * (3 * a2 / r2 - 1)),
                v2 / 2 * (1 + v2 * (a2 - 1)) + z]
    if kind == 'VV':
        return [3 * v4 * a2 / r2, -3 * v4 * a2 / r2, (1 + v2 * (a2 - 1)) ** 2 / a2 + z,
                -v4 * a2 * (6 / r2 - 1), v4 * a2 * (6 / r2 - 1), -2 * v2 * (1 + v2 * (a2 - 1)) + z]
    raise ValueError(kind)


def Bcoef(kind, v, al, x1, x2, rho):
    v2, v4, a2, a4 = v * v, v ** 4, al * al, al ** 4
    g = 1 + v2 * (a2 - 2) + v4 * (1 - a2 + a4)
    if kind == '00':
        return a2 * x1 * x2
    if kind == '0S':
        return (x1 * x2 * (x1 ** 2 + x2 ** 2) * (2 + v2 * (a2 - 2)) / 360
                - x1 * x2 * rho ** 2 * (1 + v2 * (2 * a2 - 1)) / 30)
    if kind == 'SS':
        return x1 * x2 * g / (5 * a2)
    return x1 * x2 * g / (15 * a2)       # TT, VV


def uetc(kind, v, al, xi, y1, y2, xsmall=0.02, rhofloor=2e-3):
    """Scaled UETC (t1 t2)^{1/2}<Th Th>/mu^2 at z1=exp(y1), z2=exp(y2)=k t; y1,y2 arrays (N,)."""
    z1, z2 = np.exp(y1), np.exp(y2)
    x1, x2 = xi * z1, xi * z2
    xm, xp = 0.5 * (x1 - x2), 0.5 * (x1 + x2)
    rho = np.maximum(v * np.abs(z1 - z2), rhofloor)
    sumAI = np.zeros_like(z1)
    big = np.maximum(x1, x2) >= xsmall
    if big.any():
        d = dI(xm[big], xp[big], rho[big])
        A = amplitudes(kind, v, al, rho[big])
        sumAI[big] = sum(A[i] * d[i] for i in range(6))
    sm = ~big
    if sm.any():
        sumAI[sm] = Bcoef(kind, v, al, x1[sm], x2[sm], v * np.abs(z1[sm] - z2[sm]))
    zmax = np.maximum(z1, z2)
    return np.sqrt(z1 * z2) / (xi ** 3 * zmax ** 3 * (1 - v * v)) * sumAI


def etc_C(kind, v, al, x):
    v2, v4, a2, a4 = v * v, v ** 4, al * al, al ** 4
    Si = sici(x)[0]
    sinc = np.sin(x) / x
    g = 1 + v2 * (a2 - 2) + v4 * (1 - a2 + a4)
    if kind == '00':
        return 2 * a2 * (-1 + np.cos(x) + x * Si)
    if kind == '0S':
        return (2 + v2 * (a2 - 2)) * (-4 + np.cos(x) + 3 * sinc + x * Si) / 2
    if kind == 'TT':
        g3 = 1 + v2 * (a2 - 2) + v4 * (1 - a2 + 3 * a4 / 8)
        g4 = 1 + v2 * (a2 - 2) + v4 * (1 - a2 - 5 * a4 / 8)
        return (sinc * g3 / (2 * x ** 2 * a2) + sinc * g4 / (4 * a2) - 2 * (1 - v2) * (1 + v2 * (a2 - 1)) / (3 * a2)
                + ((1 - 2 / x ** 2) * np.cos(x) + x * Si) * g3 / (4 * a2))
    if kind == 'VV':
        g3 = 1 + v2 * (a2 - 2) + v4 * (1 - a2 + 3 * a4 / 8)
        return v4 * a2 * (-2 + np.cos(x) + sinc + x * Si) / 8 + 2 * g3 / (x ** 3 * a2) * (x ** 3 / 3 + x * np.cos(x) - np.sin(x))
    if kind == 'SS':
        c11 = 1 - a2 + 11 * a4 / 8
        c3 = 1 - a2 + 3 * a4 / 8
        c4 = 1 - a2 - a4 / 8
        return (np.cos(x) * ((x ** 2 - 18) + v2 * (x ** 2 - 18) * (a2 - 2)) / (2 * x ** 2 * a2)
                + np.cos(x) * v4 * (x ** 2 * c11 - 18 * c3) / (2 * x ** 2 * a2)
                - 2 * g / a2
                + x * Si * (1 + v2 * (a2 - 2) + v4 * c11) / (2 * a2)
                + 3 * sinc * ((6 - x ** 2) + v2 * (6 - x ** 2) * (a2 - 2)) / (2 * x ** 2 * a2)
                + 3 * sinc * v4 * (6 * c3 - x ** 2 * c4) / (2 * x ** 2 * a2))


def uetc_matrix(kind, v, alpha, xi, y):
    """Symmetric scaled UETC on the ln(k tau) grid y (uniform), kind in 00, SS, TT, VV; the 0S block is not symmetric
    (<00(t1) S(t2)>) and is built full."""
    n = len(y)
    if kind == "0S":
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        return uetc(kind, v, alpha, xi, y[I.ravel()], y[J.ravel()]).reshape(n, n)
    iu = np.triu_indices(n)
    M = np.zeros((n, n))
    M[iu] = uetc(kind, v, alpha, xi, y[iu[0]], y[iu[1]])
    return M + M.T - np.diag(np.diag(M))


def eigenmodes(M, y, gamma=0.25, nmodes=64):
    """Eigen-decompose (k tau1 k tau2)^gamma * UETC on the uniform ln(k tau) grid y.
    Returns F (nmodes, n) with sum_i F_i(x1) F_i(x2) = weighted UETC, plus eigenvalues and captured trace fraction."""
    w = np.exp(gamma * y)
    Mw = w[:, None] * M * w[None, :]
    ev, U = np.linalg.eigh(Mw)
    ev, U = ev[::-1], U[:, ::-1]
    m = nmodes
    F = (np.sqrt(np.maximum(ev[:m], 0.0))[None, :] * U[:, :m]).T
    return F, ev[:m], ev[:m].sum() / ev[ev > 0].sum()
