#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of the QSOSED and AGNSED models from Kubota & Done (2018).
Please see Kubota & Done, 2018, MNRAS, 480, 1247

Includes:
    - NTHCOMP Comptonisation model (Zdziarski, Johnson & Magdziarz 1996;
      Zycki, Done & Smith 1999) — adapted from the Xspec subroutine donthcomp.f
    - agnsed: Full AGN SED model with disc, warm Compton, and hot Compton regions
    - qsosed: Simplified QSOSED model (inherits from agnsed) — automatically
      determines r_hot, r_warm, gamma_hot, and other parameters

Original code by Scott Hagen. Consolidated into a single file for Speculate.
"""

import numpy as np
import warnings
import astropy.constants as const
import astropy.units as u


# =========================================================================
# NTHCOMP  —  Comptonisation model
# =========================================================================

def _mdonthcomp(ear, gamma, kTe, kTbb):
    """
    Adapted from the subroutine donthcomp.f, distributed with XSPEC.
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)

    Seed photon spectrum is always black-body. gamma, kTe, or kTbb can be
    arrays — the code returns a spectrum for each parameter value.

    Parameters
    ----------
    ear : array
        Input energy grid — Units : keV.
    gamma : array
        Photon index — Units : dimensionless
    kTe : array
        Plasma electron temperature — Units : keV
    kTbb : array
        Black-body temperature

    Returns
    -------
    photar : array
        Output spectrum — shape = (len(ear), len(par_i))
    """
    xth, nth, spt = _thcompt(kTbb / 511.0, kTe / 511.0, gamma)

    # Normalisation factor
    xx = 1 / 511.0
    ih = np.argwhere(xx > xth)
    ih = ih[ih < nth]
    ih = ih[-1] + 1

    spp = spt[ih - 1] + (spt[ih] - spt[ih - 1]) * (
        xx - xth[ih - 1]) / (xth[ih] - xth[ih - 1])
    normfac = 1.0 / spp

    # Re-cast onto original energy grid
    photar = np.zeros((len(ear), len(gamma)))
    prim = np.zeros((len(ear), len(gamma)))

    j = 0
    for i in range(len(ear)):
        while j < nth and 511 * xth[j] < ear[i]:
            j = j + 1

        if j <= nth:
            if j > 0:
                jl = j - 1
                prim[i] = spt[jl] + ((ear[i] / 511 - xth[jl]) * (
                    spt[jl + 1] - spt[jl]) / (xth[jl + 1] - xth[jl]))
            else:
                prim[i] = spt[0]

    ne = len(ear)
    photar[1:ne] = 0.5 * (prim[1:ne] / ear[1:ne, np.newaxis]**2
                           + prim[0:ne - 1] / ear[0:ne - 1, np.newaxis]**2)
    photar[1:] *= (ear[1:ne, np.newaxis] - ear[0:ne - 1, np.newaxis]) * normfac

    return photar


def _thcompt(tempbb, theta, gamma):
    """
    Adapted from the subroutine thCompton in donthcomp.f, included in Xspec.
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)

    Parameters
    ----------
    tempbb : array
        Black-body temperature, normalised by 511 keV.
    theta : array
        Plasma electron temperature, normalised by 511 keV.
    gamma : array
        Photon index.

    Returns
    -------
    x : array
        Energy grid used for calculations.
    jmax : int
        Max index.
    sptot : array
        Output spectrum.
    """
    # Thompson optical depth
    tautom = np.sqrt(2.25 + 3 / (theta * ((gamma + 0.5)**2 - 2.25))) - 1.5

    # Initialising arrays (length 900 as in donthcomp.f)
    dphesc = np.zeros((900, len(gamma)))
    dphdot = np.zeros((900, len(gamma)))
    rel = np.zeros(900)
    bet = np.zeros((900, len(gamma)))
    c2 = np.zeros(900)
    sptot = np.zeros((900, len(gamma)))

    # jmax = num photon energies
    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * min(tempbb)
    xmax = 40.0 * max(theta)
    jmax = min(899, int(np.log10(xmax / xmin) / delta + 1))

    # Energy array
    x = np.zeros(900)
    x[:jmax + 1] = xmin * 10**(np.arange(jmax + 1) * delta)

    # c2: relativistic correction to Kompaneets eqn
    # rel: Klein-Nishina / Thompson cross-section ratio
    w = np.zeros(900)
    w[:jmax] = x[:jmax]
    w1 = np.sqrt(x[:jmax] * x[1:jmax + 1])

    c2[:jmax] = (w1**4) / (1 + 4.6 * w1 + 1.1 * w1 * w1)

    # Asymptotic limit for x < 0.05
    rel[:jmax] = 1 - 2 * w[:jmax] + 26 * w[:jmax] * w[:jmax] / 5

    z1 = np.zeros(900)
    z2 = np.zeros(900)
    z3 = np.zeros(900)
    z4 = np.zeros(900)
    z5 = np.zeros(900)
    z6 = np.zeros(900)
    z1[:jmax] = (1 + w[:jmax]) / w[:jmax]**3
    z2[:jmax] = 1 + 2 * w[:jmax]
    z3[:jmax] = np.log(z2[:jmax])
    z4[:jmax] = 2 * w[:jmax] * (1 + w[:jmax]) / z2[:jmax]
    z5[:jmax] = z3[:jmax] / 2 / w[:jmax]
    z6[:jmax] = (1 + 3 * w[:jmax]) / z2[:jmax] / z2[:jmax]

    # Overwrite for x >= 0.05
    rel[w >= 0.05] = 0.75 * (z1[w >= 0.05] * (z4[w >= 0.05] - z3[w >= 0.05])
                              + z5[w >= 0.05] - z6[w >= 0.05])

    # Thermal seed spectrum
    jmaxth = min(900, int(np.log10(50 * max(tempbb) / xmin) / delta))
    if jmaxth > jmax:
        jmaxth = jmax

    planck = 15 / (np.pi * tempbb)**4
    dphdot[:jmaxth] = planck * x[:jmaxth, np.newaxis]**2
    dphdot[:jmaxth] /= (np.exp(x[:jmaxth, np.newaxis] / tempbb) - 1)

    # Beta: probability of escape per Thompson time
    jnr = min(int(np.log10(0.1 / xmin) / delta + 1), jmax - 1)
    jrel = min(int(np.log10(1 / xmin) / delta + 1), jmax)
    xnr = x[jnr - 1]
    xr = x[jrel - 1]

    taukn = tautom * rel[:, np.newaxis]
    bet[:jnr - 1] = 1 / tautom / (1 + taukn[:jnr - 1] / 3)

    flz = 1 - ((x - xnr) / (xr - xnr))
    bet[jnr - 1:jrel] = 1 / tautom / (1 + taukn[jnr - 1:jrel] / 3
                                        * flz[jnr - 1:jrel, np.newaxis])

    bet[jrel:jmax] = 1 / tautom

    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphesc, dphdot, bet, c2)

    sptot_tst = np.zeros((900, len(gamma)))
    for j in range(0, jmax - 1):
        sptot_tst[j] = dphesc[j] * x[j]**2

    sptot[:jmax - 1] = dphesc[:jmax - 1] * x[:jmax - 1, np.newaxis]**2

    return x, jmax, sptot_tst


def _thermlc(tautom, theta, deltal, x, jmax, dphesc, dphdot, bet, c2):
    """
    Adapted from the subroutine thermlc in donthcomp.f, included in Xspec.
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)

    Parameters
    ----------
    tautom : array
        Thompson scattering cross-sections.
    theta : array
        Plasma electron temperature, normalised to 511 keV.
    deltal : float
        10-log interval of photon array.
    x : array, shape(900,)
        Energy array, normalised to 511 keV.
    jmax : int
        Max index in energy array.
    dphesc : array
        Escaping photon density.
    dphdot : array
        Photon production rate (seed spectrum).
    bet : array
        Probability of escape per Thompson time.
    c2 : array
        Coefficients in Kompaneets equation.

    Returns
    -------
    dphesc : array
        Escaping photon density.
    """
    c20 = tautom / deltal

    w1 = np.sqrt(x[1:jmax - 1] * x[2:jmax])
    w2 = np.sqrt(x[0:jmax - 2] * x[1:jmax - 1])

    a = np.zeros((900, len(theta)))
    b = np.zeros((900, len(theta)))
    c = np.zeros((900, len(theta)))
    d = np.zeros((900, len(theta)))
    a[1:jmax - 1] = -c20 * c2[1:jmax - 1, np.newaxis] * (
        theta / deltal / w1[:, np.newaxis] + 0.5)

    t1 = -c20 * c2[1:jmax - 1, np.newaxis] * (
        0.5 - theta / deltal / w1[:, np.newaxis])
    t2 = c20 * c2[0:jmax - 2, np.newaxis] * (
        theta / deltal / w2[:, np.newaxis] + 0.5)
    t3 = x[1:jmax - 1, np.newaxis]**3 * (tautom * bet[1:jmax - 1])
    b[1:jmax - 1] = t1 + t2 + t3
    c[1:jmax - 1] = c20 * c2[0:jmax - 2, np.newaxis] * (
        0.5 - theta / deltal / w2[:, np.newaxis])
    d[1:jmax - 1] = x[1:jmax - 1, np.newaxis] * dphdot[1:jmax - 1]

    # Boundary terms
    x32 = np.sqrt(x[0] * x[1])
    aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

    uu = np.zeros((900, len(theta)))
    uu[jmax - 1] = 0.0

    # Invert tridiagonal matrix
    alp = np.zeros((900, len(theta)))
    gam = np.zeros((900, len(theta)))
    g = np.zeros((900, len(theta)))
    alp[1] = b[1] + c[1] * aa
    gam[1] = a[1] / alp[1]
    g[1] = d[1] / alp[1]

    for j in range(2, jmax - 1):
        alp[j] = b[j] - c[j] * gam[j - 1]
        gam[j] = a[j] / alp[j]

        if j != jmax - 2:
            g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]

    g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * uu[jmax]
                    - c[jmax - 2] * g[jmax - 3]) / alp[jmax - 2]
    uu[jmax - 2] = g[jmax - 2]
    for j in range(2, jmax):
        jj = jmax - j
        uu[jj] = g[jj] - gam[jj] * uu[jj + 1]

    uu[0] = aa * uu[1]

    dphesc = x[:, np.newaxis] * x[:, np.newaxis] * uu * bet * tautom
    dphesc[dphesc < 0] = 0
    return dphesc


# =========================================================================
# AGNSED  —  Full AGN SED model
# =========================================================================

class agnsed:

    Emin = 1e-4
    Emax = 1e4
    numE = 2000

    A = 0.3  # disc albedo

    dr_dex = 50  # grid spacing — N points per decade

    default_units = 'cgs'
    units = 'cgs'

    warnings.filterwarnings('ignore')

    def __init__(self,
                 M=1e8,
                 dist=100,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 kTe_hot=100,
                 kTe_warm=0.2,
                 gamma_hot=1.7,
                 gamma_warm=2.7,
                 r_hot=10,
                 r_warm=20,
                 log_rout=-1,
                 fcol=1,
                 h_max=10,
                 rep=True,
                 z=0):
        """
        Parameters
        ----------
        M : float
            Black hole mass — units : Msol
        dist : float
            Co-Moving Distance — units : Mpc
        log_mdot : float
            log mass accretion rate — units : Eddington
        a : float
            Dimensionless Black Hole spin
        cos_inc : float
            cos inclination angle
        kTe_hot : float
            Electron temp for hot Compton region — units : keV
        kTe_warm : float
            Electron temp for warm Compton region — units : keV
        gamma_hot : float
            Spectral index for hot Compton region
        gamma_warm : float
            Spectral index for warm Compton region
        r_hot : float
            Outer radius of hot Compton region — units : Rg
        r_warm : float
            Outer radius of warm Compton region — units : Rg
        log_rout : float
            log of outer disc radius — units : Rg
        fcol : float
            Colour temperature correction (Done et al. 2012)
        h_max : float
            Scale height of hot Compton region — units : Rg
        rep : bool
            Switch for X-ray re-processing off the disc
        z : float
            Redshift
        """
        # Read Pars
        self.M = M
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.kTe_h = kTe_hot
        self.kTe_w = kTe_warm
        self.gamma_h = gamma_hot
        self.gamma_w = gamma_warm
        self.r_h = r_hot
        self.r_w = r_warm
        self.r_out = 10**(log_rout)
        self.fcol = fcol
        self.hmax = h_max
        self.rep = rep
        self.z = z

        self.cosinc = cos_inc

        # Initiating constants
        self._set_constants()

        # Performing checks
        self._check_spin()
        self._check_inc()

        # Calculating Disc pars
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()

        if log_rout < 0:
            self.r_out = self.r_sg

        if r_warm == -1:
            self.r_w = self.risco

        if r_hot == -1:
            self.r_h = self.risco
            self.hmax = 0

        self._check_rw()
        self._check_risco()
        self._check_hmax()

        # Physical conversion factors
        self.Mdot_edd = self.L_edd / (self.eta * self.c**2)
        self.Rg = (self.G * self.M) / (self.c**2)

        # Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(
            u.Hz, equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(
            u.AA, equivalencies=u.spectral()).value

        self.E_obs = self.Egrid / (1 + self.z)
        self.nu_obs = self.nu_grid / (1 + self.z)
        self.wave_obs = self.wave_grid * (1 + self.z)

        # Radial grid over disc and warm compton regions
        self.dlog_r = 1 / self.dr_dex
        self.logr_ad_bins = self._make_rbins(
            np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(
            np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(
            np.log10(self.risco), np.log10(self.r_h))

        # X-ray power
        self._calc_Ldiss()
        self._calc_Lseed()
        self.Lx = self.Ldiss + self.Lseed

    def _set_constants(self):
        """Sets physical constants in cgs as object attributes."""
        self.G = (const.G * const.M_sun).to(u.cm**3 / u.s**2).value
        self.sigma_sb = const.sigma_sb.to(u.erg / u.s / u.K**4 / u.cm**2).value
        self.c = const.c.to(u.cm / u.s).value
        self.h = const.h.to(u.erg * u.s).value
        self.k_B = const.k_B.to(u.erg / u.K).value
        self.m_e = const.m_e.to(u.g).value

    # -----------------------------------------------------------------
    # Parameter checks
    # -----------------------------------------------------------------

    def _check_spin(self):
        if not (-0.998 <= self.a <= 0.998):
            raise ValueError('Spin ' + str(self.a) + ' not physical! \n'
                             'Must be within: -0.998 <= a_star <= 0.998')

    def _check_inc(self):
        if not (0.09 <= self.cosinc <= 1):
            raise ValueError('Inclination out of bounds! \n'
                             'Require: 0.09 <= cos(inc) <= 0.98 \n'
                             'Translates to: 11.5 <= inc <= 85 deg')

    def _check_rw(self):
        if self.r_w < self.r_h:
            print('WARNING r_warm < r_hot ---- Setting r_warm = r_hot')
            self.r_w = self.r_h
        if self.r_w > self.r_out:
            print('WARNING r_warm > r_out ----- Setting r_warm = r_out')
            self.r_w = self.r_out

    def _check_risco(self):
        if self.r_h < self.risco:
            print('WARNING r_hot < r_isco ----- Setting r_hot = r_isco')
            self.r_h = self.risco
            self.hmax = 0
        if self.r_w < self.risco:
            print('WARNING! r_warm < r_isco ----- Setting r_warm = r_isco')
            self.r_w = self.risco

    def _check_hmax(self):
        if self.hmax > self.r_h:
            print('WARNING! hmax > r_h ------- Setting hmax = r_h')
            self.hmax = self.r_h

    # -----------------------------------------------------------------
    # Unit handling (only affects output)
    # -----------------------------------------------------------------

    def set_units(self, new_unit='cgs'):
        """
        Re-sets default units. ONLY affects attributes extracted through
        the getter methods.

        Parameters
        ----------
        new_unit : {'cgs', 'cgs_wave', 'SI', 'counts'}
            The default unit to use.
        """
        unit_lst = ['cgs', 'cgs_wave', 'SI', 'counts']
        if new_unit not in unit_lst:
            print('Invalid Unit!!!')
            print(f'Valid options are: {unit_lst}')
            print('Setting as default: cgs')
            new_unit = 'cgs'
        self.units = new_unit

    def _to_newUnit(self, L, as_spec=True):
        """Sets input luminosity/spectrum to new output units."""
        if as_spec:
            if self.units == 'cgs':
                Lnew = L
            elif self.units == 'counts':
                Lnew = (L * u.erg / u.s / u.Hz).to(
                    u.keV / u.s / u.keV,
                    equivalencies=u.spectral()).value
                if np.ndim(L) == 1:
                    Lnew /= self.Egrid
                else:
                    Lnew /= self.Egrid[:, np.newaxis]
            elif self.units == 'cgs_wave':
                Lnew = (L * u.erg / u.s / u.Hz).to(
                    u.erg / u.s / u.AA,
                    equivalencies=u.spectral_density(self.nu_grid * u.Hz)).value
            else:
                Lnew = L * 1e-7
        else:
            if self.units in ('cgs', 'counts'):
                Lnew = L
            else:
                Lnew = L * 1e-7
        return Lnew

    def _to_flux(self, L):
        """Converts luminosity to flux (accounts for distance)."""
        if self.units in ('cgs', 'counts'):
            d = self.d
        else:
            d = self.d / 100
        return L / (4 * np.pi * d**2)

    # -----------------------------------------------------------------
    # Disc properties
    # -----------------------------------------------------------------

    def _calc_Ledd(self):
        self.L_edd = 1.39e38 * self.M

    def _calc_risco(self):
        """ISCO for spinning BH (Page & Thorne 1974)."""
        Z1 = 1 + (1 - self.a**2)**(1 / 3) * (
            (1 + self.a)**(1 / 3) + (1 - self.a)**(1 / 3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)
        self.risco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2 * Z2))

    def _calc_r_selfGravity(self):
        """Self-gravity radius (Laor & Netzer 1989), alpha=0.1."""
        alpha = 0.1
        m9 = self.M / 1e9
        self.r_sg = 2150 * m9**(-2 / 9) * self.mdot**(4 / 9) * alpha**(2 / 9)

    def _calc_efficiency(self):
        """Accretion efficiency eta (GR case)."""
        self.eta = 1 - np.sqrt(1 - 2 / (3 * self.risco))

    def _calc_NTparams(self, r):
        """Novikov-Thorne relativistic factors (Krolik; Page & Thorne 1974)."""
        y = np.sqrt(r)
        y_isc = np.sqrt(self.risco)
        y1 = 2 * np.cos((1 / 3) * np.arccos(self.a) - (np.pi / 3))
        y2 = 2 * np.cos((1 / 3) * np.arccos(self.a) + (np.pi / 3))
        y3 = -2 * np.cos((1 / 3) * np.arccos(self.a))

        B = 1 - (3 / r) + ((2 * self.a) / (r**(3 / 2)))

        C1 = 1 - (y_isc / y) - ((3 * self.a) / (2 * y)) * np.log(y / y_isc)

        C2 = ((3 * (y1 - self.a)**2) / (y * y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1) / (y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2) / (y * y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2) / (y_isc - y2))
        C2 += ((3 * (y3 - self.a)**2) / (y * y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3) / (y_isc - y3))

        C = C1 - C2
        return C / B

    def calc_Tnt(self, r):
        """
        Novikov-Thorne disc temperature^4 at radius r.

        Parameters
        ----------
        r : float or array
            Radius — Units : Rg

        Returns
        -------
        T4 : float or array
            NT temperature^4 — Units : K^4
        """
        Rt = self._calc_NTparams(r)
        const_fac = (3 * self.G * self.M * self.mdot * self.Mdot_edd) / (
            8 * np.pi * self.sigma_sb * (r * self.Rg)**3)
        return const_fac * Rt

    def calc_Trep(self, r, Lx):
        """
        Re-processed temperature at r for a given X-ray luminosity.

        Parameters
        ----------
        r : float or array
            Radial coordinate — units : Rg
        Lx : float or array
            X-ray luminosity — units : erg/s

        Returns
        -------
        T4rep : float or array
            Reprocessed temperature^4.
        """
        R = r * self.Rg
        H = self.hmax * self.Rg

        Frep = (Lx) / (4 * np.pi * (R**2 + H**2))
        Frep *= H / np.sqrt(R**2 + H**2)
        Frep *= (1 - self.A)

        return (Frep / self.sigma_sb) * (1 - self.A)

    def calc_fcol(self, Tm):
        """Colour temperature correction (Done et al. 2012)."""
        if Tm > 1e5:
            Tm_j = self.k_B * Tm
            Tm_keV = (Tm_j * u.erg).to(u.keV).value
            fcol_d = (72 / Tm_keV)**(1 / 9)
        elif Tm > 3e4:
            fcol_d = (Tm / (3e4))**(0.82)
        else:
            fcol_d = 1
        return fcol_d

    # -----------------------------------------------------------------
    # Radial binning
    # -----------------------------------------------------------------

    def _make_rbins(self, logr_in, logr_out, dlog_r=None):
        """
        Creates array of radial bin edges from r_out down to r_in.

        Parameters
        ----------
        logr_in : float
            Inner radius (log10 Rg).
        logr_out : float
            Outer radius (log10 Rg).

        Returns
        -------
        logr_bins : 1D-array
            Radial bin edges (log10 Rg).
        """
        if dlog_r is not None:
            dlr = dlog_r
        else:
            dlr = self.dlog_r

        i = logr_out
        logr_bins = np.array([np.float64(logr_out)])
        while i > logr_in:
            r_next_edge = i - dlr
            logr_bins = np.insert(logr_bins, 0, r_next_edge)
            i = r_next_edge

        if logr_bins[0] != logr_in:
            if logr_bins[0] < logr_in:
                if len(logr_bins) > 1:
                    logr_bins = np.delete(logr_bins, 0)
                    logr_bins[0] = logr_in
                else:
                    logr_bins[0] = logr_in
            else:
                logr_bins[0] = logr_in

        return logr_bins

    def new_ear(self, new_es):
        """Defines new energy grid."""
        Ebins = new_es
        dEs = self.Ebins[1:] - self.Ebins[:-1]

        self.Egrid = Ebins[:-1] + 0.5 * dEs
        self.nu_grid = (self.Egrid * u.keV).to(
            u.Hz, equivalencies=u.spectral()).value
        self.nu_obs = self.nu_grid / (1 + self.z)
        self.E_obs = self.Egrid / (1 + self.z)

        self.Emin = min(self.Egrid)
        self.Emax = max(self.Egrid)
        self.numE = len(self.Egrid)

    def set_radialResolution(self, Ndex):
        """Re-sets radial binning (debugging aide)."""
        self.dr_dex = Ndex
        self.__init__(self.M, self.D, np.log10(self.mdot), self.a,
                      self.cosinc, self.kTe_h, self.kTe_w, self.gamma_h,
                      self.gamma_w, self.r_h, self.r_w, np.log10(self.r_out),
                      self.fcol, self.hmax, self.rep, self.z)

    # -----------------------------------------------------------------
    # Standard disc emission
    # -----------------------------------------------------------------

    def _bb_radiance(self, T):
        """
        Black-body radiance for temperature T.

        Returns pi * Bnu : shape=(len(nus), len(Ts)) — Units : erg/s/cm^2/Hz
        """
        pre_fac = (2 * self.h * self.nu_grid[:, np.newaxis]**3) / (self.c**2)
        exp_fac = np.exp(
            (self.h * self.nu_grid[:, np.newaxis]) / (self.k_B * T)) - 1
        return np.pi * pre_fac / exp_fac

    def _do_disc_annuli(self, r, dr):
        """Emission from annuli in standard disc region."""
        T4_ann = self.calc_Tnt(r)
        if self.rep:
            T4_ann = T4_ann + self.calc_Trep(r, self.Lx)

        Tann = T4_ann**(1 / 4)
        if self.fcol < 0:
            fcol_r = self.calc_fcol(Tann)
        else:
            fcol_r = self.fcol

        Tann *= fcol_r
        bb_ann = self._bb_radiance(Tann) / (fcol_r**4)
        return 4 * np.pi * r * dr * bb_ann * self.Rg**2 * (self.cosinc / 0.5)

    def _do_discSpec(self):
        """Total SED from standard disc region."""
        dr_bins = 10**(self.logr_ad_bins[1:]) - 10**(self.logr_ad_bins[:-1])
        rmids = 10**(self.logr_ad_bins[:-1] + self.dlog_r / 2)

        if len(rmids) == 0:
            self.Lnu_disc = np.zeros(len(self.Egrid))
        else:
            Lnu_arr = self._do_disc_annuli(rmids, dr_bins)
            self.Lnu_disc = np.sum(Lnu_arr, axis=-1)
        return self.Lnu_disc

    # -----------------------------------------------------------------
    # Warm Compton emission
    # -----------------------------------------------------------------

    def _do_warm_annuli(self, r, dr):
        """Emission from annuli in warm Comptonisation region."""
        T4_ann = self.calc_Tnt(r)
        if self.rep:
            T4_ann = T4_ann + self.calc_Trep(r, self.Lx)

        kTann = self.k_B * T4_ann**(1 / 4)
        kTann = (kTann * u.erg).to(u.keV).value

        if not isinstance(kTann, np.ndarray):
            kTann = np.array([kTann])

        gamma_arr = np.full(len(kTann), self.gamma_w)
        kTe_arr = np.full(len(kTann), self.kTe_w)

        ph_nth = _mdonthcomp(self.Egrid, gamma_arr, kTe_arr, kTann)
        ph_nth = (ph_nth * u.erg / u.s / u.keV).to(
            u.erg / u.s / u.Hz, equivalencies=u.spectral()).value

        norm = (self.sigma_sb * T4_ann * 4 * np.pi * r * dr
                * self.Rg**2 * (self.cosinc / 0.5))
        radience = np.trapz(ph_nth, self.nu_grid, axis=0)

        return norm * (ph_nth / radience)

    def _do_warmSpec(self):
        """Total spectrum from warm Compton region."""
        dr_bins = 10**(self.logr_wc_bins[1:]) - 10**(self.logr_wc_bins[:-1])
        rmids = 10**(self.logr_wc_bins[:-1] + self.dlog_r / 2)

        if len(rmids) == 0:
            self.Lnu_warm = np.zeros(len(self.Egrid))
        else:
            Lnu_arr = self._do_warm_annuli(rmids, dr_bins)
            self.Lnu_warm = np.sum(Lnu_arr, axis=-1)
        return self.Lnu_warm

    # -----------------------------------------------------------------
    # Hot Compton emission
    # -----------------------------------------------------------------

    def _calc_Ldiss(self):
        """Total power dissipated within the corona (erg/s)."""
        dr_bins = 10**(self.logr_hc_bins[1:]) - 10**(self.logr_hc_bins[:-1])
        rmids = 10**(self.logr_hc_bins[:-1] + self.dlog_r / 2)
        T4s = self.calc_Tnt(rmids)
        self.Ldiss = np.sum(
            self.sigma_sb * T4s * 4 * np.pi * rmids * dr_bins * self.Rg**2)

    def _calc_Lseed(self):
        """Seed photon luminosity seen by the hot corona (erg/s)."""
        logr_tot_bins = self._make_rbins(
            np.log10(self.r_h), np.log10(self.r_out))
        hc = min(self.r_h, self.hmax)

        drs = 10**(logr_tot_bins[1:]) - 10**(logr_tot_bins[:-1])
        rmids = 10**(logr_tot_bins[:-1] + self.dlog_r / 2)

        # Covering fraction (Hagen & Done 2023b, MNRAS, 525, 3455)
        cov_frac = np.zeros(len(rmids))
        theta0 = np.zeros(len(rmids))
        theta0[rmids >= hc] = np.arcsin(hc / rmids[rmids >= hc])
        cov_frac[rmids >= hc] = theta0[rmids >= hc] - 0.5 * np.sin(
            2 * theta0[rmids >= hc])

        T4_ann = self.calc_Tnt(rmids)
        Fr = self.sigma_sb * T4_ann
        Lr = 4 * np.pi * rmids * drs * Fr * (cov_frac / np.pi) * self.Rg**2
        self.Lseed = np.sum(Lr)

    def kTseed_hot(self):
        """Seed photon temperature for hot Compton region (keV)."""
        T4_edge = self.calc_Tnt(self.r_h)
        Tedge = T4_edge**(1 / 4)

        kT_edge = self.k_B * Tedge
        kT_edge = (kT_edge * u.erg).to(u.keV).value

        if self.r_w != self.r_h:
            ysb = (self.gamma_w * (4 / 9))**(-4.5)
            kT_seed = np.exp(ysb) * kT_edge
        else:
            kT_seed = kT_edge
            if self.fcol < 0:
                fcol_r = self.calc_fcol(Tedge)
            else:
                fcol_r = self.fcol
            kT_seed *= fcol_r

        return kT_seed

    def _do_hotSpec(self):
        """Hot Compton spectrum (erg/s/Hz)."""
        kT_seed = self.kTseed_hot()
        if kT_seed == 0:
            self.Lnu_hot = np.zeros(len(self.Egrid))
        else:
            ph_nth = _mdonthcomp(
                self.Egrid, np.array([self.gamma_h]),
                np.array([self.kTe_h]), np.array([kT_seed]))[:, 0]
            ph_nth = (ph_nth * u.erg / u.s / u.keV).to(
                u.erg / u.s / u.Hz, equivalencies=u.spectral()).value

            radience = np.trapz(ph_nth, self.nu_grid)
            self.Lnu_hot = self.Lx * (ph_nth / radience)
        return self.Lnu_hot

    # -----------------------------------------------------------------
    # SED getters
    # -----------------------------------------------------------------

    def get_SED(self, as_flux=False):
        """Extract total SED in currently set units."""
        fLnu_dsc = self.get_SEDcomponent('disc', as_flux=as_flux)
        fLnu_wrm = self.get_SEDcomponent('warm', as_flux=as_flux)
        fLnu_hot = self.get_SEDcomponent('hot', as_flux=as_flux)
        return fLnu_dsc + fLnu_wrm + fLnu_hot

    def get_SEDcomponent(self, component, as_flux=False):
        """
        Extract SED component ('disc', 'warm', or 'hot').

        Parameters
        ----------
        component : {'disc', 'warm', 'hot'}
        as_flux : bool
            Convert to flux units (includes distance).
        """
        if component not in ('disc', 'warm', 'hot'):
            raise ValueError('component must be: disc, warm, or hot')

        if hasattr(self, f'Lnu_{component}'):
            Lnu_cmp = getattr(self, f'Lnu_{component}')
        else:
            Lnu_cmp = getattr(self, f'_do_{component}Spec')()

        fLnu_cmp = self._to_newUnit(Lnu_cmp, as_spec=True)
        if as_flux:
            fLnu_cmp = self._to_flux(fLnu_cmp)
        return fLnu_cmp


# =========================================================================
# QSOSED  —  Simplified model (inherits from agnsed)
# =========================================================================

class qsosed(agnsed):

    def __init__(self,
                 M=1e8,
                 dist=100,
                 log_mdot=-1,
                 a=0,
                 cos_inc=0.5,
                 fcol=1,
                 z=0):
        """
        Parameters
        ----------
        M : float
            Black hole mass — units : Msol
        dist : float
            Co-Moving Distance — units : Mpc
        log_mdot : float
            log mass accretion rate — units : Eddington
        a : float
            Dimensionless Black Hole spin
        cos_inc : float
            cos inclination angle
        fcol : float
            Colour temperature correction (Done et al. 2012)
        z : float
            Redshift
        """
        # Read params
        self.M = M
        self.D, self.d = dist, (dist * u.Mpc).to(u.cm).value
        self.mdot = 10**(log_mdot)
        self.a = np.float64(a)
        self.inc = np.arccos(cos_inc)
        self.cosinc = cos_inc
        self.fcol = fcol
        self.z = z

        # Fixed pars
        self.kTe_h = 100  # keV
        self.kTe_w = 0.2  # keV
        self.gamma_w = 2.5

        # Initiating constants
        self._set_constants()

        # Performing checks
        self._check_spin()
        self._check_inc()
        self._check_mdot()

        # Calculating disc params
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()

        # Physical conversion factors
        self.Mdot_edd = self.L_edd / (self.eta * self.c**2)
        self.Rg = (self.G * self.M) / (self.c**2)

        # Calculating disc regions
        self.dlog_r = 1 / self.dr_dex
        self._set_rhot()
        self.r_w = 2 * self.r_h
        self.r_out = self.r_sg
        self.hmax = min(100.0, self.r_h)

        if self.r_w > self.r_sg:
            self.r_w = self.r_sg

        # Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(
            u.Hz, equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(
            u.AA, equivalencies=u.spectral()).value

        self.E_obs = self.Egrid / (1 + self.z)
        self.nu_obs = self.nu_grid / (1 + self.z)
        self.wave_obs = self.wave_grid * (1 + self.z)

        # Radial grid
        self.dlog_r = 1 / self.dr_dex
        self.logr_ad_bins = self._make_rbins(
            np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(
            np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(
            np.log10(self.risco), np.log10(self.r_h))

        # X-ray power
        self._calc_Ldiss()
        self._calc_Lseed()
        self.rep = True
        self.Lx = self.Ldiss + self.Lseed

        self._set_gammah()

        # Re-initialise grids after gamma_h is set
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self.nu_grid = (self.Egrid * u.keV).to(
            u.Hz, equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(
            u.AA, equivalencies=u.spectral()).value

        self.E_obs = self.Egrid / (1 + self.z)
        self.nu_obs = self.nu_grid / (1 + self.z)
        self.wave_obs = self.wave_grid * (1 + self.z)

        self.dlog_r = 1 / self.dr_dex
        self.logr_ad_bins = self._make_rbins(
            np.log10(self.r_w), np.log10(self.r_out))
        self.logr_wc_bins = self._make_rbins(
            np.log10(self.r_h), np.log10(self.r_w))
        self.logr_hc_bins = self._make_rbins(
            np.log10(self.risco), np.log10(self.r_h))

    def _check_mdot(self):
        """Checks mdot is within bounds."""
        lmdot = np.log10(self.mdot)
        if lmdot >= -1.65 and lmdot <= 0.5:
            pass
        elif lmdot < -1.65:
            warnings.warn(
                f'mdot {self.mdot} (log={lmdot:.2f}) is very low (< -1.65). '
                'The model will overlap the hot corona over the entire disc.')
        else:
            raise ValueError('mdot is out of bounds! \n'
                             'Require: -1.65 <= log mdot <= 0.5')

    def _set_rhot(self):
        """
        Finds r_hot based on the condition L_diss = 0.02 Ledd.
        Uses a refined radial grid.
        """
        dlr = 1 / 1000
        log_rall = self._make_rbins(
            np.log10(self.risco), np.log10(self.r_sg), dlog_r=dlr)
        Ldiss = 0.0
        i = 0
        while Ldiss < 0.02 * self.L_edd and i < len(log_rall) - 1:
            rmid = 10**(log_rall[i] + dlr / 2)
            dr = 10**(log_rall[i + 1]) - 10**(log_rall[i])

            Tnt4 = self.calc_Tnt(rmid)
            Ldiss += self.sigma_sb * Tnt4 * 4 * np.pi * rmid * dr * self.Rg**2
            i += 1

        self.r_h = rmid

    def _set_gammah(self):
        """
        Sets spectral index of hot Compton region
        (Beloborodov 1999; Kubota & Done 2018).
        """
        if self.Lseed <= 0:
            self.gamma_h = 1.4
        else:
            self.gamma_h = (7 / 3) * (self.Ldiss / self.Lseed)**(-0.1)
            if self.gamma_h < 1.1:
                self.gamma_h = 1.1
