"""
--------------------------------------------------------------------------------------
Model-free FDT analysis for fMRI signals

This model-free formulation is described in:
[Patow 2024] Off-Equilibrium Fluctuation-Dissipation Theorem Paves the Way in Alzheimer’s Disease Research
   Gustavo Patow, Juan Monti, Irene Acero-Pousa, Sebastián Idesis, Anira Escrichs, Yonatan Sanz Perl,
   Petra Ritter, Morten Kringelbach, Gustavo Deco, the Alzheimer’s Disease Neuroimaging Initiative
   bioRxiv, doi: https://doi.org/10.1101/2024.09.15.613131

Initial code by Gustavo Patow, causality and time lag additions by Melle Bolding
--------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid

def _noiseFilter(fMRI):
    """
    Filter the signal
    :param fMRI: the fMRI signal to process.
    :return: the filtered version of the signal
    """
    N, T = fMRI.shape
    signal_filt = np.zeros_like(fMRI)
    for n in range(N):
        # Apply a Wiener filter to the N-dimensional array fMRI[n].
        signal_filt[n] = signal.wiener(fMRI[n], 3)  

    return signal_filt

def _derivative2D(regionSignals, axis=0):
    """
    Derive a function
    :param regionSignals: the signals of the regions ;-)
    :param axis: the axis to apply the derivative to
    :return: the derivative of the region signals
    """
    if axis == 1:
        regionSignals = regionSignals.T
    N, T = regionSignals.shape
    d = np.zeros_like(regionSignals)
    for n in range(N):
        # d[n] = TVRegDiff(regionSignals[n], 10, 100, plotflag=False)
        d[n] = np.gradient(regionSignals[n])
    return d

def _derivative( allSignals, axis=0):
    if allSignals.ndim == 3:
        d = np.zeros_like(allSignals)
        S, N, T = allSignals.shape
        for s in range(S):
            d[s] = _derivative2D(allSignals[s], axis=axis)
    else:
        d = _derivative2D(allSignals, axis=axis)
    return d

def _splitSignal(fMRI): # MUST include 'self' since it's an instance method
        """
        Split the signal into main + noise!!!
        :param fMRI:
        :return: The input fMRI split into the signal x,
                its derivative dxdt,
                the (non-linear) function Fx,
                and the noise eta
        """
        # What we want is to decompose the signal so that
        #     dx/dt = -F[x(t)] + eta(t)
        # Let's say that our fMRI signal is x.
        x = fMRI
        
        # FIX 1: Call _derivative using the class method
        dxdt = _derivative(x) 
        
        # Now, we will decompose dxdt into F[x(t)] and eta(t)
        # For this, let's filter our derivative... (here we write mFx = -F[x(t)])
        mFx = _noiseFilter(dxdt) # Call _noiseFilter using the instance
        
        Fx = -1 * mFx  # Observe we return Fx, not mFx !!!
        # Finally, the difference signal: it is the subtraction of the filtered signal to the real signal, all WRT the derivative.
        # That is, eta(t) = dx/dt - -F[x(t)]
        eta = dxdt - mFx
        return x, dxdt, Fx, eta

def _analysisFdt1(x, eta, sigma, dt, lag_shift=0):
    T = sigma ** 2 / 2.0
    nsim, nsteps = x.shape

    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    # Covariance
    Cts = (x_centered.T @ x_centered) / nsim
    Ctt = np.diag(Cts)

    # Response (Novikov)
    Rts_raw = (x_centered.T @ eta) / nsim
 
    # Normalize
    Rts = Rts_raw / (sigma * sigma)

    # ENFORCE CAUSALITY: R(t,s)=0 if s>t
    # k=-1 is standard Ito (x_t responds to eta_{t-1})
    # k=0  is needed if x_t and eta_t are row-aligned in your array
    k_val = -1 if lag_shift != 0 else 0
    Rts = np.tril(Rts, k=k_val)

    # Integration (Cumulative Trapezoid)
    R_cum_int = cumulative_trapezoid(Rts, dx=dt, axis=1, initial=0)
    diag_integrals = np.diag(R_cum_int)
    Xts = diag_integrals[:, None] - R_cum_int
    Xts = np.tril(Xts)

    # Compute I = C(t,t) - C(t,s) - T * X(t,s)
    Its = Ctt[:, None] - Cts - (T * Xts)
    
    # normalize by variance
    with np.errstate(divide='ignore', invalid='ignore'):
        I_norm2 = Its / Ctt[:, None]
        X_norm2 = (sigma**2 * Xts) / (2 * Ctt[:, None])

    return I_norm2, X_norm2

def _computeDistanceFromEquilibrium(I):
    """
    Computes average deviation from FDT equilibrium per region.

    """
    I = np.asarray(I)

    if I.ndim == 1:
        print('one')
        I = np.abs(I[:, None] - I[None, :])

    diag_mask = np.identity(I.shape[0], dtype=bool)
    #Imask = np.ma.array(np.ones(I.shape[0])-np.abs(I)<0, mask=diag_mask)
    Imask = np.ma.array(np.ones(I.shape[0])-np.abs(I), mask=diag_mask)
    intI = np.average(Imask, axis=1)

    return intI


# --------------------------------------------------------------------------------------
# Promising option for future (paste at the end of _analysisFdt1):
# ensuring steady-state dynamics by introducing time lag
# mask = np.tril(np.ones((nsteps, nsteps)), k=-20).astype(bool)
# filtered_values = I_norm2[mask]