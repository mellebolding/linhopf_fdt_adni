# --------------------------------------------------------------------------------------
# Model-free FDT analysis for fMRI signals
#
# This model-free formulation is described in:
# [Patow 2024] Off-Equilibrium Fluctuation-Dissipation Theorem Paves the Way in Alzheimer’s Disease Research
#    Gustavo Patow, Juan Monti, Irene Acero-Pousa, Sebastián Idesis, Anira Escrichs, Yonatan Sanz Perl,
#    Petra Ritter, Morten Kringelbach, Gustavo Deco, the Alzheimer’s Disease Neuroimaging Initiative
#    bioRxiv, doi: https://doi.org/10.1101/2024.09.15.613131
#
# Code by Gustavo Patow
# --------------------------------------------------------------------------------------
import numpy as np
import scipy.signal as signal


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

def _analysisFdt2(x, eta, sigma, dt, normalize=True):
    """
    Main equilibrium (FDT) analysis
    This is a more efficient version than the old FDTAnalysis(x, eta, dt)
    :param x: the signal (rois, timepoints)
    :param eta: the noise (rois, timepoints)
    :param sigma: the noise variance
    :param dt: the time step
    :param normalize: if True, return normalized I and X matrices
    :return: the three matrices Cts, Rts, Its (and optionally I_norm2, X_norm2)
    """
    T = (sigma ** 2) / 2.
    nsim, nsteps = x.shape
    Cts = np.zeros((nsteps, nsteps))
    Rts = np.zeros((nsteps, nsteps))
    for i in range(nsim):
        Cts += np.outer(x[i], x[i])
        Rts += np.outer(x[i], eta[i])
    # Ensemble average
    Cts /= nsim
    Rts /= (nsim * sigma ** 2)
    
    # Calculates I(t,s) = C(t,t) - C(t,s) - T * X(t,s)
    # where X(t,s) = int_s^t R(t,s') ds'
    Its = np.zeros((nsteps, nsteps))
    Xts = np.zeros((nsteps, nsteps))  # Store the integrated response
    
    for tt in range(nsteps):
        for ss in range(tt):
            tintaux = dt * np.arange(ss, tt + 1)
            Rintaux = Rts[tt, ss:tt + 1]
            intRaux = np.trapz(y=Rintaux, x=tintaux)
            
            Xts[tt, ss] = intRaux  # Store X(t,s)
            Its[tt, ss] = Cts[tt, tt] - Cts[tt, ss] - T * intRaux
    
    if normalize:
        # Get diagonal of Cts: C(t,t) for each timepoint
        Ctt = np.diag(Cts)  # Shape: (nsteps,)
        
        # Normalize I: I_norm2(t,s) = I(t,s) / C(t,t)
        I_norm2 = Its / (Ctt[:, np.newaxis])
        
        # Normalize X: X_norm2(t,s) = sigma^2 * X(t,s) / (2*C(t,t))
        X_norm2 = (sigma ** 2 * Xts) / (2 * Ctt[:, np.newaxis])
        
        return Cts, Rts, Its, I_norm2, X_norm2
    else:
        return Cts, Rts, Its

def _computeDistanceFromEquilibrium(I):
        """
        Calculates the average absolute distance from FDT equilibrium for each region (node).
        
        :param I: The Integral Violation of the FDT matrix (Shape NPARCELS x NPARCELS).
        :return: An array where each element is the average deviation for a region (Shape NPARCELS,).
        """
        # print(I.shape) # Keep this for debugging if necessary
        absI = np.abs(I)
        # Create a mask for the diagonal (self-interaction)
        diag_mask = np.identity(absI.shape[0], dtype=bool)
        Imask = np.ma.array(absI, mask=diag_mask)
        
        # Calculate the average across the interaction dimension (axis=1) 
        # to get one value per region (row).
        intI = np.average(Imask, axis=1)
        return intI




   