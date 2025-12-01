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
from scipy import integrate
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
    T = (sigma ** 2) / 2.
    x = x.T  # Shape: [nsteps, nsim]
    eta = eta.T  # Shape: [nsteps, nsim]
    nsim, nsteps = x.shape

    Cts = np.zeros((nsteps, nsteps))
    Rts = np.zeros((nsteps, nsteps))

    for i in range(nsim):
        Cts += np.outer(x[i], x[i])
        Rts += np.outer(x[i], eta[i])

    Cts /= nsim
    Rts /= (nsim * sigma**2)
    print("Cts shape:", Cts.shape)

    Its = np.zeros((nsteps, nsteps))
    Xts = np.zeros((nsteps, nsteps))

    for tt in range(nsteps):
        for ss in range(tt):
            tintaux = dt * np.arange(ss, tt + 1)
            Rintaux = Rts[tt, ss:tt + 1]
            intRaux = np.trapz(y=Rintaux, x=tintaux)

            Xts[tt, ss] = intRaux
            Its[tt, ss] = Cts[tt, tt] - Cts[tt, ss] - T * intRaux

    if normalize:
        Ctt = np.diag(Cts)
        I_norm2 = Its / (Ctt[:, None])
        X_norm2 = (sigma**2 * Xts) / (2 * Ctt[:, None])

        # ---------------------------------------------------------
        # STEADY-STATE ESTIMATE (correct version)
        # average t over last 20% AND s over early 20%
        # ---------------------------------------------------------
        t_late = slice(int(0.7 * nsteps), nsteps)   # t → "large"
        s_early = slice(0, int(0.2 * nsteps))       # s → "small"
        print(Its.shape)
        # Take average over t (rows) and s (cols)
        I_inf_vec      = np.mean(Its[t_late, :][:, s_early], axis=(0, 1))
        I_norm_inf_vec = np.mean(I_norm2[t_late, :][:, s_early], axis=(0, 1))
        X_norm_inf_vec = np.mean(X_norm2[t_late, :][:, s_early], axis=(0, 1))
        print(I_norm_inf_vec)

        return (
            I_inf_vec, I_norm_inf_vec, X_norm_inf_vec
        )

    return Cts, Rts, Its

# def _analysisFdt1(x, eta, sigma, dt):
#     T = sigma ** 2 / 2.

#     # Now, compute variables of interest
#     nsim, nsteps = x.shape
#     Cts = np.zeros((nsteps, nsteps))
#     Rts = np.zeros((nsteps, nsteps))

#     for i in range(nsim):
#         Cts += np.outer(x[i], x[i])
#         Rts += np.outer(x[i], eta[i])

#     # Ensemble average
#     Cts /= nsim
#     Rts /= (nsim * sigma ** 2)

#     # Calculates I(t,s) = C(t,t) - C(t,s) - T int_s^t R(t,s)
#     Its = np.zeros((nsteps, nsteps))
#     Xts = np.zeros((nsteps, nsteps))
#     for tt in range(nsteps):
#         for ss in range(tt):
#             tintaux = dt * np.arange(ss, tt + 1)
#             Rintaux = Rts[tt, ss:tt + 1]
#             intRaux = np.trapz(y=Rintaux, x=tintaux)
#             Xts[tt, ss] = intRaux
#             Its[tt, ss] = Cts[tt, tt] - Cts[tt, ss] - T * intRaux  # np.triu(A,1).sum()
#     Ctt = np.diag(Cts)
#     I_norm2 = Its / (Ctt[:, None])
#     X_norm2 = (sigma**2 * Xts) / (2 * Ctt[:, None])
#     return Its, I_norm2, X_norm2

from scipy.integrate import cumulative_trapezoid

def _analysisFdt1(x, eta, sigma, dt, lag_shift=0,return_time_matrix=True):
    T = sigma ** 2 / 2.0
    nsim, nsteps = x.shape

    # --- DIAGNOSTIC 1: CHECK INPUTS ---
    eta_std = np.std(eta)
    if abs(eta_std - 1.0) > 0.1:
        print(f"WARNING: eta.std() is {eta_std:.4f}. It should be ~1.0.")
        print("Did you pass dW instead of eta? If so, multiply eta by 1/sqrt(dt).")

    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    # --- COMPUTE MATRICES ---
    # Covariance
    Cts = (x_centered.T @ x_centered) / nsim
    Ctt = np.diag(Cts)

    # Response (Novikov)
    # Note: We calculate the raw correlation first to check alignment
    Rts_raw = (x_centered.T @ eta) / nsim
    
    # --- DIAGNOSTIC 2: CHECK ALIGNMENT ---
    # Check the diagonal mean vs the lower-diagonal mean
    diag_strength = np.mean(np.abs(np.diag(Rts_raw)))
    lag1_strength = np.mean(np.abs(np.diag(Rts_raw, k=-1)))
    
    print(f"Signal Strength on Diagonal (k=0): {diag_strength:.5f}")
    print(f"Signal Strength on Lag-1    (k=-1): {lag1_strength:.5f}")
    
    if diag_strength > lag1_strength and lag_shift == 0:
        print(">>> SUGGESTION: Your data is diagonal-aligned. Set lag_shift=0 in the code below.")
    
    # Normalize
    Rts = Rts_raw / (sigma * np.sqrt(dt))

    # ENFORCE CAUSALITY (Crucial Step)
    # k=-1 is standard Ito (x_t responds to eta_{t-1})
    # k=0  is needed if x_t and eta_t are row-aligned in your array
    k_val = -1 if lag_shift != 0 else 0
    Rts = np.tril(Rts, k=k_val)

    # Integration (Cumulative Trapezoid)
    R_cum_int = cumulative_trapezoid(Rts, dx=dt, axis=1, initial=0)
    diag_integrals = np.diag(R_cum_int)
    Xts = diag_integrals[:, None] - R_cum_int
    Xts = np.tril(Xts)

    # Calculation
    Its = Ctt[:, None] - Cts - (T * Xts)
    
    
    
    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        I_norm2 = Its / Ctt[:, None]
        X_norm2 = (sigma**2 * Xts) / (2 * Ctt[:, None])

    mask = np.tril(np.ones((nsteps, nsteps)), k=-20).astype(bool)

    
    filtered_values = I_norm2[mask]
    if return_time_matrix:
        I_full_time = np.full((nsteps, nsim, nsim), np.nan)
        for p in range(nsteps):
            xp = x_centered[:, p]
            etap = eta[:, p]

            # Covariance and response (time×time)
            C_tt = np.outer(xp, xp)
            R_tt = np.outer(xp, etap) / nsim
            R_tt /= (sigma * np.sqrt(dt))
            R_tt = np.tril(R_tt, k=k_val)

            # Cumulative integral
            R_cum_tt = cumulative_trapezoid(R_tt, dx=dt, axis=1, initial=0)
            diag_int_tt = np.diag(R_cum_tt)
            X_tt = diag_int_tt[:, None] - R_cum_tt
            X_tt = np.tril(X_tt)

            Tval = sigma**2 / 2.0
            Its_tt = np.mean(xp**2) - C_tt - (Tval * X_tt)  # per-parcel normalization
            I_norm_tt = Its_tt / np.mean(xp**2)

            I_full_time[p] = I_norm_tt


    # return original outputs and the new time matrix if requested
    if return_time_matrix:
        return filtered_values, I_norm2, Xts, I_full_time

    return filtered_values, I_norm2, X_norm2

def _analysisFdt3(x, eta, sigma, dt, normalize=True):
    # x shape: (T, P) = (197, 400)
    T_steps, P = x.shape
    Tval = sigma**2 / 2.
    
    I_inf_vec      = np.zeros(P)
    I_norm_inf_vec = np.zeros(P)
    X_norm_inf_vec = np.zeros(P)
    
    for p in range(P):
        xp = x[:, p]
        etap = eta[:, p]
        
        Cts = np.outer(xp, xp)
        Rts = np.outer(xp, etap) / (sigma**2)
        
        # I matrix and chi matrix
        Its = np.zeros((T_steps, T_steps))
        chi_ts = np.zeros((T_steps, T_steps))
        
        for tt in range(T_steps):
            for ss in range(tt + 1):  # Include ss=tt case
                # Integrate R(t,s) from s=ss to s=tt
                # chi(t,t') = int_{t'}^{t} R(t,s) ds
                tintaux = dt * np.arange(ss, tt + 1)
                Rintaux = Rts[tt, ss:tt+1]
                chi_ts[tt, ss] = np.trapz(y=Rintaux, x=tintaux)
                
                # I(t,t') = C(t,t) - C(t,t') - T*chi(t,t')
                Its[tt, ss] = Cts[tt, tt] - Cts[tt, ss] - Tval * chi_ts[tt, ss]
        
        if normalize:
            # Normalized quantities
            I_norm = np.zeros((T_steps, T_steps))
            chi_norm = np.zeros((T_steps, T_steps))
            
            for tt in range(T_steps):
                for ss in range(tt + 1):
                    # Normalize by C(t,t)
                    I_norm[tt, ss] = Its[tt, ss] / Cts[tt, tt]
                    # chi_bar(t,t') = T * chi(t,t') / C(t,t)
                    chi_norm[tt, ss] = Tval * chi_ts[tt, ss] / Cts[tt, tt]
            
            # Steady-state averages: t→∞ (late times), t'→0 (early times)
            I_inf_vec[p]      = np.mean(Its[-15:-1, 0:5])
            I_norm_inf_vec[p] = np.mean(I_norm[-15:-1, 0:5])
            X_norm_inf_vec[p] = np.mean(chi_norm[-15:-1, 0:5])
        else:
            # Without normalization
            I_inf_vec[p] = np.mean(Its[-15:-1, 0:5])
            # For non-normalized case, still compute chi averages
            I_norm_inf_vec[p] = 0.0
            X_norm_inf_vec[p] = np.mean(chi_ts[-15:-1, 0:5])
    
    return I_inf_vec, I_norm_inf_vec, X_norm_inf_vec


def _analysisFdt4(x, eta, sigma, dt, normalize=True,
                            min_variance=1e-10,
                            steady_state_start=0.6,
                            window_length=None,
                            max_lag=None):

    T_steps, P = x.shape
    # Tval (Effective Temperature) = Diffusion / Friction. 
    # For standard Langevin dx = -x dt + sigma dW: T = sigma^2 / 2
    Tval = sigma**2 / 2.0
    
    # Determine steady-state region
    t_start = int(T_steps * steady_state_start)
    xp_data = x[t_start:, :] # Slice once globally
    T_ss = xp_data.shape[0]

    # Auto-determine max_lag if None (ensure it isn't larger than data)
    if max_lag is None:
        max_lag = min(100, T_ss // 4)
    
    # Output arrays
    I_inf_vec = np.zeros(P)
    I_norm_inf_vec = np.zeros(P)
    X_norm_inf_vec = np.zeros(P)
    
    # Pre-calculate tau array for integration
    tau_array = np.arange(max_lag + 1) * dt

    diagnostics = {
        'parcels_with_outliers': []
    }

    for p in range(P):
        xp_ss = xp_data[:, p]
        
        # Check variance
        variance = np.var(xp_ss)
        if variance < min_variance:
            I_inf_vec[p] = np.nan
            I_norm_inf_vec[p] = np.nan
            X_norm_inf_vec[p] = np.nan
            continue

        # ===== 1. EFFICIENT COVARIANCE CALCULATION =====
        # Using numpy correlate is faster and less error-prone than manual loops
        # This calculates <x(t)x(t+tau)>
        full_corr = np.correlate(xp_ss - np.mean(xp_ss), xp_ss - np.mean(xp_ss), mode='full')
        # Extract only the positive lags [0, max_lag]
        mid = len(full_corr) // 2
        # Normalize by N to get Covariance (biased estimator is standard for time series)
        C_tau = full_corr[mid : mid + max_lag + 1] / T_ss
        
        C0 = C_tau[0]

        # Smooth covariance (Recommended to reduce derivative noise)
        if window_length is not None and window_length > 1:
            # Ensure window_length is odd and usually <= len(C_tau)
            wl = min(window_length | 1, len(C_tau) | 1) # Make odd
            if wl > 3:
                 C_tau = signal.savgol_filter(C_tau, window_length=wl, polyorder=3)

        # ===== 2. COMPUTE RESPONSE FUNCTION =====
        # FDT Relation: R(tau) = -(1/T) * dC/dtau * H(tau)
        
        # Compute Gradient (Central difference)
        # np.gradient handles boundaries automatically (2nd order accurate)
        dC_dtau = np.gradient(C_tau, dt)
        
        # FIX 1: Added negative sign here
        R_tau = -1.0 * (1.0 / Tval) * dC_dtau
        
        # ===== 3. COMPUTE SUSCEPTIBILITY =====
        # FIX 3: Stop integration if correlation drops below zero or becomes noise
        # Find first zero crossing of Covariance to stop integrating noise
        zero_crossings = np.where(C_tau < 0)[0]
        cutoff_idx = zero_crossings[0] if len(zero_crossings) > 0 else max_lag
        
        # Integrate R(tau) from 0 to cutoff
        if cutoff_idx > 1:
            chi_inf = integrate.trapezoid(R_tau[:cutoff_idx], tau_array[:cutoff_idx])
        else:
            chi_inf = 0.0

        # ===== 4. INTEGRAL VIOLATION =====
        # I = C(0) - T * chi_inf (Assuming C(infinity) -> 0)
        I_inf = C0 - (Tval * chi_inf)
        
        # ===== 5. NORMALIZE =====
        if normalize and C0 > min_variance:
            I_norm_inf = I_inf / C0
            chi_norm_inf = (Tval * chi_inf) / C0
            
            # Theoretical check: I_norm + chi_norm should be 1.0
            if np.abs((I_norm_inf + chi_norm_inf) - 1.0) > 0.5:
                 diagnostics['parcels_with_outliers'].append(p)
            
            I_inf_vec[p] = I_inf
            I_norm_inf_vec[p] = I_norm_inf
            X_norm_inf_vec[p] = chi_norm_inf
        else:
            I_inf_vec[p] = I_inf
            I_norm_inf_vec[p] = np.nan
            X_norm_inf_vec[p] = chi_norm_inf

    return I_inf_vec, I_norm_inf_vec, X_norm_inf_vec

def _computeDistanceFromEquilibrium(I):
    """
    Computes average deviation from FDT equilibrium per region.

    NEW (minimal change):
    - If I is a vector of steady-state values (length NPARCELS),
      convert it to an N×N difference matrix.
    """

    I = np.asarray(I)

    # ---- Minimal new addition ----
    if I.ndim == 1:
        # Build parcel × parcel matrix of pairwise deviations
        I = np.abs(I[:, None] - I[None, :])

    # ---- Original logic below ----
    absI = np.abs(I)
    diag_mask = np.identity(absI.shape[0], dtype=bool)
    Imask = np.ma.array(absI, mask=diag_mask)
    intI = np.average(Imask, axis=1)

    return intI




   