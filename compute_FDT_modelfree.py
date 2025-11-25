import os
import numpy as np
from scipy import signal
from src.data_loaders import ADNI_A, ADNI_B, load_data_records
from src.functions_frameworks.functions_FDT_modelfree import _splitSignal, _analysisFdt2, _computeDistanceFromEquilibrium
from src.data_processing.zfilterts import filter_time_series

def detrend_data(data, detrend=True):
    from scipy.signal import detrend as scipy_detrend
    data = np.asarray(data)
    # Apply detrending if requested
    if detrend:
        if data.ndim == 2:
            # [NPARCELS, NTIMES]
            data = scipy_detrend(data, axis=1, type='linear')
        elif data.ndim == 3:
            # [NSUB, NPARCELS, NTIMES]
            # Detrend along the time axis (axis=2) for each parcel of each subject
            data = scipy_detrend(data, axis=2, type='linear')
        else:
            raise ValueError("Input data must be 2D or 3D for detrending.")
    return data
def filter_time_series(ts_array, TR=3.0, low_pass_freq=0.008, high_pass_freq=0.08):
    """
    Preprocessing for model-free FDT, implementing a band-stop filter
    to remove frequencies between low_pass_freq and high_pass_freq.
    
    Parameters
    ----------
    ts_array : ndarray
        Shape (n_parcels, n_timepoints)
    TR : float
        Repetition time in seconds
    low_pass_freq : float
        Lower cutoff frequency for the band-stop filter (Hz)
    high_pass_freq : float
        Upper cutoff frequency for the band-stop filter (Hz)
    """
    from scipy.signal import butter, filtfilt
    
    nyquist = 1.0 / (2 * TR)
    
    # --- 1. Normalize the cutoff frequencies ---
    # The normalized frequencies must be between 0 and 1.
    Wn_low = low_pass_freq / nyquist
    Wn_high = high_pass_freq / nyquist
    
    # The band-stop filter requires a 2-element array for the cutoff
    Wn = [Wn_low, Wn_high]

    # --- 2. Validation ---
    if Wn_low <= 0 or Wn_high >= 1 or Wn_low >= Wn_high:
        raise ValueError(
            f"Invalid normalized frequency range: {[Wn_low, Wn_high]}\n"
            f"Frequencies must be 0 < low < high < {nyquist:.4f} Hz."
        )

    # --- 3. Band-Stop Filter Implementation ---
    # btype='bandstop' removes the frequencies *between* Wn_low and Wn_high.
    b, a = butter(2, Wn, btype='bandstop') 
    
    # Apply the filter using filtfilt for zero-phase distortion
    ts_filtered = filtfilt(b, a, ts_array, axis=-1, padlen=150)
    
    return ts_filtered




# ====================================================================
# NEW FUNCTION FOR ASYMPTOTIC NORMALIZED VIOLATION (Matches your LaTeX)
# ====================================================================

def _analysisFdt_AsymptoticNorm(Cts, Xts, sigma, n_burn_in=None):
    """
    Computes the normalized integrated violation I^(infinity) / C^infinity
    as defined by the LaTeX equation: 1 - T * (chi^infinity / C^infinity).
    
    Args:
        Cts (array): Empirical C(t,s) matrix.
        Xts (array): Empirical X(t,s) matrix (integrated response).
        sigma (float): Noise standard deviation.
        n_burn_in (int): Number of initial time steps to discard for steady-state.
    
    Returns:
        float: The scalar value of the normalized asymptotic violation.
    """
    T = (sigma ** 2) / 2.
    nsteps = Cts.shape[0]
    
    if n_burn_in is None:
        # Default burn-in to discard the first third of data
        n_burn_in = nsteps // 3 
    
    # 1. Steady-state Variance (C^infinity): C(t,t) averaged over steady-state time
    # This is the average of the diagonal C(t,t) after burn-in.
    C_tt = np.diag(Cts) 
    C_inf = np.mean(C_tt[n_burn_in:])
    
    # 2. Asymptotic Susceptibility (chi^infinity): X(t,0) integrated up to large t
    # This is the first column of Xts (X(t, s=0)) averaged over steady-state time.
    X_t_0 = Xts[:, 0]
    Chi_inf = np.mean(X_t_0[n_burn_in:])
    
    # 3. Normalized Asymptotic Violation (I^(infinity) / C^infinity)
    # I_norm_inf = 1 - T * (Chi_inf / C_inf)
    if C_inf == 0.0:
        return np.nan # Avoid division by zero
    
    I_norm_inf = 1.0 - T * (Chi_inf / C_inf)
    
    return I_norm_inf

# Example Usage: (Assuming you run the simulation and split the signal)
# C, R, I, I_norm2, X_norm2 = _analysisFdt2(x, eta, sigma, dt, normalize=True)
# I_norm_inf = _analysisFdt_AsymptoticNorm(C, X_norm2, sigma)
# print(f"Asymptotic Normalized FDT Violation (I-bar-infinity): {I_norm_inf}")


DL_type = 'DL_B2'
filt = True
TR = 3

if DL_type == 'DL_A': 
    DL = ADNI_A.ADNI_A(normalizeBurden=False)
    NPARCELLS = 379
if DL_type == 'DL_B1': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'AD'],filt=filt)
    NPARCELLS = 400  
if DL_type == 'DL_B2': 
    DL = ADNI_B.ADNI_B_Alt(['HC', 'MCI(AB-)', 'MCI(AB+)', 'AD'],filt=filt)
    NPARCELLS = 400

all_data = []
ts = np.array([])
for group in DL.get_groupLabels():
    all_data.extend(load_data_records.load_group_data(DL, group, DL_type))
    
results = []
for i in range(len(all_data)):
    all_data[i]['MRI'][:,:NPARCELLS] = detrend_data(all_data[i]['MRI'][:,:NPARCELLS])
    #all_data[i]['MRI'][:,:NPARCELLS] = filter_time_series(all_data[i]['MRI'][:,:NPARCELLS])
    x, dxdt, Fx, eta = _splitSignal(all_data[i]['MRI'][:,:NPARCELLS])
    dt = TR / 1000.
    sigma = np.std(eta)
    C, R, I, I_norm2, X_norm2 = _analysisFdt2(x, eta, sigma, dt)
    I_norm_2 = _analysisFdt_AsymptoticNorm(C, X_norm2, sigma)
    intI = _computeDistanceFromEquilibrium(I_norm2)
    intX = _computeDistanceFromEquilibrium(X_norm2)
    results.append({
        'subject_id': all_data[i]['subject_id'],
        'group': all_data[i]['group'],
        'sigma': sigma,
        'I_norm2': intI,
        'X_norm2': intX,
    })

results_dict = {key: np.array([d[key] for d in results])
        for key in ['subject_id', 'group', 'I_norm2', 'X_norm2', 'sigma']}

repo_root = os.getcwd()
save_path = os.path.join(repo_root, "data", "FDT_DATA")
os.makedirs(save_path, exist_ok=True)
filename = f"FDT_results_{DL_type}_N{NPARCELLS}_modelfree.npz"

np.savez_compressed(f"{save_path}/{filename}", **results_dict)









    

    # what we want: I and X norm 2. 
    # not sure how to go from what we find here to that, as norm 2 seems to involve noise.
    # after implementation, it is important that this becomes clear and is consistent with the model-based version.